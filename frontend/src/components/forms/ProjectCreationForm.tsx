import { omit, random } from 'lodash';
import { FC, useEffect, useState } from 'react';
import DataTable from 'react-data-table-component';
import { Controller, SubmitHandler, useForm, useWatch } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import Select from 'react-select';
import { UploadProgressBar } from '../UploadProgressBar';

import { CanceledError } from 'axios';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { Tooltip } from 'react-tooltip';
import {
  getProjectStatus,
  useAddFeature,
  useAddProjectFile,
  useCopyExistingData,
  useCreateProject,
  useGetAvailableDatasets,
  useProjectNameAvailable,
} from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { useAppContext } from '../../core/useAppContext';
import { getRandomName, loadFile } from '../../core/utils';
import { ProjectModel } from '../../types';

// format of the data table
export interface DataType {
  headers: string[];
  data: Record<string, string | number | bigint>[];
  filename: string;
}

type Option = {
  value: string;
  label: string;
};

// component
export const ProjectCreationForm: FC = () => {
  const { resetContext } = useAppContext();

  // form management
  const maxSizeMB = 400;
  const maxSize = maxSizeMB * 1024 * 1024; // 100 MB in bytes

  const maxTrainSet = 100000;
  const langages = [
    { value: 'en', label: 'English' },
    { value: 'fr', label: 'French' },
    { value: 'es', label: 'Spanish' },
    { value: 'de', label: 'German' },
    { value: 'cn', label: 'Chinese' },
    { value: 'ja', label: 'Japanese' },
  ];

  const { register, control, handleSubmit, setValue, reset } = useForm<
    ProjectModel & { files: FileList }
  >({
    defaultValues: {
      project_name: getRandomName('Project'),
      n_train: 100,
      n_test: 0,
      n_valid: 0,
      language: 'en',
      clear_test: false,
      random_selection: true,
      seed: random(0, 10000),
      force_label: false,
    },
  });
  const { notify } = useNotifications();
  const { datasets } = useGetAvailableDatasets(true); // Include toy datasets

  const [creatingProject, setCreatingProject] = useState<boolean>(false); // state for the data
  const [dataset, setDataset] = useState<string | null>(null); // state for the data
  const [data, setData] = useState<DataType | null>(null); // state for the data
  const [computeFeatures, setComputeFeatures] = useState<boolean>(true);
  const navigate = useNavigate(); // rooting
  const createProject = useCreateProject(); // API call
  const availableProjectName = useProjectNameAvailable(); // check if the project name is available
  const addFeature = useAddFeature();

  const { addProjectFile, progression, cancel } = useAddProjectFile(); // API call
  const copyExistingData = useCopyExistingData();
  const files = useWatch({ control, name: 'files' }); // watch the files entry
  const force_label = useWatch({ control, name: 'force_label' }); // watch the force label entry

  // available columns to select, depending of the source
  const [availableFields, setAvailableFields] = useState<Option[] | undefined>(undefined);
  const [columns, setColumns] = useState<string[]>([]);
  const [lengthData, setLengthData] = useState<number>(0);
  useEffect(() => {
    // case of loading external file
    if (dataset === 'load' && data) {
      setAvailableFields(
        data?.headers.filter((h) => h !== '').map((e) => ({ value: e, label: e })),
      );
      setColumns(data.headers);
      setLengthData(data.data.length - 1);
      // case of existing project
    } else if (dataset !== 'load' && datasets) {
      const element =
        dataset?.startsWith('-toy-dataset-') && datasets.toy_datasets
          ? datasets.toy_datasets.find((e) => `-toy-dataset-${e.project_slug}` === dataset)
          : datasets.projects.find((e) => e.project_slug === dataset);

      setAvailableFields(
        element?.columns.filter((h) => h !== '').map((e) => ({ value: e, label: e })),
      );
      setColumns(element?.columns || []);
      setLengthData(element?.n_rows ?? 0);
    } else {
      setAvailableFields(undefined);
      setColumns([]);
      setLengthData(0);
    }
  }, [data, dataset, datasets]);

  // select the text on input on click
  const handleClickOnText = (event: React.MouseEvent<HTMLInputElement>) => {
    const target = event.target as HTMLInputElement;
    target.select(); // Select the content of the input
  };

  // convert paquet file in csv if needed when event on files
  useEffect(() => {
    if (files && files.length > 0) {
      const file = files[0];
      if (file.size > maxSize) {
        notify({
          type: 'error',
          message: `File is too large (maximum size: ${maxSizeMB} MB)`,
        });
        return;
      }
      loadFile(file).then((data) => {
        if (data === null) {
          notify({ type: 'error', message: 'Error reading the file.' });
          return;
        }
        setData(data);
        setValue('n_train', Math.min((data?.data.length || 1) - 1, 100));
      });
    }
  }, [files, maxSize, notify, setValue]);

  // action when form validated
  const onSubmit: SubmitHandler<ProjectModel & { files?: FileList }> = async (formData) => {
    if (data || dataset !== 'load') {
      // check the form
      if (formData.project_name === '') {
        notify({ type: 'error', message: 'Enter a project name.' });
        return;
      }
      if (formData.col_id == '') {
        notify({ type: 'error', message: 'Select a column for ID.' });
        return;
      }
      if (!formData.cols_text) {
        notify({ type: 'error', message: 'Select a column for text.' });
        return;
      }
      if (
        Number(formData.n_train) + Number(formData.n_test) + Number(formData.n_valid) >
        lengthData
      ) {
        notify({
          type: 'warning',
          message: 'You requested larger samples than your dataset, sizes were adjusted.',
        });
        setValue(
          'n_train',
          Math.max(0, lengthData - Number(formData.n_test) - Number(formData.n_valid)),
        );
        return;
      }
      // test if the project name is available
      const available = await availableProjectName(formData.project_name);
      if (!available) {
        notify({ type: 'error', message: 'Project name already taken. Enter a new one.' });
        return;
      }

      try {
        setCreatingProject(true);

        // manage the files
        // case there is data to send
        if (dataset === 'load' && files && files.length > 0) {
          await addProjectFile(formData.project_name, files[0]);
        }
        // case to use a project existing
        else if (dataset !== 'load' && dataset) {
          const from_toy_dataset = dataset.startsWith('-toy-dataset-');
          const source_project = from_toy_dataset ? dataset.slice(13) : dataset; // if from toy dataset remove prefix
          await copyExistingData(formData.project_name, source_project, from_toy_dataset);
        } else {
          notify({ type: 'error', message: 'Unknown dataset.' });
          throw new Error('Unknown dataset');
        }

        // launch the project creation (which can take a while)
        const slug = await createProject({
          ...omit(formData, 'files'),
          filename: data ? data.filename : null,
          from_project: dataset == 'load' ? null : dataset,
          from_toy_dataset: dataset.startsWith('-toy-dataset-'),
        });

        // create a limit for waiting the project creation
        const maxDuration = 5 * 60 * 1000; // 5 minutes in milliseconds
        const startTime = Date.now();
        // wait until the project is really available
        const intervalId = setInterval(async () => {
          try {
            // watch the status of the project
            const status = await getProjectStatus(slug);
            console.log('Project status:', status);

            // if an error happened or the process failed
            if (status?.startsWith('error') || status === 'not existing') {
              clearInterval(intervalId);
              const errorDetail = status?.startsWith('error:')
                ? status.slice('error:'.length).trim()
                : null;
              notify({
                type: 'error',
                message: errorDetail
                  ? `Project creation failed: ${errorDetail}`
                  : 'Project creation failed. Try to change the data format. Contact support if the error persits.',
              });
              navigate(`/projects`);
              return;
            }

            // if the project has been created
            if (status === 'existing') {
              clearInterval(intervalId);
              if (computeFeatures)
                addFeature(slug, 'sentence-embeddings', 'sentence-embeddings', { model: 'generic' });
              resetContext();
              navigate(`/projects/${slug}?fromCreatePage=true`);
              return;
            }

            // set a timeout just in case to abort the waiting
            const elapsedTime = Date.now() - startTime;
            if (elapsedTime >= maxDuration) {
              clearInterval(intervalId);
              notify({
                type: 'error',
                message: 'Timeout during the creation of the project. Try again later.',
              });
              navigate(`/projects`);
              return;
            }
          } catch (error) {
            console.error('Error fetching projects:', error);
            clearInterval(intervalId);
          }
        }, 1000);
      } catch (error) {
        setCreatingProject(false);
        if (!(error instanceof CanceledError)) notify({ type: 'error', message: error + '' });
        else notify({ type: 'success', message: 'Project creation aborted.' });
      }
    }
  };

  useEffect(() => {
    reset({
      col_id: 'row_number',
      cols_text: [],
      cols_context: [],
      cols_label: [],
      n_train: 100,
      n_test: 0,
      n_valid: 0,
      language: 'en',
      clear_test: false,
      random_selection: true,
      force_label: false,
      seed: random(0, 10000),
    });
    setData(null);
    // reset data when changing dataset
  }, [dataset, reset]);

  return (
    <div>
      <div className="explanations">Create a new project</div>

      <form onSubmit={handleSubmit(onSubmit)}>
        <label htmlFor="project_name">Project name</label>
        <input
          id="project_name"
          placeholder="Name of the project (need to be unique in the system)"
          type="text"
          disabled={creatingProject}
          {...register('project_name')}
          onClick={handleClickOnText}
        />

        <div className="my-3">
          <label style={{ cursor: 'pointer' }}>
            <input type="radio" name="dataset-origin" onClick={() => setDataset('load')} />
            Load Dataset from disk
          </label>
          <br />
          <label style={{ cursor: 'pointer' }}>
            <input type="radio" name="dataset-origin" onClick={() => setDataset('from-project')} />
            Load Dataset from another project
          </label>
        </div>

        {dataset && (
          <label>
            Dataset{' '}
            {dataset !== 'load' && (
              <select
                id="existingDataset"
                value={dataset}
                onChange={(e) => {
                  setDataset(e.target.value);
                }}
              >
                <option key="from-project" value="from-project"></option>
                <optgroup label="Select project">
                  {(datasets?.projects || []).map((d) => (
                    <option key={d.project_slug} value={d.project_slug}>
                      {d.project_slug}
                    </option>
                  ))}
                </optgroup>
                {datasets?.toy_datasets && datasets?.toy_datasets?.length > 0 && (
                  <optgroup label="Select toy dataset">
                    {(datasets?.toy_datasets || []).map((d) => (
                      <option
                        key={`-toy-dataset-${d.project_slug}`}
                        value={`-toy-dataset-${d.project_slug}`}
                      >
                        {d.project_slug}
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>
            )}
            {dataset === 'load' && (
              <>
                <input
                  className="form-control"
                  disabled={creatingProject}
                  id="csvFile"
                  type="file"
                  {...register('files')}
                />

                <div className="explanations" style={{ fontSize: 'smaller', fontWeight: 'normal' }}>
                  File format : csv, xlsx or parquet &lt; {maxSizeMB} MB
                  <br />
                  Example dataset:{' '}
                  <a href="./dataset_test.csv" download>
                    "Detecting Stance in Media On Global Warming" (download)
                  </a>
                </div>
              </>
            )}
          </label>
        )}

        {
          // display datable if data available
          dataset === 'load' && data !== null && (
            <>
              <div>
                Size of the dataset : <b>{lengthData}</b>
              </div>
              <DataTable<Record<DataType['headers'][number], string | number>>
                columns={data.headers.map((h) => ({
                  name: h,
                  selector: (row) => row[h],
                  format: (row) => {
                    const v = row[h];
                    return typeof v === 'bigint' ? Number(v) : v;
                  },
                  width: '200px',
                }))}
                data={
                  data.data
                    .slice(0, 5)
                    .map((row) =>
                      Object.fromEntries(
                        Object.entries(row).map(([key, value]) => [key, String(value)]),
                      ),
                    ) as Record<keyof DataType['headers'], string>[]
                }
              />
            </>
          )
        }

        {
          // only display if data
          availableFields && (
            <>
              <label htmlFor="col_id">
                Id column (must contain unique values; otherwise the row number will be used)
              </label>
              <select id="col_id" disabled={creatingProject} {...register('col_id')}>
                <option key="row_number" value="row_number">
                  Row number
                </option>
                {columns.map((h) => (
                  <option key={h} value={h}>
                    {h}
                  </option>
                ))}
              </select>

              <label htmlFor="cols_text">Text columns (selected fields will be concatenated)</label>

              <Controller
                name="cols_text"
                control={control}
                defaultValue={[]}
                render={({ field: { value, onChange } }) => (
                  <Select
                    options={availableFields}
                    isMulti
                    isDisabled={creatingProject}
                    value={
                      value
                        ? value
                            .map((v: string) => availableFields?.find((opt) => opt.value === v))
                            .filter(Boolean)
                        : []
                    }
                    onChange={(selectedOptions) => {
                      onChange(
                        selectedOptions ? selectedOptions.map((option) => option?.value) : [],
                      );
                    }}
                  />
                )}
              />

              <label htmlFor="language">
                Language of the corpus (for tokenization and word segmentation)
              </label>
              <select id="language" disabled={creatingProject} {...register('language')}>
                {langages.map((lang) => (
                  <option key={lang.value} value={lang.value}>
                    {lang.label}
                  </option>
                ))}
              </select>

              <label htmlFor="col_label">Column(s) for existing annotations (optional)</label>
              <Controller
                name="cols_label"
                control={control}
                defaultValue={[]}
                render={({ field: { value, onChange } }) => (
                  <Select
                    id="cols_label"
                    options={availableFields}
                    isMulti
                    value={
                      value
                        ? value
                            .map((v: string) => availableFields?.find((opt) => opt.value === v))
                            .filter(Boolean)
                        : []
                    }
                    isDisabled={creatingProject}
                    onChange={(selectedOptions) => {
                      onChange(
                        selectedOptions ? selectedOptions.map((option) => option?.value) : [],
                      );
                    }}
                  />
                )}
              />

              <label htmlFor="cols_context">Column(s) for contextual information (optional)</label>
              <Controller
                name="cols_context"
                control={control}
                render={({ field: { onChange, value } }) => (
                  <Select
                    id="cols_context"
                    options={availableFields}
                    isMulti
                    defaultValue={[]}
                    isDisabled={creatingProject}
                    value={
                      value
                        ? value
                            .map((v: string) => availableFields?.find((opt) => opt.value === v))
                            .filter(Boolean)
                        : []
                    }
                    onChange={(selectedOptions) => {
                      onChange(
                        selectedOptions ? selectedOptions.map((option) => option?.value) : [],
                      );
                    }}
                  />
                )}
              />

              <label htmlFor="n_train">Number of rows in the train set (limit : 100,000)</label>
              <input
                id="n_train"
                type="number"
                disabled={creatingProject}
                {...register('n_train')}
                max={maxTrainSet}
                min={1}
              />

              <div className="explanations">
                For machine learning best practices, see the{' '}
                <a target="_blank" href="https://activetigger.com/documentation/" rel="noreferrer">
                  documentation
                </a>
              </div>

              <label htmlFor="n_valid">
                Number of rows in the validation set (optional)
                <a className="n_valid">
                  <HiOutlineQuestionMarkCircle />
                </a>
                <Tooltip anchorSelect=".n_valid" place="top">
                  The validation is generally used for hyperparameter tuning
                </Tooltip>
              </label>
              <input
                id="n_valid"
                type="number"
                disabled={creatingProject}
                {...register('n_valid')}
                min={0}
              />

              <label htmlFor="n_test">
                Number of rows in the test set (optional)
                <a className="n_test">
                  <HiOutlineQuestionMarkCircle />
                </a>
                <Tooltip anchorSelect=".n_test" place="top">
                  The test set will be used for final evaluation
                </Tooltip>
              </label>
              <input
                id="n_test"
                type="number"
                disabled={creatingProject}
                {...register('n_test')}
                min={0}
              />

              <details>
                <summary>Advanced options</summary>
                <div className="explanations">
                  Check the{' '}
                  <a
                    target="_blank"
                    href="https://activetigger.com/documentation/"
                    rel="noreferrer"
                  >
                    documentation
                  </a>{' '}
                  for further explanations
                </div>
                <div>
                  <input
                    id="force_label"
                    type="checkbox"
                    disabled={creatingProject}
                    {...register('force_label')}
                  />
                  <label htmlFor="force_label">Prioritize rows with a label </label>
                </div>
                <div>
                  <input
                    id="random_selection"
                    type="checkbox"
                    disabled={creatingProject || force_label}
                    {...register('random_selection')}
                  />
                  <label htmlFor="random_selection">
                    Select rows at random{' '}
                    <a className="rselect">
                      <HiOutlineQuestionMarkCircle />
                    </a>
                    <Tooltip anchorSelect=".rselect" place="top">
                      If unticked, will preserve order of the original dataset (note: rows without
                      <br />
                      text are dropped) NEED A FIX only if no evaluation datasets (eval/test)
                    </Tooltip>
                  </label>
                </div>
                <div>
                  <input
                    id="stratify_train"
                    type="checkbox"
                    disabled={creatingProject || force_label}
                    {...register('stratify_train')}
                  />
                  <label htmlFor="stratify_train">
                    Stratify train set{' '}
                    <a className="stratify_train">
                      <HiOutlineQuestionMarkCircle />
                    </a>
                    <Tooltip anchorSelect=".stratify_train" place="top">
                      If ticked, ensures balanced representation of each group in the train set. See
                      documentation.
                    </Tooltip>
                  </label>
                </div>
                <div>
                  <input
                    id="stratify_test"
                    type="checkbox"
                    disabled={creatingProject || force_label}
                    {...register('stratify_test')}
                  />
                  <label htmlFor="stratify_test">
                    Stratify test set{' '}
                    <a className="stratify_train">
                      <HiOutlineQuestionMarkCircle />
                    </a>
                    <Tooltip anchorSelect=".stratify_train" place="top">
                      If ticked, ensures balanced representation of each group in the train set. See
                      documentation.
                    </Tooltip>
                  </label>
                </div>

                <label htmlFor="cols_stratify">Column(s) used for stratification</label>
                <Controller
                  name="cols_stratify"
                  control={control}
                  render={({ field: { onChange } }) => (
                    <Select
                      id="cols_stratify"
                      options={availableFields}
                      isMulti
                      isDisabled={creatingProject}
                      onChange={(selectedOptions) => {
                        onChange(
                          selectedOptions ? selectedOptions.map((option) => option.value) : [],
                        );
                      }}
                    />
                  )}
                />
                <div>
                  <input
                    id="clear_test"
                    type="checkbox"
                    disabled={creatingProject}
                    {...register('clear_test')}
                  />
                  <label htmlFor="clear_test">Drop annotations for the testset </label>
                </div>
                <label htmlFor="clear_test">
                  <input
                    id="compute_feature"
                    type="checkbox"
                    disabled={creatingProject}
                    checked={computeFeatures}
                    onChange={() => {
                      setComputeFeatures(!computeFeatures);
                    }}
                  />
                  Compute sentence embeddings
                </label>
                <label htmlFor="n_valid" className="d-flex align-items-center">
                  Seed
                  <a className="ref_seed">
                    <HiOutlineQuestionMarkCircle />
                  </a>
                  <Tooltip anchorSelect=".ref_seed" place="top">
                    Set it to ensure replicability.
                  </Tooltip>
                  <input
                    id="seed"
                    type="number"
                    disabled={creatingProject}
                    {...register('seed', { valueAsNumber: true })}
                    min={0}
                    step={1}
                    className="w-25 ms-3"
                    placeholder="0"
                  />
                </label>
              </details>
            </>
          )
        }
        {/* 
              Quasi Modal
              overlay progression bar with cancel button 
            */}
        {data && creatingProject && <UploadProgressBar progression={progression} cancel={cancel} />}

        {
          <>
            <button type="submit" className="btn-submit" disabled={creatingProject}>
              Create
            </button>
          </>
        }
      </form>
    </div>
  );
};
