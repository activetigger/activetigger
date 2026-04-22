import { omit, random } from 'lodash';
import { FC, useEffect, useState, useRef } from 'react';
import DataTable from 'react-data-table-component';
import { Controller, SubmitHandler, useForm, useWatch } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import Select from 'react-select';
import { UploadProgressBar } from '../UploadProgressBar';

import { CanceledError } from 'axios';
import { HiOutlineQuestionMarkCircle, HiMinusSm, HiOutlinePlusSm, HiCloudUpload, HiEyeOff, HiEye, HiTag } from 'react-icons/hi';
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
// Numeric Stepper
interface NumericstepperProps {
  value: number;
  set: string;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  disabled?: boolean;
}

function Stepper({ value, set, onChange, min = 0, max = Infinity, disabled }: NumericstepperProps) {
  const clamp = (v: number) => Math.max(min, Math.min(max, v));

  return (
    <div className="col">
      <div className="d-flex flex-row align-items-center gap-1">
        <button
          type="button"
          disabled={disabled || value <= min}
          onClick={() => onChange(clamp(value - 1))}
          className="btn btn-outline-secondary btn-sm d-flex align-items-center justify-content-center"
        >
          <HiMinusSm size={12} />
        </button>
        <input
          type="number"
          id={set}
          value={value}
          onChange={(e) => { const p = parseInt(e.target.value, 10); if (!isNaN(p)) onChange(clamp(p)); }}
          min={min}
          max={max}
          disabled={disabled}
          className="form-control form-control-sm text-center"
          style={{ width: '5rem' }}
        />
        <button
          type="button"
          disabled={disabled || value >= max}
          onClick={() => onChange(clamp(value + 1))}
          className="btn btn-outline-secondary btn-sm d-flex align-items-center justify-content-center"
        >
          <HiOutlinePlusSm size={12} />
        </button>
      </div>
    </div>
  );
}

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
  const TrainSelection = [
    { value: 'sequential', label: 'Sequential', hint: 'Preserves original dataset order' },
    { value: 'random', label: 'Random', hint: 'Randomly samples row' },
    { value: 'force_label', label: 'Prioritize rows with a label', hint: 'Labeled rows selected first for the trainset' },
    { value: 'stratification', label: 'Stratify', hint: 'Ensures balanced group representation' },

  ];
  const EvalSelection = [
    { value: 'random', label: 'random', hint: 'Randomly samples rows' },
    { value: 'stratification', label: 'Stratify', hint: 'Ensures balanced group representation' },
    { value: 'sequential', label: 'Sequential', hint: 'Preserves original dataset order - Needs To define Indexes' },
  ];
  //fields

  {/* Define size of Train  + (val/test)*/ }
  const Size_fields: { label: string; name: 'n_train' | 'n_valid' | 'n_test', hint: string }[] = [
    { label: 'Train-set Size', name: 'n_train', hint: 'Number of rows in the train set (limit : 100,000)' },
    { label: 'Validation-set Size', name: 'n_valid', hint: 'The validation is generally used for hyperparameter tuning' },
    { label: 'Test-set Size', name: 'n_test', hint: 'The test set will be used for final evaluation' },
  ];
  {/* Define start index for either test or validation set(s): needed when holdout pool selection is sequential*/ }
  const indexFields: { label: string; name: 's_val_idx' | 's_test_idx'; sizeField: 'n_valid' | 'n_test' }[] = [
    { label: 'Validation-set Start', name: 's_val_idx', sizeField: 'n_valid' },
    { label: 'Test-set Start', name: 's_test_idx', sizeField: 'n_test' },
  ];


  const { register, control, handleSubmit, setValue, reset, watch } = useForm<
    ProjectModel & { files: FileList }
  >({
    defaultValues: {
      project_name: getRandomName('Project'),
      n_train: 100,
      n_test: 0,
      n_valid: 0,
      language: 'en',
      clear_test: false,
      clear_valid: false,
      train_selection: 'random',
      holdout_selection: undefined,
      s_test_idx: null,
      s_val_idx: null,
      cols_stratify: [],
      seed: random(0, 10000),
    },
  });
  const { notify } = useNotifications();
  const { datasets } = useGetAvailableDatasets(true); // Include toy datasets

  const [creatingProject, setCreatingProject] = useState<boolean>(false); // state for the data
  const [dataset, setDataset] = useState<string | null>(null); // state for the data
  const [data, setData] = useState<DataType | null>(null); // state for the data
  const [computeFeatures, setComputeFeatures] = useState<boolean>(true);
  const [featureBatchSize, setFeatureBatchSize] = useState<number>(32);
  const navigate = useNavigate(); // rooting
  const createProject = useCreateProject(); // API call
  const availableProjectName = useProjectNameAvailable(); // check if the project name is available
  const addFeature = useAddFeature();

  const { addProjectFile, progression, cancel } = useAddProjectFile(); // API call
  const copyExistingData = useCopyExistingData();
  // available columns to select, depending of the source
  const [availableFields, setAvailableFields] = useState<Option[] | undefined>(undefined);
  const [columns, setColumns] = useState<string[]>([]);
  const [lengthData, setLengthData] = useState<number>(0);
  //
  const [previewVisible, setPreviewVisible] = useState(true);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);

  //
  const files = useWatch({ control, name: 'files' }); // watch the files entry
  const n_train = useWatch({ control, name: 'n_train' });
  const n_valid = useWatch({ control, name: 'n_valid' });
  const n_test = useWatch({ control, name: 'n_test' });
  const s_val_idx = useWatch({ control, name: 's_val_idx' });
  const s_test_idx = useWatch({ control, name: 's_test_idx' });
  const cols_label = useWatch({ control, name: 'cols_label' });
  //Dynamic sizes
  const maxTrain = Math.max(0, Math.min(maxTrainSet, lengthData - Number(n_valid) - Number(n_test)));
  const maxValid = Math.max(0, lengthData - Number(n_train) - Number(n_test));
  const maxTest = Math.max(0, lengthData - Number(n_train) - Number(n_valid));
  //train strategy
  const train_strat = useWatch({ control, name: "train_selection" });
  //selection from holdout strategy
  const eval_strat = useWatch({ control, name: 'holdout_selection' });


  const fileInputRef = useRef<HTMLInputElement>(null);
  //
  const sizevalues: Record<string, number> = { n_train, n_valid, n_test };
  const indexvals: Record<string, number | null | undefined> = { s_val_idx, s_test_idx };

  //showing index section → obligatory eval selection is sequential:(optional when trainselection is sequential)
  const showEvalStrat = n_valid + n_test > 0;
  const showIndexes = eval_strat === 'sequential' && showEvalStrat;

  //overlap check
  const valStart = s_val_idx ?? 0;
  const testStart = s_test_idx ?? 0;
  const valEnd = valStart + n_valid;
  const testEnd = testStart + n_test;
  const trainEnd = Number(n_train);
  const overlap_test = showEvalStrat && showIndexes && ((valStart <= testStart && valEnd >= testStart) || (valStart <= testEnd && valEnd >= testEnd));
  const ovelap_valid = showEvalStrat && showIndexes && ((valStart >= testStart && valStart <= testEnd) || (valEnd >= testStart && valEnd <= testEnd));
  const overlap_val_test = overlap_test || ovelap_valid
  const isTrainSequential = train_strat === 'sequential';
  const overlap_train_val = showIndexes && isTrainSequential && n_valid > 0 &&
    (valStart < trainEnd && valEnd > 0);
  const overlap_train_test = showIndexes && isTrainSequential && n_test > 0 &&
    (testStart < trainEnd && testEnd > 0);

  const valIndexExceedsDataset = showIndexes && n_valid > 0 && (valStart + Number(n_valid)) > lengthData;
  const testIndexExceedsDataset = showIndexes && n_test > 0 && (testStart + Number(n_test)) > lengthData;
  // Clear strategy + indexes when sizes go to 0
  useEffect(() => {
    if (!showEvalStrat) {
      setValue('holdout_selection', undefined);
      setValue('s_test_idx', null);
      setValue('s_val_idx', null);
    }
  }, [showEvalStrat, setValue]);

  useEffect(() => {
    if (!showIndexes) {
      setValue('s_test_idx', null);
      setValue('s_val_idx', null)
    }
  }, [showIndexes, setValue]);

  useEffect(() => {
    // case of loading external file
    if (dataset === 'load' && data) {
      setAvailableFields(data?.headers.filter((h) => !!h).map((e) => ({ value: e, label: e })));
      setColumns(data.headers);
      setLengthData(data.data.length - 1);
      // case of existing project
    } else if (dataset !== 'load' && datasets) {
      const element =
        dataset?.startsWith('-toy-dataset-') && datasets.toy_datasets
          ? datasets.toy_datasets.find((e) => `-toy-dataset-${e.project_slug}` === dataset)
          : datasets.projects.find((e) => e.project_slug === dataset);

      setAvailableFields(element?.columns.filter((h) => !!h).map((e) => ({ value: e, label: e })));
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
        setUploadedFileName(file.name);
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
      // check that the selected ID column contains unique values
      if (formData.col_id !== 'row_number' && data) {
        const idValues = data.data.map((row) => row[formData.col_id]);
        const uniqueValues = new Set(idValues);
        if (uniqueValues.size !== idValues.length) {
          const nDuplicates = idValues.length - uniqueValues.size;
          notify({
            type: 'error',
            message: `The selected ID column contains ${nDuplicates} duplicate values. Please choose a column with unique values or use 'Row number'.`,
          });
          return;
        }
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
      if (eval_strat === 'sequential' && (Number(formData.n_valid) > 0 || Number(formData.n_test) > 0)) {
        if (formData.s_val_idx === null && Number(formData.n_valid) > 0) {
          notify({ type: 'error', message: 'Sequential holdout strategy requires a start index for the validation set.' });
          return;
        }
        if (formData.s_test_idx === null && Number(formData.n_test) > 0) {
          notify({ type: 'error', message: 'Sequential holdout strategy requires a start index for the test set.' });
          return;
        }
      }
      if ((Number(formData.n_valid) > 0 || Number(formData.n_test) > 0) && !formData.holdout_selection) {
        notify({ type: 'error', message: 'Please select a holdout selection strategy for your validation/test set.' });
        return;
      }
      if ((train_strat === 'stratification' || eval_strat === 'stratification') && (!formData.cols_stratify || formData.cols_stratify.length === 0)) {
        notify({ type: 'error', message: 'Please select at least one stratification column.' });
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
                addFeature(slug, 'sentence-embeddings', 'default', true, {
                  model: 'generic',
                  batch_size: featureBatchSize,
                });
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
      cols_stratify: [],
      n_train: 100,
      n_test: 0,
      n_valid: 0,
      language: 'en',
      clear_test: false,
      clear_valid: false,
      train_selection: 'random',
      holdout_selection: undefined,
      s_test_idx: null,
      s_val_idx: null,
      seed: random(0, 10000),
    });
    setUploadedFileName(null);
    setData(null);
    // reset data when changing dataset
  }, [dataset, reset]);
  // FOr example DataSet
  const handleUseSuggested = async () => {
    try {
      const response = await fetch('/gwsd_train_test.csv');
      if (!response.ok) throw new Error('Could not fetch example dataset');
      const blob = await response.blob();
      const file = new File([blob], 'gwsd_train_test.csv', { type: 'text/csv' });
      const d = await loadFile(file);
      if (!d) {
        notify({ type: 'error', message: 'Could not parse example dataset.' });
        return;
      }
      const dt = new DataTransfer();
      dt.items.add(file);
      if (fileInputRef.current) {
        fileInputRef.current.files = dt.files;
      }
      setData(d);
      setDataset('load');
      setColumns(d.headers);
      setAvailableFields(d.headers.filter(Boolean).map((h) => ({ value: h, label: h })));
      setLengthData(d.data.length - 1);
      setValue('n_train', Math.min((d.data.length || 1) - 1, 100));
      setPreviewVisible(true);
      notify({ type: 'success', message: 'Example dataset loaded.' });
    } catch (e) {
      notify({ type: 'error', message: `Failed to load example: ${e}` });
    }
  };
  const colsLabelSuggestions = availableFields?.filter((f) => {
    if ((cols_label || []).includes(f.value)) return false;
    const nameMatch = /label|class|category|tag|type|sentiment|score|target/i.test(f.value);
    const hasEmptyEntries = data
      ? data.data.some((row) => {
        const val = row[f.value];
        return val === null || val === undefined || val === '' || val === 'null' || val === 'nan';
      })
      : false;
    return nameMatch || hasEmptyEntries;
  }) ?? [];

  return (
    <div className="min-vh-100 d-flex flex-wrap justify-content-center p-4 gap-3">
      <div className="bg-white rounded shadow-sm p-4 w-100">
        <div className="explanations">
          <h4 className="text-center font-weight-bold">Create a new project</h4>
        </div>
        <form onSubmit={handleSubmit(onSubmit)}>
          {/*project name*/}
          <div className='d-flex flex-column gap-2 mt-3'>
            <label htmlFor="project_name">Project name</label>
            <input
              id="project_name"
              placeholder="Name of the project (need to be unique in the system)"
              type="text"
              disabled={creatingProject}
              {...register('project_name')}
              onClick={handleClickOnText}
            />
          </div>
          {/*File source Selection*/}
          <ul className="nav nav-tabs mb-3 mt-3" style={{
            borderBottom: '1px solid #d1d9e6',
            background: 'linear-gradient(to bottom, #f8f9fb, #eef1f7)',
            borderRadius: '8px 8px 0 0',
            padding: '4px 4px 0',
            gap: '4px',
            display: 'flex'
          }}>
            <li className="nav-item">

              <a className={`nav-link ${dataset === 'load' ? 'active' : ''}`}
                style={{
                  cursor: 'pointer',
                  borderRadius: '6px 6px 0 0',
                  color: dataset === 'load' ? '#3771b8' : '#6b7a99',
                  background: dataset === 'load' ? 'white' : 'transparent',
                  border: dataset === 'load' ? '1px solid #d1d9e6' : '1px solid transparent',
                  borderBottom: dataset === 'load' ? '1px solid white' : 'none',
                  fontWeight: dataset === 'load' ? 500 : 400,
                  transition: 'color 0.15s'
                }}
                onClick={() => setDataset('load')}
              >
                Upload file
              </a>
            </li>
            <li className="nav-item">
              <a className={`nav-link ${dataset === 'from-project' ? 'active' : ''}`}
                style={{
                  cursor: 'pointer', borderRadius: '6px 6px 0 0', color: dataset === 'from-project' ? '#3771b8' : '#6b7a99', background: dataset === 'from-project' ? 'white' : 'transparent', border: dataset === 'from-project' ? '1px solid #d1d9e6' : '1px solid transparent', borderBottom: dataset === 'from-project' ? '1px solid white' : 'none',
                  fontWeight: dataset === 'from-project' ? 500 : 400,
                  transition: 'color 0.15s'
                }}
                onClick={() => setDataset('from-project')}
              >
                From existing project
              </a>
            </li>
          </ul>
          {dataset && (
            <div className="mb-3">
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
                <div className='d-flex flex-column justify-content-center align-items-center py-4'>
                  <div className='w-100'>
                    <label
                      htmlFor='file-upload'
                      className='d-flex flex-column align-items-center gap-2 p-5 w-100 rounded border border-2 text-center font-weight-normal'
                      style={{ cursor: 'pointer', borderStyle: 'dashed', backgroundColor: uploadedFileName ? '#eaf4ea' : '#e9ecef', borderColor: uploadedFileName ? '#6abf69' : undefined }}
                      onMouseOver={(e) => (e.currentTarget.style.backgroundColor = uploadedFileName ? '#d6ecd6' : '#e2e4e7')}
                      onMouseOut={(e) => (e.currentTarget.style.backgroundColor = uploadedFileName ? '#eaf4ea' : '#e9ecef')}
                    >
                      <HiCloudUpload size={48} style={{ color: uploadedFileName ? '#4caf50' : '#8fbeed' }} />
                      {uploadedFileName ? (
                        <>
                          <span className='fw-semibold text-success'>{uploadedFileName}</span>
                          <span className='fw-light text-muted' style={{ fontSize: '0.85em' }}>Click to replace file</span>
                        </>
                      ) : (
                        <>
                          <span className='fw-light text-wrap'>Click to upload or drag and drop your file</span>
                          <span className='fs-6 fst-italic'>File format: csv, xlsx or parquet &lt; {maxSizeMB} MB</span>
                        </>
                      )}
                      <input
                        id='file-upload'
                        type='file'
                        style={{ display: 'none' }}
                        disabled={creatingProject}
                        {...register('files')}
                      />
                    </label>
                  </div>
                  <div className="explanations" style={{ fontSize: 'smaller', fontWeight: 'normal' }}>
                    <p className="upload-example">
                      Try an example:{' '}
                      <button
                        type="button"
                        className="upload-example__link"
                        onClick={handleUseSuggested}
                      >
                        "Detecting Stance in Media on Global Warming" : use it
                      </button>
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
          {/*table display*/}
          {
            // display datable if data available
            dataset === 'load' && data !== null && (
              <>
                <div className='d-flex gap-3 align-items-center'>
                  <span className="text-muted small">Size of the dataset: <b>{lengthData}</b></span>
                  <button
                    type="button"
                    onClick={() => setPreviewVisible((v) => !v)}
                    className="btn btn-sm"
                    style={{ color: '#6b7a99', border: '1px solid #d1d9e6', background: 'transparent' }}
                  >
                    {previewVisible ? <HiEyeOff size={12} /> : <HiEye size={12} />}
                    {previewVisible ? ' Hide' : ' Show'}
                  </button>
                </div>
                {previewVisible && (
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
                )}
              <hr className="my-3" />
              </>
            )
          }
          {/* data is avaliable */}
          {
            // only display if data
            availableFields && (

              <>
                {/* id */}
                <div className='d-flex flex-column gap-2 mb-3'>
                  <label htmlFor="col_id">Id column (must contain unique values)</label>
                  <select id="col_id" disabled={creatingProject} {...register('col_id')}>
                    <option key="row_number" value="row_number">
                      Row number
                    </option>
                    {columns
                      .filter((h) => !!h)
                      .map((h) => (
                        <option key={h} value={h}>
                          {h}
                        </option>
                      ))}
                  </select>
                </div>
                {/* Text cols*/}
                <div className='d-flex flex-column gap-2 mb-3'>
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
                </div>
                {/* language */}
                <div className='d-flex flex-column gap-2 mb-3'>
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
                </div>
                {/* annotation selection */}
                <div className='d-flex flex-column gap-2 mb-3'>
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
                            ? value.map((v: string) => availableFields?.find((opt) => opt.value === v)).filter(Boolean)
                            : []
                        }
                        isDisabled={creatingProject}
                        onChange={(selectedOptions) => {
                          onChange(selectedOptions ? selectedOptions.map((option) => option?.value) : []);
                        }}
                      />
                    )}
                  />
                  {colsLabelSuggestions.length > 0 && (
                    <div className="col-suggestions">
                      <span className="col-suggestions__label">Suggestions:</span>
                      {colsLabelSuggestions.slice(0, 3).map((s) => (
                        <button
                          key={s.value}
                          type="button"
                          onClick={() => {
                            const cur = watch('cols_label') || [];
                            if (!cur.includes(s.value)) setValue('cols_label', [...cur, s.value]);
                          }}
                          className="col-suggestion-btn"
                        >
                          <HiTag size={10} />
                          {s.value}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                {/* context cols */}
                <div className='d-flex flex-column gap-2 mb-3'>
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
                </div>
                <hr className="my-3" />
                {/* defining train,(valid/test) sizes*/}
                <div className="explanations">
                  For machine learning best practices, see the{' '}
                  <a target="_blank" href="https://activetigger.com/documentation/" rel="noreferrer">
                    documentation
                  </a>
                </div>
                <hr className="my-3" />
                {/* numm stepper */}
                <div className="d-flex flex-column gap-3 mt-3">
                  <label>Set sizes</label>
                  <div className="d-flex flex-wrap justify-content-around gap-3">
                    {Size_fields.map(({ label, name, hint }) => (
                      <div key={name} className="d-flex flex-column align-items-center gap-2 flex-fill">
                        <div className='d-flex flex-row align-items-center gap-1'>
                          <span className="text-muted small text-center">{label}</span>
                          <span
                            data-tooltip-id={`tooltip-size-${name}`}
                            style={{ cursor: 'pointer', color: '#6b7a99' }}
                          >
                            <HiOutlineQuestionMarkCircle />
                          </span>
                          <Tooltip id={`tooltip-size-${name}`} place="top" content={hint} />
                        </div>
                        <Stepper
                          value={sizevalues[name]}
                          set={name}
                          onChange={(v) => setValue(name, v)}
                          min={0}
                          max={name === 'n_train' ? maxTrain : name === 'n_valid' ? maxValid : maxTest}
                          disabled={creatingProject}
                        />
                      </div>
                    ))}
                  </div>
                  {lengthData > 0 && (
                    <p className="split-total">
                      Total used: {(Number(n_train) + Number(n_valid) + Number(n_test)).toLocaleString()} / {lengthData.toLocaleString()}
                    </p>
                  )}
                </div>
                <hr className="my-3" />
                {/* train selection strategies */}
                <div className='d-flex flex-column gap-2 mt-3'>
                  <label>Training-set Selection Strategy</label>
                  <div className='d-flex flex-wrap justify-content-around'>
                    {TrainSelection.map((opt) => (
                      <div key={opt.value}>
                        <input
                          type="checkbox"
                          className="btn-check"
                          id={`train-btn-check-${opt.value}`}
                          checked={train_strat === opt.value}
                          onChange={() => setValue('train_selection', train_strat === opt.value ? "random" : opt.value as ProjectModel['train_selection'])}
                          disabled={creatingProject}
                        />
                        <label
                          className={`btn ${train_strat === opt.value ? 'btn-outline-success' : 'btn-outline-secondary'}`}
                          htmlFor={`train-btn-check-${opt.value}`}
                          title={opt.hint}
                        >
                          {opt.label}
                        </label>
                      </div>
                    ))}
                  </div>
                  <p className="text-muted mb-0" style={{ fontSize: '0.82em' }}>
                    If no strategy is selected, default is <strong>Random</strong>.
                  </p>
                </div>

                {/*eval selection*/}
                {showEvalStrat && (
                  <>
                    <hr className="my-3" />
                    <div className='d-flex flex-column gap-2 mt-3'>
                      <label>Selection From Holdout-Pool Strategy</label>
                      <div className='d-flex justify-content-around'>
                        {EvalSelection.map((opt) => (
                          <div key={opt.value}>
                            <input
                              type="checkbox"
                              className="btn-check"
                              id={`eval-btn-check-${opt.value}`}
                              checked={eval_strat === opt.value}
                              onChange={() => setValue(
                                'holdout_selection',
                                eval_strat === opt.value ? undefined : opt.value as ProjectModel['holdout_selection']
                              )}
                              disabled={creatingProject}
                            />
                            <label
                              className={`btn ${eval_strat === opt.value ? 'btn-outline-success' : 'btn-outline-secondary'}`}
                              htmlFor={`eval-btn-check-${opt.value}`}
                              title={opt.hint}
                            >
                              {opt.label}
                            </label>
                          </div>
                        ))}
                      </div>
                      <p className="text-muted mb-0" style={{ fontSize: '0.82em' }}>
                        If no strategy is selected, default is <strong>None</strong>
                      </p>
                    </div>
                  </>
                )}
                {/*startification selection cols*/}
                {(train_strat === 'stratification' || eval_strat === 'stratification') && (
                  <>
                  <hr className="my-3" />
                  <div className='d-flex flex-column gap-2 mt-3'>
                    <label>Stratification Column(s)</label>
                    <Controller
                      name="cols_stratify"
                      control={control}
                      defaultValue={[]}
                      render={({ field: { value, onChange } }) => (
                        <Select
                          options={availableFields}
                          isMulti
                          isDisabled={creatingProject}
                          placeholder="Select column(s) to stratify on..."
                          value={
                            value
                              ? value.map((v: string) => availableFields?.find((opt) => opt.value === v)).filter(Boolean)
                              : []
                          }
                          onChange={(selectedOptions) => {
                            onChange(selectedOptions ? selectedOptions.map((option) => option?.value) : []);
                          }}
                        />
                      )}
                    />
                  </div>
                </>
                )}
                {/* indexes */}

                {showIndexes && (
                  <>
                    <hr className="my-3" />
                    <div className='d-flex flex-column gap-2 mt-3'>
                      <label>Set start indexes</label>
                      <div className="d-flex flex-row justify-content-around gap-3">
                        {indexFields.map(({ label, name, sizeField }) => {
                          const sizeIsZero = Number(sizevalues[sizeField]) === 0;
                          return (
                            <div key={name} className="d-flex flex-column align-items-center gap-2">
                              <span className={`text-muted small text-center ${sizeIsZero ? 'opacity-50' : ''}`}>{label}</span>
                              <Stepper
                                value={indexvals[name] ?? 1}
                                set={name}
                                onChange={(v) => setValue(name, v)}
                                min={1}
                                max={lengthData}
                                disabled={creatingProject || sizeIsZero}
                              />
                              {sizeIsZero && (
                                <span className="text-muted" style={{ fontSize: '0.75em' }}>
                                  Set size &gt; 0 to enable
                                </span>
                              )}
                            </div>
                          );
                        })}
                      </div>
                      {(valIndexExceedsDataset || testIndexExceedsDataset) && (
                        <div className="alert alert-warning alert-dismissible fade show d-flex align-items-start gap-2" role="alert">
                          <strong>Index out of bounds!</strong>{' '}
                          {valIndexExceedsDataset && <span>Validation [{valStart}–{valEnd}] exceeds dataset size ({lengthData}). </span>}
                          {testIndexExceedsDataset && <span>Test [{testStart}–{testEnd}] exceeds dataset size ({lengthData}).</span>}
                          <button type="button" className="btn-close ms-auto" aria-label="Close" onClick={() => {
                            if (valIndexExceedsDataset) setValue('s_val_idx', Math.max(0, lengthData - Number(n_valid)));
                            if (testIndexExceedsDataset) setValue('s_test_idx', Math.max(0, lengthData - Number(n_test)));
                          }} />
                        </div>
                      )}
                      {overlap_val_test && n_valid > 0 && n_test > 0 && (
                        <div className="alert alert-warning alert-dismissible fade show" role="alert">
                          <strong>Overlap!</strong> Validation [{valStart}–{valEnd}] and Test [{testStart}–{testEnd}] sets overlap.
                          <button type="button" className="btn-close" data-bs-dismiss="alert" aria-label="Close" />
                        </div>
                      )}
                      {(overlap_train_val || overlap_train_test) && (
                        <div className="alert alert-warning alert-dismissible fade show" role="alert">
                          <strong>Overlap with train set!</strong>{' '}
                          Train set spans rows [0–{trainEnd}].{' '}
                          {overlap_train_val && <span>Validation [{valStart}–{valEnd}] overlaps. </span>}
                          {overlap_train_test && <span>Test [{testStart}–{testEnd}] overlaps.</span>}
                          <button type="button" className="btn-close" data-bs-dismiss="alert" aria-label="Close" />
                        </div>
                      )}
                    </div>
                  </>
                )}
                <hr className="my-3" />
                {/* details*/}
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
                  {n_test > 0 && (
                    <div>
                      <input
                        id="clear_test"
                        type="checkbox"
                        disabled={creatingProject}
                        {...register('clear_test')}
                      />
                      <label htmlFor="clear_test">Drop annotations for the Test Set </label>
                    </div>)}
                  {n_valid > 0 && (
                    <div>
                      <input
                        id="clear_valid"
                        type="checkbox"
                        disabled={creatingProject}
                        {...register('clear_valid')}
                      />
                      <label htmlFor="clear_test">Drop annotations for the Validation Set </label>
                    </div>)}
                  <div className="d-flex align-items-center gap-2">
                    <label htmlFor="compute_feature" className="m-0">
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
                    {computeFeatures && (
                      <label className="batch-size-label">
                        batch
                        <input
                          type="number"
                          min={1}
                          max={512}
                          value={featureBatchSize}
                          onChange={(e) =>
                            setFeatureBatchSize(Math.max(1, parseInt(e.target.value) || 1))
                          }
                          title="Batch size for embedding computation"
                          disabled={creatingProject}
                        />
                      </label>
                    )}
                  </div>
                  <label htmlFor="seed" className="d-flex align-items-center">
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

          {(data || dataset !== 'load') && (
            <>
              <button type="submit" className="btn-submit" disabled={creatingProject}>
                Create
              </button>
            </>
          )}
        </form>
      </div >
    </div >
  );
};