import { omit,random } from 'lodash';
import { FC, useEffect, useRef, useState } from 'react';
import { Controller, useForm, useWatch,SubmitHandler } from 'react-hook-form';
import { UploadProgressBar } from '../UploadProgressBar';
import ReactSelect from 'react-select';
import { useNavigate } from 'react-router-dom';
import { CanceledError } from 'axios';
import { getProjectStatus, useAddFeature,useAddProjectFile,useCopyExistingData,useCreateProject, useGetAvailableDatasets, useProjectNameAvailable } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { useAppContext } from '../../core/useAppContext';
import { getRandomName, loadFile } from '../../core/utils'; 
import { ProjectModel } from '../../types';
import { FiChevronDown, FiChevronUp, FiEye, FiEyeOff, FiUpload, FiFolder, FiPlus, FiMinus, FiHelpCircle, FiTag } from 'react-icons/fi';



export interface DataType {
  headers: string[];
  data: Record<string, string | number | bigint>[];
  filename: string;
}

type Option = { value: string; label: string };
type TrainSelection = 'sequential' | 'random' | 'stratify' | 'force_label';
type HoldoutSelection = 'random' | 'stratify' | null;

interface NumericStepperProps {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  disabled?: boolean;
}

function NumericStepper({ value, onChange, min = 0, max = Infinity, disabled }: NumericStepperProps) {
  const clamp = (v: number) => Math.max(min, Math.min(max, v));

  const handleInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const parsed = parseInt(e.target.value, 10);
    if (!isNaN(parsed)) onChange(clamp(parsed));
  };

  return (
    <div className="stepper">
      <button
        type="button"
        disabled={disabled || value <= min}
        onClick={() => onChange(clamp(value - 1))}
        className="stepper__btn"
      >
        <FiMinus size={12} />
      </button>
      <input
        type="number"
        value={value}
        onChange={handleInput}
        min={min}
        max={max}
        disabled={disabled}
        className="stepper__input"
      />
      <button
        type="button"
        disabled={disabled || value >= max}
        onClick={() => onChange(clamp(value + 1))}
        className="stepper__btn"
      >
        <FiPlus size={12} />
      </button>
    </div>
  );
}

interface SplitSliderProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max: number;
  disabled?: boolean;
  color: string;
}

function SplitSlider({ label, value, onChange, min = 0, max, disabled, color }: SplitSliderProps) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;

  return (
    <div className="split-slider">
      <div className="split-slider__header">
        <span className="split-slider__label">{label}</span>
        <span className="split-slider__pct">{pct}%</span>
      </div>
      <div className="split-slider__track">
        <div
          className="split-slider__fill"
          style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: color }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={1}
          value={value}
          onChange={(e) => onChange(parseInt(e.target.value, 10))}
          disabled={disabled}
          className="split-slider__range"
        />
      </div>
      <div className="split-slider__controls">
        <span className="split-slider__bound">{min.toLocaleString()}</span>
        <NumericStepper value={value} onChange={onChange} min={min} max={max} disabled={disabled} />
        <span className="split-slider__bound">{max.toLocaleString()}</span>
      </div>
    </div>
  );
}

const languages = [
  { value: 'en', label: 'English' },
  { value: 'fr', label: 'French' },
  { value: 'es', label: 'Spanish' },
  { value: 'de', label: 'German' },
  { value: 'cn', label: 'Chinese' },
  { value: 'ja', label: 'Japanese' },
];

const TRAIN_STRATEGIES: { value: TrainSelection; label: string; hint: string }[] = [
  { value: 'random', label: 'Random', hint: 'Random sample from the full dataset' },
  { value: 'sequential', label: 'Sequential', hint: 'First N rows in order' },
  { value: 'stratify', label: 'Stratified', hint: 'Balanced by a column value' },
  { value: 'force_label', label: 'Force label', hint: 'Prioritise labelled rows' },
];

export const ProjectCreationForm: FC = () => {
  const maxSizeMB = 400;
  const maxSize = maxSizeMB * 1024 * 1024;
  const maxTrainSet = 100_000;

  const { register, control, handleSubmit, setValue, watch } = useForm<ProjectModel & { num_rows_val: number; num_rows_t: number }>({
    defaultValues: {
      project_name: getRandomName('Project'),
      n_train: 100,
      n_test: 0,
      n_valid: 0,
      language: 'en',
      clear_test: false,
      seed: random(0, 10000),
      train_selection: 'random',
      holdout_selection: null,
      start_index_val: null,
      num_rows_val: 0,
      start_index_test: null,
      num_rows_t: 0,
    },
  });

  // local state
  const [mode, setMode] = useState<'upload' | 'existing'>('upload');
  const [data, setData] = useState<DataType | null>(null);
  const [dataset, setDataset] = useState<string | null>(null);
  const [lengthData, setLengthData] = useState(0);
  const [columns, setColumns] = useState<string[]>([]);
  const [availableFields, setAvailableFields] = useState<Option[] | undefined>(undefined);
  const [previewVisible, setPreviewVisible] = useState(true);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [computeFeatures, setComputeFeatures] = useState(true);
  //const [notification, setNotification] = useState<{ type: 'error' | 'success' | 'warning'; message: string } | null>(null);
  const [creatingProject, setCreatingProject] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [featureBatchSize, setFeatureBatchSize] = useState<number>(32);
  //------
  const { resetContext } = useAppContext();
  const { notify } = useNotifications(); // replace your local notify state
  const navigate = useNavigate();
  const { datasets } = useGetAvailableDatasets(true);
  const createProject = useCreateProject();
  const availableProjectName = useProjectNameAvailable();
  const addFeature = useAddFeature();
  const { addProjectFile, progression, cancel } = useAddProjectFile();
  const copyExistingData = useCopyExistingData();
  // ── Watched values ──
  const train_selection = useWatch({ control, name: 'train_selection' });
  const holdout_selection = useWatch({ control, name: 'holdout_selection' });
  const n_train = useWatch({ control, name: 'n_train' });
  const n_valid = useWatch({ control, name: 'n_valid' });
  const n_test = useWatch({ control, name: 'n_test' });
  const cols_label = useWatch({ control, name: 'cols_label' });
  // ── Derived max values ──
  const maxTrain = Math.max(0, Math.min(maxTrainSet, lengthData - Number(n_valid) - Number(n_test)));
  const maxValid = Math.max(0, lengthData - Number(n_train) - Number(n_test));
  const maxTest = Math.max(0, lengthData - Number(n_train) - Number(n_valid));
  // ── File load ──
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > maxSize) {
      notify({ type: 'error', message: `File is too large (max ${maxSizeMB} MB)` });
      return;
    }
    const d = await loadFile(file);
    if (!d) {
      notify({ type: 'error', message: 'Could not read file. Supported formats: csv, xlsx.' });
      return;
    }
    setData(d);
    setDataset('load');
    setColumns(d.headers);
    setAvailableFields(d.headers.filter(Boolean).map((h) => ({ value: h, label: h })));
    setLengthData(d.data.length - 1);
    setValue('n_train', Math.min((d.data.length || 1) - 1, 100));
    setPreviewVisible(true);
  };
  //Mode switch resets
  useEffect(() => {
    setData(null);
    setDataset(null);
    setColumns([]);
    setAvailableFields(undefined);
    setLengthData(0);
  }, [mode]);
  useEffect(() => {
  if (dataset === 'load' && data) {
    setAvailableFields(data.headers.filter(Boolean).map((e) => ({ value: e, label: e })));
    setColumns(data.headers);
    setLengthData(data.data.length - 1);
  } else if (dataset && dataset !== 'load' && datasets) {
    const istoy = dataset.startsWith('-toy-dataset-');
    const element = istoy
      ? datasets.toy_datasets?.find((e) => `-toy-dataset-${e.project_slug}` === dataset)
      : datasets.projects.find((e) => e.project_slug === dataset);
    setAvailableFields(element?.columns.filter(Boolean).map((e) => ({ value: e, label: e })));
    setColumns(element?.columns || []);
    setLengthData(element?.n_rows ?? 0);
  } else {
    setAvailableFields(undefined);
    setColumns([]);
    setLengthData(0);
  }
}, [data, dataset, datasets]);
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
    setMode('upload');
    notify({ type: 'success', message: 'Example dataset loaded.' });
  } catch (e) {
    notify({ type: 'error', message: `Failed to load example: ${e}` });
  }
};

const onSubmit: SubmitHandler<ProjectModel> = async (formData) => {
  if (!formData.project_name) { notify({ type: 'error', message: 'Enter a project name.' }); return; }
  if (!formData.col_id) { notify({ type: 'error', message: 'Select an ID column.' }); return; }
  if (!formData.cols_text?.length) { notify({ type: 'error', message: 'Select at least one text column.' }); return; }
  if (Number(formData.n_train) + Number(formData.n_test) + Number(formData.n_valid) > lengthData) {
    notify({ type: 'warning', message: 'Sizes exceed dataset length — please adjust.' });
    return;
  }

  const available = await availableProjectName(formData.project_name);
  if (!available) { notify({ type: 'error', message: 'Project name already taken.' }); return; }

  try {
    setCreatingProject(true);
    if (dataset === 'load' && data) {
      const file = fileInputRef.current?.files?.[0];
      if (file) await addProjectFile(formData.project_name, file);
    } else if (dataset && dataset !== 'load') {
      const from_toy_dataset = dataset.startsWith('-toy-dataset-');
      const source_project = from_toy_dataset ? dataset.slice(13) : dataset;
      await copyExistingData(formData.project_name, source_project, from_toy_dataset);
    } else {
      notify({ type: 'error', message: 'Unknown dataset.' });
      return;
    }
    const slug = await createProject({
      project_name: formData.project_name,
      col_id: formData.col_id,
      cols_text: formData.cols_text,
      cols_label: formData.cols_label ?? [],
      cols_context: formData.cols_context ?? [],
      cols_stratify: formData.cols_stratify ?? [],
      language: formData.language,
      n_train: Number(formData.n_train),
      n_valid: Number(formData.n_valid),
      n_test: Number(formData.n_test),
      seed: formData.seed,
      clear_test: formData.clear_test,
      train_selection: formData.train_selection,
      holdout_selection: formData.holdout_selection,
      filename: data ? data.filename : null,
      from_project: dataset === 'load' ? null : dataset,
      from_toy_dataset: dataset?.startsWith('-toy-dataset-') ?? false,
      start_index_val: formData.train_selection === 'sequential' ? formData.start_index_val || null : null,
      start_index_test: formData.train_selection === 'sequential' ? formData.start_index_test || null : null,
      embeddings: [],
      n_skip: 0,
      default_scheme: [],
      test: false,
      valid: false,
      clear_valid: false,
      force_computation: false
    });
    const maxDuration = 5 * 60 * 1000;
    const startTime = Date.now();
    const intervalId = setInterval(async () => {
      try {
        const status = await getProjectStatus(slug);
        console.log('Project status:', status);

        if (status?.startsWith('error') || status === 'not existing') {
          clearInterval(intervalId);
          notify({ type: 'error', message: 'Project creation failed.' });
          navigate('/projects');
          return;
        }

        // if the project has been created
        if (status === 'existing') {
          clearInterval(intervalId);
          if (computeFeatures)
            await addFeature(slug, 'sentence-embeddings', 'default', true, {
              model: 'generic',
              batch_size: featureBatchSize,
            });
          setCreatingProject(false);
          resetContext();
          navigate(`/projects/${slug}?fromCreatePage=true`);
          return;
        }

        if (Date.now() - startTime >= maxDuration) {
          clearInterval(intervalId);
          notify({ type: 'error', message: 'Timeout. Try again later.' });
          navigate('/projects');
        }
      } catch (error) {
        console.error('Polling error:', error);
        clearInterval(intervalId);
      }
    }, 1000);

  } catch (error) {
    setCreatingProject(false);
    if (!(error instanceof CanceledError)) notify({ type: 'error', message: error + '' });
    else notify({ type: 'success', message: 'Project creation aborted.' });
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
    <div className="projectcontainer">
      <div className="projectcard">
        <div className="projectelement">

          <h1 className="page-title">New Project</h1>

          <form onSubmit={handleSubmit(onSubmit)} className="project-form">

            {/* ── Project name ── */}
            <div className="form-section">
              <label className="field-label" htmlFor="project_name">
                Project name
              </label>
              <input
                id="project_name"
                type="text"
                placeholder="Project name"
                disabled={creatingProject}
                {...register('project_name')}
                onClick={(e) => (e.target as HTMLInputElement).select()}
                className="form-input"
              />
            </div>

            {/*Dataset source */}
            <div className="form-section">
              <label className="field-label">Dataset source</label>

              {/* Toggle */}
              <div className="mode-toggle">
                <button
                  type="button"
                  onClick={() => setMode('upload')}
                  className={`mode-toggle__btn${mode === 'upload' ? ' mode-toggle__btn--active' : ''}`}
                >
                  <FiUpload size={14} />
                  Upload new file
                </button>
                <button
                  type="button"
                  onClick={() => setMode('existing')}
                  className={`mode-toggle__btn${mode === 'existing' ? ' mode-toggle__btn--active' : ''}`}
                >
                  <FiFolder size={14} />
                  Choose existing
                </button>
              </div>

              {/* Upload panel */}
              {mode === 'upload' && (
                <div>
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    className={`upload-zone${data ? ' upload-zone--filled' : ''}`}
                  >
                    <svg
                      viewBox="0 0 24 24"
                      className="upload-icon"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        clipRule="evenodd"
                        d="M10 1C9.73478 1 9.48043 1.10536 9.29289 1.29289L3.29289 7.29289C3.10536 7.48043 3 7.73478 3 8V20C3 21.6569 4.34315 23 6 23H7C7.55228 23 8 22.5523 8 22C8 21.4477 7.55228 21 7 21H6C5.44772 21 5 20.5523 5 20V9H10C10.5523 9 11 8.55228 11 8V3H18C18.5523 3 19 3.44772 19 4V9C19 9.55228 19.4477 10 20 10C20.5523 10 21 9.55228 21 9V4C21 2.34315 19.6569 1 18 1H10ZM9 7H6.41421L9 4.41421V7ZM14 15.5C14 14.1193 15.1193 13 16.5 13C17.8807 13 19 14.1193 19 15.5V16V17H20C21.1046 17 22 17.8954 22 19C22 20.1046 21.1046 21 20 21H13C11.8954 21 11 20.1046 11 19C11 17.8954 11.8954 17 13 17H14V16V15.5ZM16.5 11C14.142 11 12.2076 12.8136 12.0156 15.122C10.2825 15.5606 9 17.1305 9 19C9 21.2091 10.7909 23 13 23H20C22.2091 23 24 21.2091 24 19C24 17.1305 22.7175 15.5606 20.9844 15.122C20.7924 12.8136 18.858 11 16.5 11Z"
                      />
                    </svg>
                    <span className="upload-filename">
                      {data ? data.filename : 'Click to upload file'}
                    </span>
                    <span className="upload-meta">
                      {data ? `${lengthData} rows` : `csv / xlsx · max ${maxSizeMB} MB`}
                    </span>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".csv,.xlsx,.xls"
                      style={{ display: 'none' }}
                      disabled={creatingProject}
                      onChange={handleFileChange}
                      onClick={(e) => e.stopPropagation()}
                    />
                  </div>

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
              )}
              {mode === 'existing' && (
                <div className="existing-panel">
                  <select
                    className="form-select"
                    value={dataset ?? ''}
                    onChange={(e) => setDataset(e.target.value || null)}
                    disabled={creatingProject}
                  >
                    <option value="">— select a project —</option>
                    <optgroup label="Projects">
                      {(datasets?.projects || []).map((d) => (
                        <option key={d.project_slug} value={d.project_slug}>{d.project_slug}</option>
                      ))}
                    </optgroup>
                    {datasets?.toy_datasets && datasets.toy_datasets.length > 0 && (
                      <optgroup label="Example datasets">
                        {datasets.toy_datasets.map((d) => (
                          <option key={`-toy-dataset-${d.project_slug}`} value={`-toy-dataset-${d.project_slug}`}>
                            {d.project_slug}
                          </option>
                        ))}
                      </optgroup>
                    )}
                  </select>
                </div>
              )}
              {/* Data preview */}
              {mode === 'upload' && data && (
                <div className="data-preview">
                  <div className="data-preview__header">
                    <span className="data-preview__label">
                      Data preview — {lengthData.toLocaleString()} rows
                    </span>
                    <button
                      type="button"
                      onClick={() => setPreviewVisible((v) => !v)}
                      className="data-preview__toggle"
                    >
                      {previewVisible ? <FiEyeOff size={12} /> : <FiEye size={12} />}
                      {previewVisible ? 'Hide' : 'Show'}
                    </button>
                  </div>
                  {previewVisible && (
                    <div className="data-preview__table-wrap">
                      <table className="data-preview__table">
                        <thead>
                          <tr>
                            {data.headers.map((h) => (
                              <th key={h}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {data.data.slice(0, 5).map((row, i) => (
                            <tr key={i}>
                              {data.headers.map((h) => (
                                <td key={h} title={String(row[h])}>
                                  {String(row[h])}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </div>
            {availableFields && (
              <>
                <div className="col-mapping">
                  <p className="col-mapping__title">Column mapping</p>
                  <div className="col-fields">

                    {/* ID column */}
                    <div>
                      <label className="field-sublabel" htmlFor="col_id">
                        ID column
                      </label>
                      <select
                        id="col_id"
                        disabled={creatingProject}
                        {...register('col_id')}
                        className="form-select"
                      >
                        <option value="row_number">Row number</option>
                        {columns.filter(Boolean).map((h) => (
                          <option key={h} value={h}>{h}</option>
                        ))}
                      </select>
                    </div>

                    {/* Text columns */}
                    <div>
                      <label className="field-sublabel">
                        Text column(s) <span className="field-sublabel__note">— will be concatenated</span>
                      </label>
                      <Controller
                        name="cols_text"
                        control={control}
                        defaultValue={[]}
                        render={({ field: { value, onChange } }) => (
                          <ReactSelect
                            options={availableFields}
                            isMulti
                            isDisabled={creatingProject}
                            value={value?.map((v) => availableFields.find((o) => o.value === v)).filter(Boolean)}
                            onChange={(sel) => onChange(sel ? sel.map((o) => o?.value ?? '') : [])}
                            classNamePrefix="rs"
                            placeholder="Select text column(s)..."
                            styles={{ control: (base) => ({ ...base, borderColor: '#e5e7eb', borderRadius: '0.5rem', fontSize: '14px' }) }}
                          />
                        )}
                      />
                    </div>

                    {/* Label columns */}
                    <div>
                      <div className="field-label-row">
                        <label className="field-sublabel" style={{ marginBottom: 0 }}>
                          Label column(s) <span className="field-sublabel__note">— optional</span>
                        </label>
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
                                <FiTag size={10} />
                                {s.value}
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                      <Controller
                        name="cols_label"
                        control={control}
                        defaultValue={[]}
                        render={({ field: { value, onChange } }) => (
                          <ReactSelect
                            options={availableFields}
                            isMulti
                            isDisabled={creatingProject}
                            value={value?.map((v) => availableFields.find((o) => o.value === v)).filter(Boolean)}
                            onChange={(sel) => onChange(sel ? sel.map((o) => o?.value ?? '') : [])}
                            classNamePrefix="rs"
                            placeholder="Select label column(s)..."
                            styles={{ control: (base) => ({ ...base, borderColor: '#e5e7eb', borderRadius: '0.5rem', fontSize: '14px' }) }}
                          />
                        )}
                      />
                    </div>
                    {/* Context columns */}
                    <div>
                      <label className="field-sublabel">
                        Context column(s) <span className="field-sublabel__note">— optional</span>
                      </label>
                      <Controller
                        name="cols_context"
                        control={control}
                        defaultValue={[]}
                        render={({ field: { value, onChange } }) => (
                          <ReactSelect
                            options={availableFields}
                            isMulti
                            isDisabled={creatingProject}
                            value={value?.map((v) => availableFields.find((o) => o.value === v)).filter(Boolean)}
                            onChange={(sel) => onChange(sel ? sel.map((o) => o?.value ?? '') : [])}
                            classNamePrefix="rs"
                            placeholder="Select context column(s)..."
                            styles={{ control: (base) => ({ ...base, borderColor: '#e5e7eb', borderRadius: '0.5rem', fontSize: '14px' }) }}
                          />
                        )}
                      />
                    </div>
                    {/* Language */}
                    <div>
                      <label className="field-sublabel" htmlFor="language">
                        Language
                      </label>
                      <select
                        id="language"
                        disabled={creatingProject}
                        {...register('language')}
                        className="form-select"
                      >
                        {languages.map((l) => (
                          <option key={l.value} value={l.value}>{l.label}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
                {/*Split sizes*/}
                <div className="form-section">
                  <p className="section-title--mb">Split sizes</p>
                  <div className="split-grid">
                    <SplitSlider
                      label="Train"
                      value={Number(n_train)}
                      onChange={(v) => setValue('n_train', v)}
                      min={1}
                      max={maxTrain}
                      disabled={creatingProject}
                      color="#3B82F6"
                    />
                    <SplitSlider
                      label="Validation"
                      value={Number(n_valid)}
                      onChange={(v) => setValue('n_valid', v)}
                      min={0}
                      max={maxValid}
                      disabled={creatingProject}
                      color="#8B5CF6"
                    />
                    <SplitSlider
                      label="Test"
                      value={Number(n_test)}
                      onChange={(v) => setValue('n_test', v)}
                      min={0}
                      max={maxTest}
                      disabled={creatingProject}
                      color="#10B981"
                    />
                  </div>
                  {lengthData > 0 && (
                    <p className="split-total">
                      Total used: {(Number(n_train) + Number(n_valid) + Number(n_test)).toLocaleString()} / {lengthData.toLocaleString()}
                    </p>
                  )}
                </div>
                {/*Train selection strategy */}
                <div className="form-section">
                  <div className="section-title-row">
                    <p className="section-title">Train selection strategy</p>
                    <FiHelpCircle size={14} className="help-icon" title="Controls how training rows are picked from your dataset" />
                  </div>
                  <div className="strategy-grid">
                    {TRAIN_STRATEGIES.map((s) => (
                      <button
                        key={s.value}
                        type="button"
                        onClick={() => setValue('train_selection', s.value)}
                        disabled={creatingProject}
                        title={s.hint}
                        className={`strategy-btn${train_selection === s.value ? ' strategy-btn--active' : ''}`}
                      >
                        {s.label}
                      </button>
                    ))}
                  </div>
                  {train_selection === 'sequential' && (
                    <div className="slice-panel">
                      <p className="slice-panel__desc">
                        Define where validation/test slices start in your ordered data. Leave at 0 to auto-place after the train block.
                      </p>
                      <div className="slice-panel__grid">
                        {[
                          { startName: 'start_index_val' as const, sizeName: 'num_rows_val' as const, label: 'Validation slice', show: Number(n_valid) > 0 },
                          { startName: 'start_index_test' as const, sizeName: 'num_rows_t' as const, label: 'Test slice', show: Number(n_test) > 0 },
                        ].map(({ startName, sizeName, label, show }) => (
                          <div key={startName} className={`slice-item${show ? '' : ' slice-item--disabled'}`}>
                            <p className="slice-item__title">{label}</p>
                            <div className="slice-item__fields">
                              <div>
                                <div className="slice-field__header">
                                  <span>Start row</span>
                                  <span>{Number(watch(startName)).toLocaleString()}</span>
                                </div>
                                <input
                                  type="range" min={0} max={lengthData} step={1}
                                  disabled={creatingProject || !show}
                                  {...register(startName, { valueAsNumber: true })}
                                  className="slice-field__range"
                                />
                              </div>
                              <div>
                                <div className="slice-field__header">
                                  <span>Size</span>
                                  <span>{Number(watch(sizeName)).toLocaleString()}</span>
                                </div>
                                <input
                                  type="range" min={0} max={lengthData} step={1}
                                  disabled={creatingProject || !show}
                                  {...register(sizeName, { valueAsNumber: true })}
                                  className="slice-field__range"
                                />
                              </div>
                            </div>
                            {!show && <p className="slice-item__empty-hint">Set size &gt; 0 above to configure</p>}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {/* Stratified column picker */}
                  {train_selection === 'stratify' && (
                    <div className="stratify-picker">
                      <label className="stratify-picker__label">Stratification column(s)</label>
                      <Controller
                        name="cols_stratify"
                        control={control}
                        render={({ field: { value, onChange } }) => (
                          <ReactSelect
                            options={availableFields}
                            isMulti
                            isDisabled={creatingProject}
                            value={(value ?? []).map((v) => availableFields?.find((o) => o.value === v)).filter(Boolean) as Option[]}
                            onChange={(sel) => onChange(sel ? sel.map((o) => o?.value ?? '') : [])}
                            classNamePrefix="rs"
                            placeholder="Select column(s) to stratify by..."
                            styles={{ control: (base) => ({ ...base, borderColor: '#e5e7eb', borderRadius: '0.5rem', fontSize: '14px' }) }}
                          />
                        )}
                      />
                    </div>
                  )}
                </div>
                {(Number(n_valid) > 0 || Number(n_test) > 0) && (
                  <div className="form-section">
                    <p className="section-title">Holdout selection strategy</p>
                    <div className="holdout-grid">
                      {(['random', 'stratify'] as HoldoutSelection[]).map((s) => (
                        <button
                          key={s}
                          type="button"
                          onClick={() => setValue('holdout_selection', s)}
                          disabled={creatingProject}
                          className={`strategy-btn${holdout_selection === s ? ' strategy-btn--active' : ''}`}
                          style={{ textTransform: 'capitalize' }}
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                    {holdout_selection === 'stratify' && (
                      <div className="stratify-picker">
                        <label className="stratify-picker__label">Stratification column(s) for holdout</label>
                        <Controller
                          name="cols_stratify"
                          control={control}
                          render={({ field: { value, onChange } }) => (
                            <ReactSelect
                              options={availableFields}
                              isMulti
                              isDisabled={creatingProject}
                              value={(value ?? []).map((v) => availableFields?.find((o) => o.value === v)).filter(Boolean) as Option[]}
                              onChange={(sel) => onChange(sel ? sel.map((o) => o?.value ?? '') : [])}
                              classNamePrefix="rs"
                              placeholder="Select column(s)..."
                              styles={{ control: (base) => ({ ...base, borderColor: '#e5e7eb', borderRadius: '0.5rem', fontSize: '14px' }) }}
                            />
                          )}
                        />
                      </div>
                    )}
                  </div>
                )}

                {/*Drop annotations for test (only when n_test > 0) */}
                {Number(n_test) > 0 && (
                  <div className="drop-annotations">
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        disabled={creatingProject}
                        {...register('clear_test')}
                      />
                      Drop annotations for test set
                    </label>
                  </div>
                )}
                <div className="form-section">
                  <button
                    type="button"
                    onClick={() => setAdvancedOpen((v) => !v)}
                    className="advanced-toggle"
                  >
                    {advancedOpen ? <FiChevronUp size={14} /> : <FiChevronDown size={14} />}
                    Advanced options
                  </button>
                  {advancedOpen && (
                    <div className="advanced-panel">
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
                      <div className="seed-row">
                        <span>Seed</span>
                        <FiHelpCircle size={13} className="help-icon" title="Set for reproducibility" />
                        <input
                          type="number"
                          disabled={creatingProject}
                          {...register('seed', { valueAsNumber: true })}
                          min={0}
                          step={1}
                          className="seed-input"
                        />
                      </div>
                    </div>
                  )}
                </div>
                <div className="form-section">
                  <button type="submit" disabled={creatingProject} className="submit-btn">
                    {creatingProject ? 'Creating project…' : 'Create project'}
                  </button>
                  {/* Upload progress rendered separately, outside the button */}
                  {data && creatingProject && (
                    <UploadProgressBar progression={progression} cancel={cancel} />
                  )}
                </div>
              </>
            )}
          </form>
        </div>
      </div>
    </div>
  );
};
