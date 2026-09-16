import { FC, useEffect, useState } from 'react';
import { FaArrowDown, FaArrowUp, FaPlusCircle, FaRegTrashAlt } from 'react-icons/fa';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { Tooltip } from 'react-tooltip';
import {
  getCredentialsModels,
  useCreateGenPipeline,
  useGenCredentials,
  usePostprocessSteps,
} from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { GenPipelineCreate, GenerationParams, PostprocessStep } from '../../types';
import { AddCredentials } from './AddCredentials';

// JSON schema of a step's parameters, as served by /generate/postprocess/steps
interface StepSchema {
  properties?: Record<string, { type?: string; default?: unknown }>;
  required?: string[];
}

/**
 * Ordered list of post-treatment steps: add / remove / reorder, with a
 * small form for each step built from the registry's parameter schemas.
 */
export const StepsEditor: FC<{
  steps: PostprocessStep[];
  setSteps: (steps: PostprocessStep[]) => void;
  stepSchemas: Record<string, StepSchema>;
}> = ({ steps, setSteps, stepSchemas }) => {
  const [stepToAdd, setStepToAdd] = useState<string>('');

  const defaultParams = (name: string): Record<string, string | number | boolean> => {
    const properties = stepSchemas[name]?.properties || {};
    return Object.fromEntries(
      Object.entries(properties)
        .filter(([, schema]) => schema.default !== undefined)
        .map(([key, schema]) => [key, schema.default as string | number | boolean]),
    );
  };

  const updateStepParam = (index: number, key: string, value: string | number | boolean) => {
    setSteps(
      steps.map((step, i) =>
        i === index ? { ...step, params: { ...step.params, [key]: value } } : step,
      ),
    );
  };

  const move = (index: number, direction: number) => {
    const target = index + direction;
    if (target < 0 || target >= steps.length) return;
    const reordered = [...steps];
    [reordered[index], reordered[target]] = [reordered[target], reordered[index]];
    setSteps(reordered);
  };

  const paramInput = (
    index: number,
    key: string,
    type: string | undefined,
    value: string | number | boolean | undefined,
  ) => {
    if (type === 'boolean')
      return (
        <label className="d-flex align-items-center gap-1 m-0">
          <input
            type="checkbox"
            checked={Boolean(value)}
            onChange={(e) => updateStepParam(index, key, e.target.checked)}
          />
          {key}
        </label>
      );
    return (
      <label className="d-flex align-items-center gap-1 m-0">
        {key}
        <input
          type={type === 'integer' || type === 'number' ? 'number' : 'text'}
          className="form-control form-control-sm"
          style={{ width: type === 'integer' || type === 'number' ? '5em' : '10em' }}
          value={value === undefined ? '' : String(value)}
          onChange={(e) =>
            updateStepParam(
              index,
              key,
              type === 'integer' || type === 'number' ? Number(e.target.value) : e.target.value,
            )
          }
        />
      </label>
    );
  };

  return (
    <div>
      {steps.length === 0 && (
        <div className="text-muted small mb-1">
          No post-treatment: the raw output is used directly
        </div>
      )}
      {steps.map((step, index) => (
        <div key={index} className="d-flex align-items-center gap-2 mb-1 flex-wrap">
          <span className="badge bg-secondary">{index + 1}</span>
          <b>{step.name}</b>
          {Object.entries(stepSchemas[step.name]?.properties || {}).map(([key, schema]) =>
            paramInput(
              index,
              key,
              schema.type,
              (step.params as Record<string, string | number | boolean>)?.[key],
            ),
          )}
          <button type="button" className="btn btn-sm p-0" onClick={() => move(index, -1)}>
            <FaArrowUp />
          </button>
          <button type="button" className="btn btn-sm p-0" onClick={() => move(index, 1)}>
            <FaArrowDown />
          </button>
          <button
            type="button"
            className="btn btn-sm p-0 text-danger"
            onClick={() => setSteps(steps.filter((_, i) => i !== index))}
          >
            <FaRegTrashAlt />
          </button>
        </div>
      ))}
      <div className="d-flex align-items-center gap-2 mt-1">
        <select
          className="form-select form-select-sm w-auto"
          value={stepToAdd}
          onChange={(e) => setStepToAdd(e.target.value)}
        >
          <option value="">Add a step...</option>
          {Object.keys(stepSchemas).map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="btn btn-sm btn-outline-primary"
          disabled={!stepToAdd}
          onClick={() => {
            setSteps([...steps, { name: stepToAdd, params: defaultParams(stepToAdd) }]);
            setStepToAdd('');
          }}
        >
          <FaPlusCircle /> Add
        </button>
      </div>
    </div>
  );
};

/**
 * Create a new generative pipeline: model (credentials + slug + parameters),
 * prompt template, post-treatment steps.
 */
export const GenPipelineForm: FC<{
  projectSlug: string;
  schemes: { name: string; kind: string }[];
  contextColumns: string[];
  onCreated: () => void;
  cancel: () => void;
}> = ({ projectSlug, schemes, contextColumns, onCreated, cancel }) => {
  const { notify } = useNotifications();
  const { createGenPipeline } = useCreateGenPipeline(projectSlug);
  const { postprocessSteps } = usePostprocessSteps();

  const [credentialsRefresh, setCredentialsRefresh] = useState(0);
  const { genCredentials } = useGenCredentials(credentialsRefresh);
  const [showAddCredentials, setShowAddCredentials] = useState(false);

  const [name, setName] = useState('');
  const [schemeName, setSchemeName] = useState<string>('');
  const [credentialsId, setCredentialsId] = useState<number | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [modelSlug, setModelSlug] = useState('');
  const [prompt, setPrompt] = useState('');
  const [steps, setSteps] = useState<PostprocessStep[]>([
    { name: 'strip', params: {} },
    { name: 'exact_match', params: { ignore_case: true } },
  ]);
  const [parameters, setParameters] = useState<GenerationParams>({});
  const [showAdvanced, setShowAdvanced] = useState(false);

  const multiclassSchemes = schemes.filter((s) => s.kind === 'multiclass');

  // fetch the models exposed by the selected credentials
  useEffect(() => {
    setAvailableModels([]);
    if (credentialsId === null) return;
    getCredentialsModels(credentialsId)
      .then(setAvailableModels)
      .catch((e) => notify({ type: 'warning', message: `Could not list models: ${e.message}` }));
  }, [credentialsId, notify]);

  const setParam = (key: keyof GenerationParams, raw: string) => {
    setParameters((prev) => ({ ...prev, [key]: raw === '' ? null : Number(raw) }));
  };

  const numberParam = (key: keyof GenerationParams, label: string, step = 0.1) => (
    <div className="me-3">
      <label className="form-label mb-0">{label}</label>
      <input
        type="number"
        step={step}
        className="form-control"
        style={{ width: '7em' }}
        value={
          parameters[key] === null || parameters[key] === undefined ? '' : String(parameters[key])
        }
        onChange={(e) => setParam(key, e.target.value)}
      />
    </div>
  );

  const submit = async () => {
    const pipeline: GenPipelineCreate = {
      name: name,
      scheme_name: schemeName === '' ? null : schemeName,
      credentials_id: credentialsId as number,
      model_slug: modelSlug,
      parameters: parameters,
      prompt: prompt,
      postprocess: steps,
    };
    const created = await createGenPipeline(pipeline);
    if (created !== null) onCreated();
  };

  return (
    <div>
      <div className="mb-2">
        <label className="form-label">Pipeline name</label>
        <input
          type="text"
          className="form-control"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
      </div>

      <div className="mb-2">
        <label className="form-label">
          Scheme{' '}
          <a className="gen-scheme-help">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".gen-scheme-help" place="top">
            Bind the pipeline to a multiclass scheme (outputs are mapped to its labels), or keep it
            free (the post-treated text is the output)
          </Tooltip>
        </label>
        <select
          className="form-select"
          value={schemeName}
          onChange={(e) => {
            setSchemeName(e.target.value);
            // free generation has no codebook to match
            if (e.target.value === '')
              setSteps((prev) => prev.filter((s) => !s.name.includes('match')));
          }}
        >
          <option value="">Free generation (no scheme)</option>
          {multiclassSchemes.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name}
            </option>
          ))}
        </select>
      </div>

      <div className="mb-2">
        <label className="form-label">Credentials (endpoint + key)</label>
        <div className="d-flex gap-2 align-items-center">
          <select
            className="form-select"
            value={credentialsId === null ? '' : String(credentialsId)}
            onChange={(e) =>
              setCredentialsId(e.target.value === '' ? null : Number(e.target.value))
            }
          >
            <option value="">Select credentials</option>
            {(genCredentials || []).map((c) => (
              <option key={c.id} value={c.id}>
                {c.name} {c.kind === 'instance' ? '(instance)' : ''} — {c.endpoint}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="btn btn-outline-primary text-nowrap"
            onClick={() => setShowAddCredentials(!showAddCredentials)}
          >
            <FaPlusCircle /> New
          </button>
        </div>
        {showAddCredentials && (
          <div className="border rounded p-2 mt-2">
            <AddCredentials
              onSuccess={() => {
                setShowAddCredentials(false);
                setCredentialsRefresh((k) => k + 1);
              }}
            />
          </div>
        )}
      </div>

      <div className="mb-2">
        <label className="form-label">Model</label>
        {availableModels.length > 0 ? (
          <select
            className="form-select"
            value={modelSlug}
            onChange={(e) => setModelSlug(e.target.value)}
          >
            <option value="">Select a model</option>
            {availableModels.map((slug) => (
              <option key={slug} value={slug}>
                {slug}
              </option>
            ))}
          </select>
        ) : (
          <input
            type="text"
            className="form-control"
            placeholder="model identifier, e.g. gpt-4.1-mini"
            value={modelSlug}
            onChange={(e) => setModelSlug(e.target.value)}
          />
        )}
      </div>

      <div className="mb-2">
        <label className="form-label">Generation parameters (empty = provider default)</label>
        <div className="d-flex flex-wrap align-items-end">
          {numberParam('temperature', 'Temperature')}
          {numberParam('max_tokens', 'Max tokens', 1)}
          {numberParam('seed', 'Seed', 1)}
          <div className="me-3">
            <label className="d-flex align-items-center gap-1 mb-2">
              <input
                type="checkbox"
                checked={parameters.thinking === true}
                onChange={(e) =>
                  setParameters((prev) => ({ ...prev, thinking: e.target.checked ? true : null }))
                }
              />
              thinking
            </label>
          </div>
          <button
            type="button"
            className="btn btn-link btn-sm mb-2"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? 'Hide advanced' : 'Advanced...'}
          </button>
        </div>
        {showAdvanced && (
          <div className="d-flex flex-wrap align-items-end">
            {numberParam('top_p', 'Top-p')}
            {numberParam('presence_penalty', 'Presence penalty')}
            {numberParam('frequency_penalty', 'Frequency penalty')}
          </div>
        )}
      </div>

      <div className="mb-2">
        <label className="form-label">
          Prompt template{' '}
          <a className="gen-prompt-help">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".gen-prompt-help" place="top">
            [[TEXT]] is replaced by the element text (appended at the end if absent); context
            columns can be inserted the same way
          </Tooltip>
        </label>
        <div className="mb-1">
          {['TEXT', ...contextColumns].map((tag) => (
            <button
              key={tag}
              type="button"
              className="btn btn-sm btn-outline-secondary me-1"
              onClick={() => setPrompt(prompt + `[[${tag}]]`)}
            >
              {tag}
            </button>
          ))}
        </div>
        <textarea
          className="form-control"
          rows={5}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="e.g. Classify this text as Positive or Negative. Answer with the label only.&#10;&#10;[[TEXT]]"
        />
      </div>

      <div className="mb-3">
        <label className="form-label">Post-treatment of the raw output</label>
        <StepsEditor
          steps={steps}
          setSteps={setSteps}
          stepSchemas={(postprocessSteps || {}) as Record<string, StepSchema>}
        />
      </div>

      <div className="d-flex gap-2">
        <button
          type="button"
          className="btn btn-primary"
          disabled={!name || credentialsId === null || !modelSlug || !prompt}
          onClick={submit}
        >
          Create pipeline
        </button>
        <button type="button" className="btn btn-outline-secondary" onClick={cancel}>
          Cancel
        </button>
      </div>
    </div>
  );
};
