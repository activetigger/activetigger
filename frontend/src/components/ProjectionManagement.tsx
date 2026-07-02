import { pick } from 'lodash';
import { FC, useCallback, useEffect, useMemo, useState } from 'react';
import { Controller, SubmitHandler, useForm } from 'react-hook-form';
import Select from 'react-select';

import chroma from 'chroma-js';
import cx from 'classnames';
import { Modal } from 'react-bootstrap';
import { FaPlusCircle } from 'react-icons/fa';
import { FaGear } from 'react-icons/fa6';
import { ModelParametersTab } from '../components/ModelParametersTab';
import {
  useAddAnnotation,
  useDeleteProjection,
  useGetProjectionData,
  useUpdateProjection,
} from '../core/api';
import { useNotifications } from '../core/notifications';
import { useAppContext } from '../core/useAppContext';
import { getRandomName } from '../core/utils';
import { ProjectionParametersModel } from '../types';
import { MulticlassInput } from './Annotation/MulticlassInput';
import { MultilabelInput } from './Annotation/MultilabelInput';
import { ButtonNewFeature } from './ButtonNewFeature';
import { ModelsPillDisplay } from './ModelsPillDisplay';
import { ProjectionExplorer } from './ProjectionExplorer';
import { StopProcessButton } from './StopProcessButton';

interface ProjectionManagementProps {
  projectName: string | null;
  projectSlug?: string;
  currentScheme: string | null;
  availableFeatures: string[];
  currentElementId?: string;
}

// define the component
export const ProjectionManagement: FC<ProjectionManagementProps> = ({
  projectName,
  currentScheme,
  availableFeatures,
  currentElementId,
}) => {
  // hook for all the parameters
  const {
    appContext: {
      currentProject: project,
      currentProjection,
      currentProjectionName,
      isComputing,
      labelColorMapping,
      activeModel,
    },
    setAppContext,
  } = useAppContext();
  const { notify } = useNotifications();

  // available projections state from server (name -> id)
  const availableProjections = useMemo(() => project?.projections, [project?.projections]);
  const availableNames = useMemo(
    () => Object.keys(availableProjections?.available || {}),
    [availableProjections?.available],
  );

  // sync currentProjectionName with what actually exists on the server
  useEffect(() => {
    // pick a default if none selected
    if (!currentProjectionName && availableNames.length > 0) {
      setAppContext((prev) => ({ ...prev, currentProjectionName: availableNames[0] }));
      return;
    }
    // drop selection if the projection no longer exists
    if (currentProjectionName && !availableNames.includes(currentProjectionName)) {
      setAppContext((prev) => ({
        ...prev,
        currentProjectionName: availableNames[0] || null,
        currentProjection: undefined,
      }));
    }
  }, [availableNames, currentProjectionName, setAppContext]);

  // fetch projection data with the API (null if no model)
  const { projectionData, reFetchProjectionData } = useGetProjectionData(
    projectName,
    currentScheme,
    currentProjectionName || null,
    activeModel || null,
  );

  const setCurrentProjectionName = useCallback(
    (nameOrUpdater: React.SetStateAction<string | null>) => {
      setAppContext((prev) => {
        const next =
          typeof nameOrUpdater === 'function'
            ? (nameOrUpdater as (v: string | null) => string | null)(
                prev.currentProjectionName || null,
              )
            : nameOrUpdater;
        return { ...prev, currentProjectionName: next, currentProjection: undefined };
      });
    },
    [setAppContext],
  );

  const deleteProjection = useDeleteProjection(projectName);

  // states for dynamic interactions
  const [forceRefresh, setForceRefresh] = useState<boolean>(false);
  const [showComputeNewProjection, setShowComputeNewProjection] = useState<boolean>(false);
  const [showParameters, setShowParameters] = useState<boolean>(false);

  // unique labels
  const uniqueLabels = useMemo(
    () => (projectionData ? [...new Set(projectionData.nodes.map((o) => o.label))] : []),
    [projectionData],
  );
  const baseColors = chroma.brewer.Dark2;

  const colormap = useMemo(() => {
    return uniqueLabels.map((_, i) => baseColors[i % baseColors.length]);
  }, [uniqueLabels, baseColors]);

  const { register, handleSubmit, watch, control, reset, setValue } =
    useForm<ProjectionParametersModel>({
      defaultValues: {
        name: getRandomName('projection'),
        method: 'umap',
        parameters: {
          //common
          n_components: 2,
          // T-SNE
          perplexity: 30,
          learning_rate: 'auto',
          init: 'random',
          // UMAP
          metric: 'cosine',
          n_neighbors: 15,
          min_dist: 0.1,
        },
        // Normalize
        normalize_features: false,
      },
    });
  const selectedMethod = watch('method'); // state for the model selected to modify parameters

  // available features
  const features = availableFeatures
    .map((e) => ({ value: e, label: e }))
    .sort((a, b) => a.label.localeCompare(b.label));

  // action when form validated
  const { updateProjection } = useUpdateProjection(projectName, currentScheme);
  const onSubmit: SubmitHandler<ProjectionParametersModel> = async (formData) => {
    // fromData has all fields whatever the selected method

    // validate the name
    const name = formData.name?.trim();
    if (!name) {
      notify({ type: 'error', message: 'Please provide a name for the projection' });
      return;
    }
    if (availableNames.includes(name)) {
      notify({ type: 'error', message: `A projection named "${name}" already exists` });
      return;
    }

    // discard unrelevant fields depending on selected method
    const relevantParams =
      selectedMethod === 'tsne'
        ? ['perplexity', 'n_components', 'learning_rate', 'init']
        : selectedMethod === 'umap'
          ? ['n_neighbors', 'min_dist', 'n_components']
          : [];
    const params = pick(formData.parameters, relevantParams);
    const data = { ...formData, name, parameters: params };
    const watchedFeatures = watch('features');
    if (watchedFeatures.length == 0) {
      notify({ type: 'error', message: 'Select at least one feature' });
      return;
    }
    await updateProjection(data);
    // pre-select the new projection so it becomes active as soon as computed
    setCurrentProjectionName(name);
    reset();
    setShowComputeNewProjection(false);
  };

  useEffect(() => {
    if (projectionData) {
      const labeledColors = uniqueLabels.reduce<Record<string, string>>(
        (acc, label, index: number) => {
          acc[label as string] = colormap[index];
          return acc;
        },
        {},
      );
      setAppContext((prev) => ({ ...prev, labelColorMapping: labeledColors }));
    }
  }, [colormap, projectionData, setAppContext, uniqueLabels]);

  // sync current projection in context with the fetched data
  useEffect(() => {
    if (!currentProjectionName) return;

    // no projection currently in context — pick up the fetched data
    if (!currentProjection && projectionData) {
      setAppContext((prev) => ({ ...prev, currentProjection: projectionData }));
      return;
    }

    // switch: server reports a different id for the selected name
    const expectedId = availableProjections?.available[currentProjectionName];
    if (currentProjection && expectedId !== undefined && currentProjection.status !== expectedId) {
      reFetchProjectionData();
      if (projectionData) {
        setAppContext((prev) => ({ ...prev, currentProjection: projectionData }));
      }
    }

    // after annotating, force refresh so the viz stays in sync
    if (currentProjection && forceRefresh) {
      reFetchProjectionData();
      if (projectionData) {
        setAppContext((prev) => ({ ...prev, currentProjection: projectionData }));
      }
      setForceRefresh(false);
    }
  }, [
    availableProjections?.available,
    currentProjection,
    currentProjectionName,
    reFetchProjectionData,
    projectionData,
    setAppContext,
    setForceRefresh,
    forceRefresh,
  ]);

  type Feature = {
    label: string;
    value: string;
  };
  const filterFeatures = (features: Feature[]) => {
    const filtered = features.filter((e) =>
      /sentence-embeddings|embeddings|fasttext/i.test(e.label),
    );
    return filtered;
  };
  const defaultFeatures = filterFeatures(features);

  // define parameters for configuration panels
  const availableLabels =
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].labels
      : [];
  const [kindScheme] = useState<string>(
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].kind || 'multiclass'
      : 'multiclass',
  );
  // post an annotation
  // hooks to manage annotation
  const { addAnnotation } = useAddAnnotation(projectName || null, currentScheme || null, 'train');

  const postAnnotation = useCallback(
    (label: string | null, elementId?: string) => {
      if (elementId) {
        addAnnotation(elementId, label, '', '');
        notify({ type: 'success', message: `Row ${elementId} was annotated` });
        setForceRefresh(true);
      }
    },
    [addAnnotation, notify],
  );

  const trainingEntries = Object.entries(availableProjections?.training || {});

  return (
    <div className="explore-container">
      <div className="d-flex my-2 flex-wrap align-items-center" style={{ gap: 8 }}>
        <ModelsPillDisplay
          modelNames={availableNames}
          currentModelName={currentProjectionName || null}
          setCurrentModelName={setCurrentProjectionName}
          deleteModelFunction={deleteProjection}
        >
          {!isComputing ? (
            <button
              onClick={() => {
                setValue('name', getRandomName('projection'));
                setShowComputeNewProjection(true);
              }}
              className="model-pill"
              id="create-new-projection"
            >
              <FaPlusCircle size={20} /> Compute new projection
            </button>
          ) : (
            <StopProcessButton projectSlug={projectName} />
          )}
        </ModelsPillDisplay>
        {projectionData && labelColorMapping && (
          <button className="btn-secondary-action" onClick={() => setShowParameters(true)}>
            <FaGear size={18} /> Parameters
          </button>
        )}
      </div>
      {trainingEntries.length > 0 && (
        <div className="small text-muted mb-2">
          Computing:{' '}
          {trainingEntries.map(([name, method]) => (
            <span key={name} className={cx('badge', 'bg-warning', 'text-dark', 'me-1')}>
              {name} ({method})
            </span>
          ))}
        </div>
      )}
      {projectionData && labelColorMapping && (
        <ProjectionExplorer
          projectName={projectName}
          data={projectionData}
          selectedId={currentElementId}
          labelColorMapping={labelColorMapping}
          schemeKind={kindScheme}
          availableLabels={availableLabels as string[]}
          containerClassName="explore-viz-container"
          vizClassName="explore-viz-column"
          panelClassName="explore-annotation-column"
        >
          {(element, clearSelection) => (
            <>
              <h5 className="subsection">Annotate this element</h5>
              <div className="annotation-block force-one-column-layout compact">
                {kindScheme == 'multiclass' && (
                  <MulticlassInput
                    elementId={element.element_id}
                    element={element}
                    postAnnotation={(label, elementId) => {
                      postAnnotation(label, elementId);
                      clearSelection();
                    }}
                    labels={availableLabels}
                    phase="train"
                  />
                )}
                {kindScheme == 'multilabel' && (
                  <MultilabelInput
                    elementId={element.element_id}
                    postAnnotation={(label, elementId) => {
                      postAnnotation(label, elementId);
                      clearSelection();
                    }}
                    labels={availableLabels}
                  />
                )}
              </div>
            </>
          )}
        </ProjectionExplorer>
      )}

      <Modal
        show={showComputeNewProjection}
        onHide={() => setShowComputeNewProjection(false)}
        size="xl"
        id="viz-projection"
      >
        <Modal.Header closeButton>
          <Modal.Title>Compute a new projection</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <form onSubmit={handleSubmit(onSubmit)}>
            <label htmlFor="projection-name">Name</label>
            <input
              id="projection-name"
              type="text"
              placeholder="e.g. umap-sbert"
              {...register('name', { required: true })}
              className="form-control mb-2"
            />
            <label htmlFor="features">Select features</label>
            <div style={{ flex: '1 1 auto' }}>
              <Controller
                name="features"
                control={control}
                defaultValue={defaultFeatures.map((e) => e.value)}
                render={({ field: { onChange, value } }) => (
                  <Select
                    options={features}
                    isMulti
                    value={features.filter((option) => value.includes(option.value))}
                    onChange={(selectedOptions) => {
                      onChange(
                        selectedOptions ? selectedOptions.map((option) => option.value) : [],
                      );
                    }}
                  />
                )}
              />
            </div>
            <ButtonNewFeature projectSlug={projectName || ''} />
            <details>
              <summary>Advanced parameters</summary>
              <label htmlFor="model">Select a model</label>
              <select id="model" {...register('method')}>
                {Object.keys(availableProjections?.options || {}).map((e) => (
                  <option key={e} value={e}>
                    {e}
                  </option>
                ))}{' '}
              </select>
              {availableProjections?.options && selectedMethod == 'tsne' && (
                <>
                  <label htmlFor="perplexity">perplexity</label>
                  <input
                    type="number"
                    step="1"
                    id="perplexity"
                    {...register('parameters.perplexity', { valueAsNumber: true })}
                  ></input>
                  <label>Learning rate</label>
                  <select {...register('parameters.learning_rate')}>
                    <option key="auto" value="auto">
                      auto
                    </option>
                  </select>
                  <label>Init</label>
                  <select {...register('parameters.init')}>
                    <option key="random" value="random">
                      random
                    </option>
                  </select>
                </>
              )}
              {availableProjections?.options && selectedMethod == 'umap' && (
                <>
                  <label htmlFor="n_neighbors">n_neighbors</label>
                  <input
                    type="number"
                    step="1"
                    id="n_neighbors"
                    {...register('parameters.n_neighbors', { valueAsNumber: true })}
                  ></input>
                  <label htmlFor="min_dist">min_dist</label>
                  <input
                    type="number"
                    id="min_dist"
                    step="0.01"
                    {...register('parameters.min_dist', { valueAsNumber: true })}
                  ></input>
                </>
              )}
              <input type="checkbox" {...register('normalize_features')} />
              <label>Feature scaling</label>
            </details>
            <button className="btn-submit">Compute</button>
          </form>
        </Modal.Body>
      </Modal>
      <Modal show={showParameters} id="parameters-modal" onHide={() => setShowParameters(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Parameters of the current visualisation</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <ModelParametersTab
            params={
              projectionData?.parameters
                ? {
                    method: projectionData.parameters.method,
                    features: projectionData.parameters.features,
                    ...projectionData.parameters.parameters,
                  }
                : ({} as Record<string, unknown>)
            }
          />
        </Modal.Body>
      </Modal>
    </div>
  );
};
