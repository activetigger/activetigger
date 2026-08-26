import { FC, useEffect, useMemo, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { HiBan } from 'react-icons/hi';
import { LuRefreshCw } from 'react-icons/lu';
import { Link } from 'react-router-dom';
import Select from 'react-select';
import { useRetrainQuickModel, useTrainQuickModel } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { useAppContext } from '../../core/useAppContext';
import { getRandomName, pickDefaultQuickModelFeature, sortDatesAsStrings } from '../../core/utils';
import { ActiveModel, QuickModelInModel } from '../../types';
import { ButtonNewFeature } from '../ButtonNewFeature';

interface SelectActiveLearningProps {
  display: boolean;
  setActiveMenu: (value: boolean) => void;
  setSelectFirstModelTrained?: (value: boolean) => void;
  selectFirstModelTrained?: boolean;
  numberAnnotated?: number;
  authorize?: boolean;
}

type ModelOption = {
  type: string;
  value: string;
  label: string;
  time?: string; // optional because you use availableBertModels?.[e]?.time
  labels_excluded: string[]; // always present
  isDisabled?: boolean; // react-select uses it to gray out the option
};

type GroupedModels = Array<{
  label: string;
  options: ModelOption[];
}>;

export const SelectActiveLearning: FC<SelectActiveLearningProps> = ({
  display,
  setActiveMenu,
  setSelectFirstModelTrained,
  selectFirstModelTrained,
  numberAnnotated = 0,
  authorize,
}) => {
  const { notify } = useNotifications();

  const {
    appContext: { freqRefreshQuickModel, activeModel, currentScheme, currentProject: project },
    setAppContext,
  } = useAppContext();

  const projectSlug = project?.params.project_slug;

  // existing models
  const availableQuickModels = useMemo(
    () => project?.quickmodel.available[currentScheme || ''] || [],
    [project?.quickmodel, currentScheme],
  );
  const availableBertModels = useMemo(
    () => project?.languagemodels.available[currentScheme || ''] || {},
    [project?.languagemodels, currentScheme],
  );
  const availableImageModels = useMemo(
    () => project?.imagemodels?.available[currentScheme || ''] || {},
    [project?.imagemodels, currentScheme],
  );

  const groupedModels: GroupedModels = [
    {
      label: 'Quick Models',
      options: (availableQuickModels ?? [])
        .filter((e) => e?.name) // <-- protect against undefined/missing name
        .map((e) => {
          const labelsDropped = ((e.parameters.exclude_labels as string[]) || []).length > 0;
          return {
            value: e.name,
            label: labelsDropped ? e.name + ' (labels dropped)' : e.name,
            type: 'quickmodel',
            time: e.time ?? '',
            labels_excluded: e.parameters.exclude_labels as string[],
          };
        })
        .sort((quickModelA, quickModelB) =>
          sortDatesAsStrings(quickModelA?.time, quickModelB?.time, true),
        ),
    },
    {
      label: 'Language Models',
      options: Object.keys(availableBertModels || {})
        .filter((e) => e) // <-- ensure non-null
        .map((e) => {
          const excluded = availableBertModels?.[e]?.exclude_labels ?? [];
          return {
            value: e,
            label: excluded.length > 0 ? e + ' (labels dropped)' : e,
            type: 'languagemodel',
            time: availableBertModels?.[e]?.time ?? '',
            labels_excluded: excluded,
            isDisabled: !availableBertModels?.[e]?.predicted,
          };
        }),
    },
    {
      label: 'Image Models',
      options: Object.keys(availableImageModels || {})
        .filter((e) => e)
        .map((e) => {
          const excluded = availableImageModels?.[e]?.exclude_labels ?? [];
          return {
            value: e,
            label: excluded.length > 0 ? e + ' (labels dropped)' : e,
            type: 'imagemodel',
            time: availableImageModels?.[e]?.time ?? '',
            labels_excluded: excluded,
            isDisabled: !availableImageModels?.[e]?.predicted,
          };
        }),
    },
  ];

  // some listed models can't be activated because their predictions are not computed
  const hasDisabledModels = groupedModels.some((group) =>
    group.options.some((option) => option.isDisabled),
  );

  const { trainQuickModel } = useTrainQuickModel(projectSlug || null, currentScheme || null);
  const availableFeatures = project?.features.available ? project?.features.available : [];
  const startTrainQuickModel = () => {
    // default quickmodel

    if (availableFeatures.length === 0) {
      setActiveMenu(false);
      notify({
        type: 'warning',
        message: 'No set of features available.',
      });
    }

    const defaultFeature = pickDefaultQuickModelFeature(availableFeatures);
    const defaultFeatures = defaultFeature ? [defaultFeature] : availableFeatures;

    const formData = {
      name: getRandomName('QuickModel') + '-default',
      model: 'logistic-l2',
      scheme: currentScheme || '',
      params: {
        costLogL2: 1,
        costLogL1: 1,
        n_neighbors: 3,
        alpha: 1,
        n_estimators: 500,
        max_features: null,
      },
      dichotomize: null,
      features: defaultFeatures,
      cv10: false,
      standardize: false,
      balance_classes: false,
      exclude_labels: [],
    };
    trainQuickModel(formData as unknown as QuickModelInModel);
    setActiveMenu(false);
    if (setSelectFirstModelTrained) setSelectFirstModelTrained(true);
  };

  // deactivate active model if it has been removed from available models
  useEffect(() => {
    if (
      activeModel &&
      !availableQuickModels.find((model) => model.name === activeModel.value) &&
      activeModel.type === 'quickmodel'
    ) {
      setAppContext((prev) => ({ ...prev, activeModel: null }));
    }
    if (
      activeModel &&
      !Object.keys(availableBertModels)?.includes(activeModel.value) &&
      activeModel.type === 'languagemodel'
    ) {
      setAppContext((prev) => ({ ...prev, activeModel: null }));
    }
    if (
      activeModel &&
      !Object.keys(availableImageModels)?.includes(activeModel.value) &&
      activeModel.type === 'imagemodel'
    ) {
      setAppContext((prev) => ({ ...prev, activeModel: null }));
    }
  }, [availableQuickModels, activeModel, setAppContext, availableBertModels, availableImageModels]);

  // fastrack active learning model
  useEffect(() => {
    if (selectFirstModelTrained && availableQuickModels.length > 0) {
      // select the first trained model
      setAppContext((prev) => ({
        ...prev,
        activeModel: {
          type: 'quickmodel',
          value: availableQuickModels[0].name,
          label: availableQuickModels[0].name,
          time: availableQuickModels[0].time,
          labels_excluded: availableQuickModels[0].parameters.exclude_labels as string[],
        },
        selectionConfig: { ...prev.selectionConfig, mode: 'active' },
      }));
      // one-shot: reset so later quick-model retrains don't re-assert 'active' mode
      if (setSelectFirstModelTrained) setSelectFirstModelTrained(false);
    }
  }, [availableQuickModels, selectFirstModelTrained, setSelectFirstModelTrained, setAppContext]);

  // retrain quick model
  const { retrainQuickModel } = useRetrainQuickModel(projectSlug || null, currentScheme || null);
  const [updatedQuickModel, setUpdatedQuickModel] = useState(false);

  // model picked in the dropdown, only applied when the user explicitly validates it
  const [pendingModel, setPendingModel] = useState<ModelOption | null>(null);

  useEffect(() => {
    if (
      !updatedQuickModel &&
      authorize &&
      freqRefreshQuickModel &&
      activeModel &&
      numberAnnotated > 0 &&
      numberAnnotated % freqRefreshQuickModel == 0 &&
      activeModel.type === 'quickmodel'
    ) {
      setUpdatedQuickModel(true);
      retrainQuickModel(activeModel.value);
    }
    if (
      updatedQuickModel &&
      freqRefreshQuickModel &&
      numberAnnotated % freqRefreshQuickModel != 0
    ) {
      setUpdatedQuickModel(false);
    }
  }, [
    freqRefreshQuickModel,
    setUpdatedQuickModel,
    activeModel,
    updatedQuickModel,
    retrainQuickModel,
    numberAnnotated,
    projectSlug,
    authorize,
  ]);

  // function to change refresh frequency
  const refreshFreq = (newValue: number) => {
    setAppContext((prev) => ({ ...prev, freqRefreshQuickModel: newValue }));
  };
  const setActiveModel = (newValue: ActiveModel | null) => {
    setAppContext((prev) => ({ ...prev, activeModel: newValue }));
  };

  const modelTypeLabels: Record<string, string> = {
    quickmodel: 'Quick Model',
    languagemodel: 'Language Model',
    imagemodel: 'Image Model',
  };

  return (
    <Modal show={display} onHide={() => setActiveMenu(false)} id="active-modal" size="lg">
      <Modal.Header closeButton>
        <Modal.Title>Configure active learning</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        {availableFeatures.length === 0 && (
          <div className="horizontal center mb-3">
            <div>No features available for quickmodel</div>
            <ButtonNewFeature projectSlug={projectSlug || ''} />
          </div>
        )}
        {availableQuickModels.length + Object.keys(availableBertModels).length > 0 && (
          <>
            {/* current active model status */}
            <div className="horizontal center mb-2 gap-2">
              {activeModel ? (
                <>
                  <span>
                    Currently active model : <b>{activeModel.label}</b>
                    {modelTypeLabels[activeModel.type]
                      ? ` (${modelTypeLabels[activeModel.type]})`
                      : ''}
                  </span>
                  <button
                    className="btn-secondary-action d-flex align-items-center gap-1"
                    onClick={() => {
                      setActiveModel(null);
                    }}
                  >
                    <HiBan size={16} /> Deactivate
                  </button>
                </>
              ) : (
                <span>No model currently activated for active learning</span>
              )}
            </div>

            {/* retrain controls for the active quick model */}
            {activeModel?.type === 'quickmodel' && (
              <div className="horizontal center mb-3 gap-2">
                <button
                  className="btn-secondary-action d-flex align-items-center gap-1"
                  onClick={() => {
                    retrainQuickModel(activeModel.value);
                  }}
                >
                  <LuRefreshCw size={16} /> Retrain now
                </button>
                <span>or auto-retrain every</span>
                <input
                  type="number"
                  id="frequencySlider"
                  min="0"
                  max="500"
                  value={freqRefreshQuickModel}
                  onChange={(e) => {
                    refreshFreq(Number(e.currentTarget.value));
                  }}
                  step="5"
                  style={{ width: '70px' }}
                />
                <span>annotations</span>
              </div>
            )}

            <hr />

            {/* pick a model + explicit validation */}
            <div className="horizontal center mb-3 gap-2">
              <Select<ModelOption, false, { label: string; options: ModelOption[] }>
                options={groupedModels}
                value={pendingModel}
                onChange={(selectedOption) => {
                  setPendingModel(selectedOption);
                }}
                isSearchable
                placeholder="Select a model for active learning"
                className="w-50"
              />
              <button
                className="btn-primary-action"
                disabled={!pendingModel}
                onClick={() => {
                  if (pendingModel) {
                    setActiveModel(pendingModel as ActiveModel);
                    setPendingModel(null);
                  }
                }}
              >
                Use this model
              </button>
            </div>
            {hasDisabledModels && (
              <div className="horizontal center text-muted" style={{ fontSize: '0.85em' }}>
                Deactivated models need their predictions to be computed in the&nbsp;
                <Link to={`/projects/${projectSlug}/model`} onClick={() => setActiveMenu(false)}>
                  Model panel
                </Link>
              </div>
            )}
          </>
        )}

        {availableQuickModels.length === 0 && availableFeatures.length > 0 && (
          <>
            <div className="horizontal center">
              No quick model currently available. Go to model tab or
            </div>
            <button className="btn-submit" onClick={startTrainQuickModel}>
              Train a default quick model
            </button>
          </>
        )}
      </Modal.Body>
    </Modal>
  );
};
