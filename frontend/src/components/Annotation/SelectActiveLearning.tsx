import { FC, useEffect, useMemo, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { HiBan } from 'react-icons/hi';
import { LuRefreshCw } from 'react-icons/lu';
import Select from 'react-select';
import { Tooltip } from 'react-tooltip';
import { useRetrainQuickModel, useTrainQuickModel } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { useAppContext } from '../../core/useAppContext';
import { getRandomName, sortDatesAsStrings } from '../../core/utils';
import { ActiveModel } from '../../types';
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

  const availableBertModelsWithPrediction = Object.entries(availableBertModels || {})
    .filter(([_, v]) => v && v.predicted)
    .map(([k, _]) => k);

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
      options: (availableBertModelsWithPrediction ?? [])
        .filter((e) => e) // <-- ensure non-null
        .map((e) => ({
          value: e,
          label: e,
          type: 'languagemodel',
          time: availableBertModels?.[e]?.time ?? '',
          labels_excluded: [],
        })),
    },
  ];

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
    const formData = {
      name: getRandomName('QuickModel') + '-default',
      model: 'logistic-l1',
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
      features: availableFeatures,
      cv10: false,
      standardize: false,
      balance_classes: false,
      exclude_labels: [],
    };
    trainQuickModel(formData);
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
  }, [availableQuickModels, activeModel, setAppContext, availableBertModels]);

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
    }
  }, [availableQuickModels, selectFirstModelTrained, setAppContext]);

  // retrain quick model
  const { retrainQuickModel } = useRetrainQuickModel(projectSlug || null, currentScheme || null);
  const [updatedQuickModel, setUpdatedQuickModel] = useState(false);

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
            <div className="horizontal center mb-3">
              <Select<ModelOption, false, { label: string; options: ModelOption[] }>
                options={groupedModels}
                value={activeModel as ModelOption | null}
                onChange={(selectedOption) => {
                  setActiveModel(selectedOption ? (selectedOption as ActiveModel) : null);
                }}
                isSearchable
                placeholder="Select a model for active learning"
                className="w-50"
              />

              <div>
                <button
                  className="btn-secondary-action"
                  onClick={() => {
                    setActiveModel(null);
                  }}
                >
                  <HiBan size={20} data-tooltip-id="delete-tooltip" />
                </button>
                {activeModel?.type === 'quickmodel' && (
                  <button
                    className="btn-secondary-action"
                    onClick={() => {
                      retrainQuickModel(activeModel.value);
                    }}
                  >
                    <LuRefreshCw size={20} data-tooltip-id="retrain-tooltip" />
                  </button>
                )}
                <Tooltip id="retrain-tooltip" place="bottom" content="Retrain model" />
                <Tooltip id="delete-tooltip" place="bottom" content="Deactivate model" />
              </div>
              {activeModel?.type === 'quickmodel' && (
                <div className="d-flex align-items-center ms-3">
                  <span className="me-2">Retrain every</span>
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
                    style={{ flex: '1 1 30%', width: '60px' }}
                  />
                </div>
              )}
            </div>
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
