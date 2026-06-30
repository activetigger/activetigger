import { FC, useState } from 'react';
import { useParams } from 'react-router-dom';

import { useComputeModelPrediction, useModelInformations } from '../core/api';
import { useAppContext } from '../core/useAppContext';
import { DisplayTrainingProcesses } from './DisplayTrainingProcesses';
import { ImportPredictionDataset } from './forms/ImportPredictionDataset';

export const ModelPredict: FC<{ currentModel: string | null; kind?: string }> = ({
  currentModel,
  kind = 'bert',
}) => {
  const { projectName: projectSlug } = useParams();

  const [batchSize, setBatchSize] = useState<number>(32);

  const {
    appContext: { currentScheme, currentProject: project, isComputing },
  } = useAppContext();

  // available labels from context
  const { model } = useModelInformations(
    projectSlug || null,
    currentModel || null,
    kind,
    isComputing,
  );

  // compute model prediction
  const { computeModelPrediction } = useComputeModelPrediction(projectSlug || null, batchSize);

  // display external form
  const [displayExternalForm, setDisplayExternalForm] = useState<boolean>(false);
  // Look up the right manager's `predicted_external` flag. NER models live
  // under `nermodels`; everything else under `languagemodels`.
  const availabilityMap =
    kind === 'ner' ? project?.nermodels?.available : project?.languagemodels?.available;
  const availablePredictionExternal =
    (currentScheme &&
      currentModel &&
      availabilityMap?.[currentScheme]?.[currentModel]?.['predicted_external']) ??
    false;
  // Polling source for the in-progress prediction badge — same split.
  const trainingMap =
    kind === 'ner' ? project?.nermodels?.training : project?.languagemodels?.training;

  return (
    <div>
      <div className="horizontal align-items-center">
        {model && (
          <button
            className="btn-primary-action mt-4"
            onClick={() => {
              setDisplayExternalForm(true);
            }}
            disabled={isComputing}
          >
            Prediction on an external dataset
          </button>
        )}
        {model && (
          <button
            className="btn-primary-action mt-4"
            disabled={isComputing}
            onClick={() => {
              setDisplayExternalForm(false);
              computeModelPrediction(currentModel, 'all', currentScheme || '', kind);
            }}
          >
            Prediction on the entire dataset
          </button>
        )}
        {model && (
          <label className="batch-size-label mt-4">
            batch
            <input
              type="number"
              min={1}
              max={512}
              value={batchSize}
              onChange={(e) => setBatchSize(Math.max(1, parseInt(e.target.value) || 1))}
              title="Batch size for prediction"
              disabled={isComputing}
            />
          </label>
        )}
      </div>
      <div>
        {model && displayExternalForm && (
          <ImportPredictionDataset
            projectSlug={projectSlug || ''}
            modelName={currentModel || ''}
            scheme={currentScheme || ''}
            availablePredictionExternal={availablePredictionExternal || false}
            batchSize={batchSize}
            kind={kind}
          />
        )}
        <DisplayTrainingProcesses
          projectSlug={projectSlug || null}
          processes={trainingMap}
          processStatus="predicting"
          displayStopButton={isComputing}
        />
      </div>
    </div>
  );
};
