import { FC, useState } from 'react';
import { useParams } from 'react-router-dom';

import { useComputeModelPrediction, useModelInformations } from '../core/api';
import { useAppContext } from '../core/useAppContext';
import { DisplayTrainingProcesses } from './DisplayTrainingProcesses';
import { ImportPredictionDataset } from './forms/ImportPredictionDataset';

export const ModelPredict: FC<{ currentModel: string | null }> = ({ currentModel }) => {
  const { projectName: projectSlug } = useParams();

  const [batchSize, setBatchSize] = useState<number>(32);

  const {
    appContext: { currentScheme, currentProject: project, isComputing },
  } = useAppContext();

  // available labels from context
  const { model } = useModelInformations(
    projectSlug || null,
    currentModel || null,
    'bert',
    isComputing,
  );

  // compute model prediction
  const { computeModelPrediction } = useComputeModelPrediction(projectSlug || null, batchSize);

  // display external form
  const [displayExternalForm, setDisplayExternalForm] = useState<boolean>(false);
  const availablePredictionExternal =
    (currentScheme &&
      currentModel &&
      project?.languagemodels?.available?.[currentScheme]?.[currentModel]?.[
        'predicted_external'
      ]) ??
    false;

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
            Prediction external dataset
          </button>
        )}
        {model && (
          <button
            className="btn-primary-action mt-4"
            onClick={() => {
              setDisplayExternalForm(false);
              computeModelPrediction(currentModel, 'all', currentScheme || '', 'bert');
            }}
            disabled={isComputing}
          >
            Prediction complete dataset (+ imported)
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
          />
        )}
        <DisplayTrainingProcesses
          projectSlug={projectSlug || null}
          processes={project?.languagemodels.training}
          processStatus="predicting"
          displayStopButton={isComputing}
        />
      </div>
    </div>
  );
};
