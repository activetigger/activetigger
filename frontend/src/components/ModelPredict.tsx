import { FC, useState } from 'react';
import { useParams } from 'react-router-dom';

import { useComputeModelPrediction, useModelInformations } from '../core/api';
import { useAppContext } from '../core/useAppContext';
import { DisplayTrainingProcesses } from './DisplayTrainingProcesses';
import { ImportPredictionDataset } from './forms/ImportPredictionDataset';

export const ModelPredict: FC<{ currentModel: string | null; batchSize?: number }> = ({
  currentModel,
  batchSize,
}) => {
  const { projectName: projectSlug } = useParams();

  const useBatchSize = batchSize || 32;

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

  // const availablePrediction =
  //   currentScheme &&
  //   currentModel &&
  //   project?.languagemodels.available[currentScheme] &&
  //   project?.languagemodels.available[currentScheme][currentModel]
  //     ? project?.languagemodels.available[currentScheme][currentModel]['predicted']
  //     : false;

  // compute model preduction
  const { computeModelPrediction } = useComputeModelPrediction(projectSlug || null, useBatchSize);

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
      <div className="horizontal">
        {model && (
          <button
            className="btn-primary-action mt-4"
            onClick={() => {
              console.log('clickedd');
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
            Prediction complete dataset
          </button>
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
