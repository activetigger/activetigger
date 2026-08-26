import { FC, useState } from 'react';
import { Link, useParams } from 'react-router-dom';

import { useAppContext } from '../core/useAppContext';
import { DisplayTrainingProcesses } from './DisplayTrainingProcesses';
import { ImportPredictionDataset } from './forms/ImportPredictionDataset';
import { ValidateButtons } from './ValidateButton';

/**
 * Prediction UI for a quickmodel. Mirrors ModelPredict (BERT/NER):
 *  - a "Predict on the entire dataset" button that runs the
 *    PredictWithFeatures pipeline over the whole source dataset,
 *  - an "External dataset" flow that uploads a file and predicts on it
 *    (reuses ImportPredictionDataset with kind="quick").
 *
 * Progress is polled through project.quickmodel.training. Exports live
 * on the Export page — see ProjectExportPage.
 */
export const QuickModelPredict: FC<{ currentModel: string | null }> = ({ currentModel }) => {
  const { projectName: projectSlug } = useParams();
  const {
    appContext: { currentScheme, currentProject: project, isComputing },
  } = useAppContext();

  const [displayExternalForm, setDisplayExternalForm] = useState<boolean>(false);

  const trainingMap = project?.quickmodel?.training as
    | Record<string, { [k: string]: string | number | null | undefined }>
    | undefined;

  const currentModelInfo = project?.quickmodel?.available?.[currentScheme || '']?.find(
    (m) => m.name === currentModel,
  ) as { predicted_all?: boolean; predicted_external?: boolean } | undefined;
  const predictedAll = Boolean(currentModelInfo?.predicted_all);
  const predictedExternal = Boolean(currentModelInfo?.predicted_external);

  if (!currentModel) return null;

  return (
    <div>
      <div className="horizontal align-items-center gap-2 mt-4 wrap">
        <ValidateButtons
          modelName={currentModel}
          kind="quick"
          id="compute-prediction-whole-quickmodel"
          buttonLabel="Predict on the entire dataset"
          dataset="all"
          batchInput={false}
          existingPrediction={predictedAll}
        />
        <button
          className="btn-primary-action"
          onClick={() => setDisplayExternalForm((v) => !v)}
          disabled={isComputing}
        >
          Prediction on an external dataset
        </button>
      </div>
      {(predictedAll || predictedExternal) && (
        <div className="alert alert-info mt-3 py-2 small mb-0" role="alert">
          A prediction already exists for this model
          {predictedAll && predictedExternal
            ? ' (entire dataset and external dataset)'
            : predictedAll
              ? ' (entire dataset)'
              : ' (external dataset)'}
          . Go to the <Link to={`/projects/${projectSlug}/export`}>Export page</Link> to download
          it.
        </div>
      )}
      {displayExternalForm && (
        <div className="mt-3">
          <ImportPredictionDataset
            projectSlug={projectSlug || ''}
            modelName={currentModel}
            scheme={currentScheme || ''}
            kind="quick"
          />
        </div>
      )}
      <DisplayTrainingProcesses
        projectSlug={projectSlug || null}
        processes={trainingMap}
        processStatus="predicting"
        displayStopButton={isComputing}
        showLossChart={false}
        stopKind="quickmodel"
      />
    </div>
  );
};
