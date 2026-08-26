import { FC, useState } from 'react';
import { Button, Modal } from 'react-bootstrap';
import { Link, useParams } from 'react-router-dom';

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
  const [showOverwriteConfirm, setShowOverwriteConfirm] = useState(false);
  // Look up the right manager's flags. NER models live under
  // `nermodels`; everything else under `languagemodels`.
  const availabilityMap =
    kind === 'ner' ? project?.nermodels?.available : project?.languagemodels?.available;
  const modelStatus =
    (currentScheme && currentModel && availabilityMap?.[currentScheme]?.[currentModel]) || null;
  const availablePredictionAll = Boolean(modelStatus?.['predicted_all']);
  const availablePredictionExternal = Boolean(modelStatus?.['predicted_external']);
  // Polling source for the in-progress prediction badge — same split.
  const trainingMap =
    kind === 'ner' ? project?.nermodels?.training : project?.languagemodels?.training;

  const launchWholeDatasetPrediction = () => {
    setDisplayExternalForm(false);
    computeModelPrediction(currentModel, 'all', currentScheme || '', kind);
  };

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
              if (availablePredictionAll) {
                setShowOverwriteConfirm(true);
                return;
              }
              launchWholeDatasetPrediction();
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
      {model && (availablePredictionAll || availablePredictionExternal) && (
        <div className="alert alert-info mt-3 py-2 small mb-0" role="alert">
          A prediction already exists for this model
          {availablePredictionAll && availablePredictionExternal
            ? ' (entire dataset and external dataset)'
            : availablePredictionAll
              ? ' (entire dataset)'
              : ' (external dataset)'}
          . Go to the <Link to={`/projects/${projectSlug}/export`}>Export page</Link> to download
          it.
        </div>
      )}
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
      <Modal show={showOverwriteConfirm} onHide={() => setShowOverwriteConfirm(false)} centered>
        <Modal.Header closeButton>
          <Modal.Title>Overwrite existing prediction?</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          A prediction on the entire dataset already exists for this model. Computing it again will
          overwrite the previous one.
        </Modal.Body>
        <Modal.Footer>
          <Button
            variant="danger"
            onClick={() => {
              setShowOverwriteConfirm(false);
              launchWholeDatasetPrediction();
            }}
          >
            Predict
          </Button>
          <Button variant="secondary" onClick={() => setShowOverwriteConfirm(false)}>
            Cancel
          </Button>
        </Modal.Footer>
      </Modal>
    </div>
  );
};
