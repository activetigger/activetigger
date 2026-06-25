import cx from 'classnames';
import { FC, useState } from 'react';
import { Button, Modal } from 'react-bootstrap';
import { GrValidate } from 'react-icons/gr';

import { useParams } from 'react-router-dom';
import { CSSProperties } from 'styled-components';
import { useComputeModelPrediction } from '../core/api';
import { useAppContext } from '../core/useAppContext';

interface validateButtonsProps {
  modelName: string | null;
  kind: string | null;
  className?: string;
  id?: string;
  buttonLabel?: string;
  style?: CSSProperties;
  batchInput?: boolean;
  existingPrediction?: boolean;
}

export const ValidateButtons: FC<validateButtonsProps> = ({
  modelName,
  kind,
  className,
  id,
  buttonLabel,
  style,
  batchInput = true,
  existingPrediction = false,
}) => {
  const {
    appContext: { currentScheme, isComputing },
    setAppContext,
  } = useAppContext();
  const { projectName } = useParams();
  const [batchSize, setBatchSize] = useState(16);
  const [showOverwriteConfirm, setShowOverwriteConfirm] = useState(false);
  const { computeModelPrediction } = useComputeModelPrediction(projectName || null, batchSize);

  const launchPrediction = () => {
    setAppContext((prev) => ({ ...prev, isComputing: true }));
    computeModelPrediction(modelName || '', 'annotable', currentScheme || '', kind);
  };

  return (
    <div className="d-flex align-items-center gap-2">
      <button
        className={cx(className ? className : 'btn-primary-action')}
        style={style ? style : { color: 'white' }}
        onClick={() => {
          if (existingPrediction) {
            setShowOverwriteConfirm(true);
            return;
          }
          launchPrediction();
        }}
        id={id}
        disabled={isComputing}
      >
        <GrValidate size={20} /> {buttonLabel ? buttonLabel : 'Compute predictions'}
      </button>
      {batchInput && (
        <label className="batch-size-label">
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
      <Modal show={showOverwriteConfirm} onHide={() => setShowOverwriteConfirm(false)} centered>
        <Modal.Header closeButton>
          <Modal.Title>Overwrite existing prediction?</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          A prediction already exists for this model. Computing it again will overwrite the previous
          one.
        </Modal.Body>
        <Modal.Footer>
          <Button
            variant="danger"
            onClick={() => {
              setShowOverwriteConfirm(false);
              launchPrediction();
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
