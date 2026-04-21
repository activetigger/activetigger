import cx from 'classnames';
import { FC, useState } from 'react';
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
}

export const ValidateButtons: FC<validateButtonsProps> = ({
  modelName,
  kind,
  className,
  id,
  buttonLabel,
  style,
}) => {
  const {
    appContext: { currentScheme, isComputing },
    setAppContext,
  } = useAppContext();
  const { projectName } = useParams();
  const [batchSize, setBatchSize] = useState(16);
  const { computeModelPrediction } = useComputeModelPrediction(projectName || null, batchSize);
  return (
    <div className="d-flex align-items-center gap-2">
      <button
        className={cx(className ? className : 'btn-primary-action')}
        style={style ? style : { color: 'white' }}
        onClick={() => {
          setAppContext((prev) => ({ ...prev, isComputing: true }));
          computeModelPrediction(modelName || '', 'annotable', currentScheme || '', kind);
        }}
        id={id}
        disabled={isComputing}
      >
        <GrValidate size={20} /> {buttonLabel ? buttonLabel : 'Compute predictions'}
      </button>
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
    </div>
  );
};
