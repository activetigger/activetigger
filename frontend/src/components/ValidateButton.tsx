import cx from 'classnames';
import { FC } from 'react';
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
  const { computeModelPrediction } = useComputeModelPrediction(projectName || null, 16);
  return (
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
  );
};
