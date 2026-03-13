import cx from 'classnames';
import { FC, useCallback, useEffect, useMemo, useState } from 'react';
import { IoIosRadioButtonOff, IoIosRadioButtonOn } from 'react-icons/io';
import { useNavigate, useParams } from 'react-router-dom';
import { useAppContext } from '../../core/useAppContext';
import { useAnnotationSessionHistory } from '../../core/useHistory';
import { reorderLabels } from '../../core/utils';
import { ElementOutModel } from '../../types';
import { ActiveAnnotationIcon, EmptyAnnotationIcon, NoAnnotationIcon } from '../Icons';
import { MiddleEllipsis } from './MiddleEllipsis';

interface MulticlassInputProps {
  elementId: string;
  labels: string[];
  postAnnotation: (label: string | null, elementId: string, comment?: string) => void;
  small?: boolean;
  phase?: string;
  element?: ElementOutModel;
  secondaryLabels?: boolean;
}

interface LabelType {
  id: number;
  label: string;
}

export const MulticlassInput: FC<MulticlassInputProps> = ({
  elementId,
  postAnnotation,
  labels,
  phase,
  element,
  small = false,
  secondaryLabels = true,
}) => {
  // get the context and set the labels
  const {
    appContext: { displayConfig, activeModel },
  } = useAppContext();
  const { projectName } = useParams();
  const navigate = useNavigate();

  const { addElementInAnnotationSessionHistory } = useAnnotationSessionHistory();

  const skipAnnotation = useCallback(() => {
    //update history
    if (element)
      addElementInAnnotationSessionHistory(
        element.element_id,
        element.text,
        undefined,
        undefined,
        true,
      );

    // move to next element
    navigate(`/projects/${projectName}/tag/`);
  }, [navigate, projectName, addElementInAnnotationSessionHistory, element]);

  //grab comment from element history once API has been modified to add this
  const [comment, setComment] = useState<string>(
    element?.history ? element.history[0]?.comment || '' : '',
  );

  //reset comment as for now it's not available in annotation history
  useEffect(() => setComment(element?.history ? element.history[0]?.comment || '' : ''), [element]);

  const availableLabels = useMemo<LabelType[]>(
    () =>
      reorderLabels(labels || [], displayConfig.labelsOrder || []).map((label, index) => ({
        id: index,
        label: label,
      })),
    [displayConfig.labelsOrder, labels],
  );

  const handleKeyboardEvents = useCallback(
    (ev: KeyboardEvent) => {
      // prevent shortkey to perturb the inputs
      const activeElement = document.activeElement;
      const isFormField =
        activeElement?.tagName === 'INPUT' ||
        activeElement?.tagName === 'TEXTAREA' ||
        activeElement?.tagName === 'SELECT';
      if (isFormField) return;

      if (activeModel && ev.code === 'KeyP') {
        postAnnotation(element?.predict.label || '', elementId, comment);
      }
      if (ev.code === 'KeyS' && skipAnnotation) {
        skipAnnotation();
      }
      if (ev.code === 'Delete') {
        postAnnotation(null, elementId, comment);
      }
      if (availableLabels.length < 10)
        availableLabels.forEach((item, i) => {
          if (ev.code === `Digit` + (i + 1) || ev.code === `Numpad` + (i + 1)) {
            postAnnotation(item.label, elementId, comment);
          }
        });
    },
    [availableLabels, postAnnotation, elementId, element, activeModel, skipAnnotation, comment],
  );

  useEffect(() => {
    // manage keyboard shortcut if less than 10 label
    if (availableLabels.length > 0 && availableLabels.length < 10) {
      document.addEventListener('keydown', handleKeyboardEvents);
    }

    return () => {
      if (availableLabels.length > 0 && availableLabels.length < 10) {
        document.removeEventListener('keydown', handleKeyboardEvents);
      }
    };
  }, [availableLabels, handleKeyboardEvents]);

  const predict_proba = element?.predict.proba ? element.predict.proba.toFixed(2) : 'NA';
  // const predict_entropy = element?.predict.entropy ? element.predict.entropy.toFixed(2) : 'NA';

  const lastAnnotation = useMemo(() => {
    return element?.history && element.history.length > 0 ? element?.history[0] : null;
  }, [element?.history]);

  return (
    <div className=" tag-action-container">
      {/* TAGS ACTIONS */}

      {
        // display buttons for label from the user
        availableLabels.map((e, i) => (
          <button
            type="button"
            key={e.label}
            value={e.label}
            className="tag-action-button btn-annotate-action"
            onClick={(v) => {
              if (element) postAnnotation(v.currentTarget.value, elementId, comment);
            }}
          >
            {displayConfig.displayAnnotation ? (
              lastAnnotation && lastAnnotation.label === e.label ? (
                <ActiveAnnotationIcon />
              ) : (
                <EmptyAnnotationIcon />
              )
            ) : null}
            <MiddleEllipsis label={e.label} />
            {availableLabels.length < 10 && <span className="badge hotkey">{i + 1}</span>}
          </button>
        ))
      }

      {/* PREDICTION */}
      {phase == 'train' && displayConfig.displayPrediction && element?.predict.label && (
        <div className="prediction-container">
          {/* {displayConfig.displayPredictionStat && (
            <small className="d-flex align-items-center gap-1 prediction-stats">
              <MdOnlinePrediction size="20" title="Prediction by model" id="prediction-icon" />{' '}
              <Tooltip anchorSelect="#prediction-icon" place="top">
                Prediction by model
              </Tooltip>
              <Tooltip anchorSelect="#predict-probability" place="top">
                prediction's probability: {predict_proba}
              </Tooltip>
            </small>
          )} */}
          <button
            type="button"
            value={element?.predict.label as unknown as string}
            className={cx(
              'btn-annotate-predicted-action tag-action-button',
              small ? ' icon-small' : '',
            )}
            onClick={(e) => {
              postAnnotation(e.currentTarget.value, elementId, comment);
            }}
          >
            <MiddleEllipsis label={element?.predict.label} />{' '}
            <span className="badge m-0 p-1" id="predict-probability">
              p={predict_proba}
            </span>
            <span className="badge hotkey">P</span>
          </button>
        </div>
      )}

      {/* NO TAG OPTION */}
      <button
        type="button"
        className="btn-annotate-general-action no-tag-action"
        onClick={() => {
          postAnnotation(null, elementId, comment);
        }}
      >
        {lastAnnotation === null ? <IoIosRadioButtonOn /> : <IoIosRadioButtonOff />}
        <span className="text">
          <NoAnnotationIcon /> No tag
        </span>
        <span className="badge hotkey">DEL</span>
      </button>

      {/* SKIP */}
      {skipAnnotation && secondaryLabels && (
        <button
          type="button"
          className="btn-annotate-general-action tag-action-button"
          onClick={() => {
            skipAnnotation();
          }}
        >
          Skip <span className="badge hotkey">S</span>
        </button>
      )}

      {/* COMMENT */}
      <textarea
        className="form-control annotation-comment"
        placeholder="Comment"
        value={comment}
        onChange={(e) => setComment(e.target.value)}
      />
    </div>
  );
};
