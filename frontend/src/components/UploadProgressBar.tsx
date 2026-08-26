import { FC } from 'react';
import ClipLoader from 'react-spinners/ClipLoader';

interface UploadProgressBarProps {
  progression: { loaded?: number; total?: number };
  cancel?: AbortController;
  // alternative to cancel for processes that are not stopped through an AbortController
  onCancel?: () => void;
  statusMessage?: string;
  showProgress?: boolean;
}

export const UploadProgressBar: FC<UploadProgressBarProps> = ({
  progression,
  cancel,
  onCancel,
  statusMessage = 'Uploading dataset',
  showProgress = true,
}) => {
  const formatProgression = (loaded: number, total: number) => {
    if (!loaded || !total || total === 0) return '--';
    return ((loaded / total) * 100).toFixed(0);
  };
  return (
    <div id="progress-bar-window">
      <div id="progress-bar-container">
        <div className="horizontal center">
          <ClipLoader /> <span>{statusMessage}</span>{' '}
        </div>
        {showProgress && (
          <div id="progress-container">
            {progression.loaded && progression.total ? (
              <span>{formatProgression(progression.loaded, progression.total)}%</span>
            ) : null}

            <progress id="upload-progress" value={progression.loaded} max={progression.total} />
          </div>
        )}
        {(cancel || onCancel) && (
          <button
            className="btn-submit-danger"
            onClick={() => {
              if (onCancel) onCancel();
              else cancel?.abort();
            }}
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  );
};
