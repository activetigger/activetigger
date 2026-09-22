import { FC, useState } from 'react';
import { FaCloudDownloadAlt } from 'react-icons/fa';
import { MLStatisticsModel } from '../types';
import { DisplayTableStatistics } from './DisplayTableStatistics';
import { DisplayTableStatisticsReact } from './DisplayTableStatisticsReact';
import { DisplayTableStatisticsReactMultilabel } from './DisplayTableStatisticsReactMultiLabel';
import { WrongPredictionsModal } from './WrongPredictionsModal';

export interface DisplayScoresProps {
  title: string | null;
  scores: MLStatisticsModel;
  modelName?: string;
  projectSlug?: string | null;
  dataset?: string;
  exclude_labels?: string[];
}

/**
 * DisplayScores component to show model statistics and wrong predictions.
 * It includes a table of statistics and a data grid for wrong predictions.
 **/
export const DisplayScores: FC<DisplayScoresProps> = ({
  title,
  scores,
  modelName,
  projectSlug,
  dataset = 'data',
  exclude_labels,
}) => {
  const [viewTable] = useState<boolean>(false);
  const downloadModel = () => {
    if (!scores) return; // Ensure model is not null or undefined

    // Convert the model object to a JSON string
    const modelJson = JSON.stringify(scores, null, 2);

    // Create a Blob from the JSON string
    const blob = new Blob([modelJson], { type: 'application/json' });

    // Create a temporary link element
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = modelName || 'model.json';
    link.click();
  };
  const [showWrongPredictions, setShowWrongPredictions] = useState(false);
  if (!scores) return;
  return (
    <div>
      <div className="d-flex flex-column">
        {(exclude_labels || []).length > 0 && (
          <span className="explanations">
            Labels{' '}
            {(exclude_labels || []).map((l) => (
              <span className="badge" key={l}>
                {l}
              </span>
            ))}{' '}
            are excluded from training
          </span>
        )}
      </div>
      <span className="fs-5">
        Macro F1 score on {dataset.replace('_scores', '')} set : <b>{scores.f1_macro}</b>
      </span>

      {scores.training_kind === 'multilabel' ? (
        <DisplayTableStatisticsReactMultilabel scores={scores} title={title} />
      ) : viewTable ? (
        <DisplayTableStatistics scores={scores} title={title} />
      ) : (
        <DisplayTableStatisticsReact scores={scores} title={title} />
      )}
      <div>
        {/* <button
          className="btn btn-link p-0"
          onClick={() => {
            setViewTable(!viewTable);
          }}
          title="Toggle view"
        >
          <HiOutlineViewGrid size={20} />
        </button> */}
      </div>
      {scores['false_predictions'] && (
        <button className="btn-secondary-action" onClick={() => setShowWrongPredictions(true)}>
          Show wrong predictions
        </button>
      )}
      <button
        className="btn-secondary-action"
        onClick={(e) => {
          e.preventDefault();
          downloadModel();
        }}
      >
        <FaCloudDownloadAlt size={15} className="me-2" />
        Download as JSON
      </button>

      <WrongPredictionsModal
        show={showWrongPredictions}
        onHide={() => setShowWrongPredictions(false)}
        scores={scores}
        modelName={modelName}
        projectSlug={projectSlug}
        dataset={dataset}
      />
    </div>
  );
};
