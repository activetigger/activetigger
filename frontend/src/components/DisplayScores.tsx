import { FC, useState } from 'react';
import { Modal } from 'react-bootstrap';
import DataGrid, { Column } from 'react-data-grid';
import { FaCloudDownloadAlt } from 'react-icons/fa';
import { Link } from 'react-router-dom';
import { MLStatisticsModel } from '../types';
import { DisplayTableStatistics } from './DisplayTableStatistics';
import { DisplayTableStatisticsReact } from './DisplayTableStatisticsReact';
import { DisplayTableStatisticsReactMultilabel } from './DisplayTableStatisticsReactMultiLabel';

export interface DisplayScoresProps {
  title: string | null;
  scores: MLStatisticsModel;
  modelName?: string;
  projectSlug?: string | null;
  dataset?: string;
  exclude_labels?: string[];
}

interface Row {
  id: string;
  label: string;
  prediction: string;
  text: string;
}

/**
 * DisplayScores component to show model statistics and false predictions.
 * It includes a table of statistics and a data grid for false predictions.
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
  const datasetClean = dataset.includes('test')
    ? 'test'
    : dataset.includes('valid')
      ? 'valid'
      : 'train';
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
  const [showFalsePredictions, setShowFalsePredictions] = useState(false);
  const columns: readonly Column<Row>[] = [
    {
      key: 'id',
      name: 'Id',
      resizable: true,
      width: 180,
      renderCell: (props) => (
        <div>
          {projectSlug ? (
            <Link to={`/projects/${projectSlug}/tag/${props.row.id}?dataset=${datasetClean}`}>
              {props.row.id}
            </Link>
          ) : (
            props.row.id
          )}
        </div>
      ),
    },
    {
      name: 'Label',
      key: 'GS-label',
      resizable: true,
    },
    {
      name: 'Prediction',
      key: 'prediction',
      resizable: true,
    },
    {
      name: 'Text',
      key: 'text',
      resizable: true,
    },
  ];
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
        <button className="btn-secondary-action" onClick={() => setShowFalsePredictions(true)}>
          Show false predictions
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

      <Modal
        show={showFalsePredictions}
        id="quickmodel-modal"
        onHide={() => setShowFalsePredictions(false)}
        centered
        size="xl"
      >
        <Modal.Header closeButton>
          <Modal.Title>False prediction of the model {modelName}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {' '}
          <DataGrid<Row>
            className="fill-grid rdg-light"
            columns={columns}
            rows={scores['false_predictions'] as Row[]}
          />
        </Modal.Body>
      </Modal>
    </div>
  );
};
