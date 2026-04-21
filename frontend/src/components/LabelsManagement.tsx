import { FC, useEffect, useState } from 'react';
import { FaPlusCircle, FaRegTrashAlt } from 'react-icons/fa';

import { MdDriveFileRenameOutline } from 'react-icons/md';
import { useAddLabel, useDeleteLabel, useRenameLabel, useStatistics } from '../core/api';
import { useNotifications } from '../core/notifications';

import { Modal } from 'react-bootstrap';
import { ReactSortable } from 'react-sortablejs';
import { AppContextValue } from '../core/context';

/**
 * Component to manage one label
 */

interface LabelsManagementProps {
  projectSlug: string | null;
  currentScheme: string | null;
  availableLabels: string[];
  kindScheme: string;
  setAppContext: React.Dispatch<React.SetStateAction<AppContextValue>>;
  canEdit?: boolean;
}

interface LabelCardProps {
  label: string;
  countTrain: number;
  countValid?: number;
  countTest?: number;
  removeLabel: (label: string) => void;
  renameLabel: (formerLabel: string, newLabel: string) => void;
  canEdit?: boolean;
}

interface LabelType {
  id: number;
  label: string;
}

export const LabelCard: FC<LabelCardProps> = ({
  label,
  countTrain,
  countValid,
  countTest,
  removeLabel,
  renameLabel,
  canEdit,
}) => {
  const [showRename, setShowRename] = useState(false);
  const [showDelete, setShowDelete] = useState(false);
  const [newLabel, setNewLabel] = useState(label);
  return (
    <tr key={label} className="content">
      <td className="label-col">{label}</td>
      <td className="dataset-col">{countTrain ? countTrain : 0}</td>
      <td className="dataset-col">{countValid ? countValid : 0}</td>
      <td className="dataset-col">{countTest ? countTest : 0}</td>
      {canEdit && (
        <>
          <td className="text-center">
            <button className="btn btn-link p-0" title="Delete" onClick={() => setShowDelete(true)}>
              <FaRegTrashAlt />
            </button>
            <button
              className="btn btn-link p-0"
              title="Rename"
              onClick={() => setShowRename(!showRename)}
            >
              <MdDriveFileRenameOutline size={21} />
            </button>
          </td>
        </>
      )}
      <Modal show={showDelete} onHide={() => setShowDelete(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Delete a label</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          Are you sure you want to delete the label <b>{label}</b>?
          <button
            className="btn-submit-danger"
            onClick={() => {
              removeLabel(label);
              setShowDelete(false);
            }}
          >
            Delete
          </button>
        </Modal.Body>
      </Modal>
      <Modal show={showRename} onHide={() => setShowRename(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Rename label {label}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <input
            type="text"
            placeholder="Enter new label"
            onChange={(e) => setNewLabel(e.target.value)}
          />
          <button
            onClick={() => {
              renameLabel(label, newLabel);
              setShowRename(false);
            }}
            className="btn-submit"
          >
            Rename
          </button>
        </Modal.Body>
      </Modal>
    </tr>
  );
};

/**
 * Component to manage the labels of a project
 */

export const LabelsManagement: FC<LabelsManagementProps> = ({
  projectSlug,
  currentScheme,
  availableLabels,
  kindScheme,
  setAppContext,
  canEdit,
}) => {
  const { notify } = useNotifications();

  const { statistics, reFetchStatistics } = useStatistics(projectSlug, currentScheme);

  // hooks to manage labels
  const { addLabel } = useAddLabel(projectSlug || null, currentScheme || null);
  const { deleteLabel } = useDeleteLabel(projectSlug || null, currentScheme || null);
  const { renameLabel } = useRenameLabel(projectSlug || null, currentScheme || null);

  const [labels, setLabels] = useState<LabelType[]>(
    availableLabels.map((label, index) => ({
      id: index,
      label: label,
    })),
  );

  // update labels when availableLabels change
  useEffect(() => {
    reFetchStatistics();
    setLabels(
      availableLabels.map((label, index) => ({
        id: index,
        label: label,
      })),
    );
  }, [availableLabels, reFetchStatistics]);

  // manage label creation
  const [createLabelValue, setCreateLabelValue] = useState('');
  const handleCreateLabelChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.value.includes('|')) {
      notify({ type: 'error', message: 'Labels name cannot contain "|"' });
      return;
    }
    setCreateLabelValue(event.target.value);
  };
  const createLabel = () => {
    addLabel(createLabelValue);
    setCreateLabelValue('');
  };

  // useEffect(() => {
  //   reFetchStatistics();
  // }, [reFetchStatistics, currentScheme, availableLabels]);

  // update the labels in the state and context
  const updateLabels = (newLabels: LabelType[]) => {
    setLabels(newLabels);
    setAppContext((state) => ({
      ...state,
      displayConfig: {
        ...state.displayConfig,
        labelsOrder: newLabels.map((e) => e.label),
      },
    }));
  };

  return (
    <>
      <table id="label-table" className="mt-1">
        <thead>
          <tr>
            <th scope="col" className="label-col">
              <span className="explanations">({kindScheme})</span> Label
            </th>
            <th scope="col" className="dataset-col">
              Train
            </th>
            <th scope="col" className="dataset-col">
              Valid
            </th>
            <th scope="col" className="dataset-col">
              Test
            </th>
            <th scope="col" className="empty-col"></th>
          </tr>
        </thead>
        <ReactSortable list={labels} setList={updateLabels} tag="tbody">
          {labels.map((label) => (
            <LabelCard
              key={label.label}
              label={label.label}
              removeLabel={() => {
                deleteLabel(label.label);
              }}
              renameLabel={renameLabel}
              countTrain={
                statistics ? Number(statistics['train_annotated_distribution'][label.label]) : 0
              }
              countValid={
                statistics && statistics['valid_annotated_distribution']
                  ? Number(statistics['valid_annotated_distribution'][label.label])
                  : 0
              }
              countTest={
                statistics && statistics['test_annotated_distribution']
                  ? Number(statistics['test_annotated_distribution'][label.label])
                  : 0
              }
              canEdit={canEdit}
            />
          ))}
        </ReactSortable>
        <tbody>
          <tr>
            <td className="label-col">
              <b>Annotated</b>
            </td>
            <td className="dataset-col">{statistics ? statistics['train_annotated_n'] : ''}</td>
            <td className="dataset-col">{statistics ? statistics['valid_annotated_n'] : ''}</td>
            <td className="dataset-col">{statistics ? statistics['test_annotated_n'] : ''}</td>
          </tr>
          <tr>
            <td className="label-col">
              <b>Total</b>
            </td>
            <td className="dataset-col">{statistics ? statistics['train_set_n'] : ''}</td>
            <td className="dataset-col">{statistics ? statistics['valid_set_n'] : ''}</td>
            <td className="dataset-col">{statistics ? statistics['test_set_n'] : ''}</td>
          </tr>
        </tbody>
      </table>
      {canEdit && (
        <div className="d-flex align-items-center gap-2 mt-2" style={{ maxWidth: '400px' }}>
          <FaPlusCircle size={16} className="text-muted flex-shrink-0" />
          <input
            type="text"
            id="new-label"
            value={createLabelValue}
            onChange={handleCreateLabelChange}
            placeholder="Add a new label..."
            style={{ padding: '6px 10px', fontSize: '0.9rem' }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                createLabel();
              }
            }}
          />
          <button
            onClick={createLabel}
            className="btn-primary-action flex-shrink-0"
            disabled={!createLabelValue.trim()}
          >
            Add
          </button>
        </div>
      )}
    </>
  );
};
