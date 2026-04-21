import cx from 'classnames';
import { Dispatch, FC, ReactNode, SetStateAction, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { FaRegTrashAlt } from 'react-icons/fa';

interface ModelsNameInput {
  modelNames: string[];
  currentModelName: string | null;
  setCurrentModelName: Dispatch<SetStateAction<string | null>>;
  deleteModelFunction: (model_name: string) => Promise<boolean | null | undefined | void> | void;
  children?: ReactNode;
}

export const ModelsPillDisplay: FC<ModelsNameInput> = ({
  modelNames,
  currentModelName,
  setCurrentModelName,
  deleteModelFunction,
  children,
}) => {
  const [modelToDelete, setModelToDelete] = useState<string | null>(null);

  return (
    <div className="model-pill-selection">
      {(modelNames || []).map((name) => (
        <button
          key={name}
          className={cx('ms-0 model-pill', currentModelName === name && 'selected')}
          onClick={() => setCurrentModelName(name)}
        >
          {name}

          <span
            onClick={(e) => {
              e.stopPropagation();
              setModelToDelete(name);
            }}
            className="mx-2"
          >
            <FaRegTrashAlt size={12} />
          </span>
        </button>
      ))}

      {children}

      <Modal show={!!modelToDelete} onHide={() => setModelToDelete(null)}>
        <Modal.Header closeButton>
          <Modal.Title>Delete the model</Modal.Title>
        </Modal.Header>

        <Modal.Body>
          <p>
            Are you sure you want to delete <b>{modelToDelete}</b>?
          </p>

          <button
            className="btn-submit-danger"
            onClick={async () => {
              setCurrentModelName(null);
              if (modelToDelete) {
                await deleteModelFunction(modelToDelete);
              }
              setModelToDelete(null);
            }}
          >
            Delete
          </button>
        </Modal.Body>
      </Modal>
    </div>
  );
};
