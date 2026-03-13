import { FC, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { FaPlusCircle } from 'react-icons/fa';
import { useAppContext } from '../core/useAppContext';
import { CreateNewFeature } from './forms/CreateNewFeature';
export interface ButtonNewFeatureProps {
  projectSlug: string;
  className?: string;
}

export const ButtonNewFeature: FC<ButtonNewFeatureProps> = ({ projectSlug, className }) => {
  const [displayNewFeature, setDisplayNewFeature] = useState(false);
  const {
    appContext: { currentProject },
  } = useAppContext();

  return (
    <>
      <button
        type="button"
        className={className ? className : 'btn-secondary-action'}
        onClick={() => setDisplayNewFeature(true)}
        id="create-new"
      >
        <FaPlusCircle size={18} /> Add a new feature
      </button>
      <Modal
        show={displayNewFeature}
        id="features-modal"
        onHide={() => setDisplayNewFeature(false)}
      >
        <Modal.Header closeButton>
          <Modal.Title>Add a new feature</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <CreateNewFeature
            projectName={projectSlug}
            featuresOption={currentProject?.features.options || {}}
            columns={currentProject?.params.all_columns || []}
            callback={() => setDisplayNewFeature(false)}
          />
        </Modal.Body>
      </Modal>
    </>
  );
};
