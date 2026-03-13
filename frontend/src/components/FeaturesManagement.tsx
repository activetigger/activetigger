import cx from 'classnames';
import { FC, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { useParams } from 'react-router-dom';
import { useDeleteFeature, useGetFeatureInfo, useResetFeatures } from '../core/api';
import { useAppContext } from '../core/useAppContext';
import { sortDatesAsStrings } from '../core/utils';
import { FeatureDescriptionModelOut } from '../types';
import { ButtonNewFeature } from './ButtonNewFeature';
import { ModelParametersTab } from './ModelParametersTab';
import { ModelsPillDisplay } from './ModelsPillDisplay';

export default function SimpleTable(data: FeatureDescriptionModelOut) {
  return (
    <div className="m-3 w-75">
      <ModelParametersTab
        params={{
          ...data.parameters,
        }}
      />
    </div>
  );
}

export const FeaturesManagement: FC = () => {
  const { projectName } = useParams();

  // get element from the state
  const {
    appContext: { currentProject: project, isComputing },
  } = useAppContext();

  // API calls
  const { featuresInfo } = useGetFeatureInfo(projectName || null, project);
  const deleteFeature = useDeleteFeature(projectName || null);
  const resetFeatures = useResetFeatures(projectName || null);

  // show the menu
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);
  const [showResetModal, setShowResetModal] = useState(false);

  const deleteSelectedFeature = async (element: string) => {
    await deleteFeature(element);
    setSelectedFeature(null);
  };

  if (!project) {
    return <div>No project selected</div>;
  }
  return (
    <div className="row">
      <ModelsPillDisplay
        modelNames={Object.values(featuresInfo || {})
          .sort((featureA, featureB) => sortDatesAsStrings(featureA?.time, featureB?.time, true))
          .map((feature) => (feature && feature.name ? feature.name : ''))}
        currentModelName={selectedFeature}
        setCurrentModelName={setSelectedFeature}
        deleteModelFunction={deleteSelectedFeature}
      >
        <ButtonNewFeature
          projectSlug={projectName || ''}
          className={cx('model-pill ', isComputing ? 'disabled' : '')}
        />
      </ModelsPillDisplay>
      {/* Display computing features */}
      {Object.entries(project?.features.training).map(([key, element]) => (
        <div className="card text-bg-light m-3 bg-warning w-75" key={key}>
          <div className="d-flex m-2 align-items-center">
            Currently computing {element ? element.name : ''}
            {element?.progress ? ` (${element.progress}%)` : ''}
          </div>
        </div>
      ))}
      {featuresInfo &&
        selectedFeature &&
        SimpleTable(featuresInfo[selectedFeature] as FeatureDescriptionModelOut)}
      <div className="mt-3">
        <button className="btn-danger" onClick={() => setShowResetModal(true)}>
          Reset all features
        </button>
      </div>
      <Modal show={showResetModal} onHide={() => setShowResetModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Reset all features</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          This will delete all computed features and recreate the features file. Continue?
          <div className="horizontal">
            <button onClick={() => setShowResetModal(false)} style={{ flex: '1 1 auto' }}>
              Cancel
            </button>
            <button
              className="btn-danger"
              style={{ flex: '1 1 auto' }}
              onClick={() => {
                resetFeatures();
                setSelectedFeature(null);
                setShowResetModal(false);
              }}
            >
              Confirm
            </button>
          </div>
        </Modal.Body>
      </Modal>
    </div>
  );
};
