import { FC, useCallback, useEffect, useState } from 'react';
import Modal from 'react-bootstrap/Modal';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import { useBlocker, useParams } from 'react-router-dom';
import { useAppContext } from '../core/useAppContext';

import { useLocation } from 'react-router-dom';
import { AnnotationDisagreementManagement } from '../components/Annotation/AnnotationDisagreementManagement';
import { AnnotationManagement } from '../components/Annotation/AnnotationManagement';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { SchemesComparisonManagement } from '../components/SchemesComparisonManagement';

/**
 * Annotation page
 */
export const ProjectTagPage: FC = () => {
  // parameters
  const { projectName } = useParams();
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const tab = queryParams.get('tab');
  const {
    appContext: { currentProject: project, displayConfig },
  } = useAppContext();
  const canEdit = displayConfig.interfaceType !== 'annotator';
  const [activeTab, setActiveTab] = useState<string>('tag');
  useEffect(() => {
    setActiveTab(tab || 'tag');
  }, [tab]);

  // nb users
  const nbUsers = project?.users?.users ? Object.keys(project.users?.users).length : 0;
  const [dataset, setDataset] = useState('train');

  const isValid = project?.params.valid;
  const isTest = project?.params.test;

  // track unsaved changes in the curate panel
  const [hasDirtyChanges, setHasDirtyChanges] = useState(false);
  const [pendingTab, setPendingTab] = useState<string | null>(null);

  const handleDirtyChange = useCallback((dirty: boolean) => {
    setHasDirtyChanges(dirty);
  }, []);

  // block route navigation (sidebar) when there are unsaved changes
  const blocker = useBlocker(({ currentLocation, nextLocation }) => {
    return currentLocation.pathname !== nextLocation.pathname && hasDirtyChanges;
  });

  // intercept tab switches
  const handleTabSelect = (key: string | null) => {
    const k = key || 'tag';
    if (hasDirtyChanges && k !== activeTab) {
      setPendingTab(k);
    } else {
      setActiveTab(k);
    }
  };

  if (!projectName) return;

  return (
    <ProjectPageLayout projectName={projectName} currentAction="tag">
      {!canEdit || nbUsers < 2 ? (
        <AnnotationManagement />
      ) : (
        <Tabs className="mt-3" activeKey={activeTab} onSelect={handleTabSelect}>
          <Tab eventKey="tag" title="Tag">
            <AnnotationManagement />
          </Tab>

          {nbUsers > 1 && (
            <Tab eventKey="curate" title="Curate">
              <div className="parameter-div">
                <label className="form-label label-small-gray">Dataset</label>
                <select
                  className="form-select"
                  value={dataset}
                  onChange={(e) => setDataset(e.target.value)}
                >
                  <option value="train">train</option>
                  {isValid && <option value="valid">validation</option>}
                  {isTest && <option value="test">test</option>}
                </select>
              </div>
              <Tabs id="panel" className="mt-3" defaultActiveKey="scheme">
                <Tab eventKey="scheme" title="Current scheme">
                  <AnnotationDisagreementManagement
                    projectSlug={projectName}
                    dataset={dataset}
                    onDirtyChange={handleDirtyChange}
                  />
                </Tab>
                <Tab eventKey="between" title="Between schemes">
                  <SchemesComparisonManagement projectSlug={projectName} dataset={dataset} />
                </Tab>
              </Tabs>
            </Tab>
          )}
        </Tabs>
      )}

      {/* Modal for tab switch with unsaved changes */}
      <Modal show={pendingTab !== null} onHide={() => setPendingTab(null)}>
        <Modal.Header>
          <Modal.Title>Unsaved changes</Modal.Title>
        </Modal.Header>
        <Modal.Body>Are you sure you want to leave? Unsaved modifications will be lost.</Modal.Body>
        <Modal.Footer>
          <button className="btn btn-secondary" onClick={() => setPendingTab(null)}>
            Cancel
          </button>
          <button
            className="btn btn-danger"
            onClick={() => {
              setHasDirtyChanges(false);
              setActiveTab(pendingTab!);
              setPendingTab(null);
            }}
          >
            Leave
          </button>
        </Modal.Footer>
      </Modal>

      {/* Modal for route navigation with unsaved changes */}
      <Modal show={blocker.state === 'blocked'} onHide={blocker.reset}>
        <Modal.Header>
          <Modal.Title>Unsaved changes</Modal.Title>
        </Modal.Header>
        <Modal.Body>Are you sure you want to leave? Unsaved modifications will be lost.</Modal.Body>
        <Modal.Footer>
          <button className="btn btn-secondary" onClick={blocker.reset}>
            Cancel
          </button>
          <button className="btn btn-danger" onClick={blocker.proceed}>
            Leave
          </button>
        </Modal.Footer>
      </Modal>
    </ProjectPageLayout>
  );
};
