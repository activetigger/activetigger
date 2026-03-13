import { FC, useEffect, useState } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import { useParams } from 'react-router-dom';
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

  if (!projectName) return;

  return (
    <ProjectPageLayout projectName={projectName} currentAction="tag">
      {!canEdit || nbUsers < 2 ? (
        <AnnotationManagement />
      ) : (
        <Tabs className="mt-3" activeKey={activeTab} onSelect={(k) => setActiveTab(k || 'tag')}>
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
                  <AnnotationDisagreementManagement projectSlug={projectName} dataset={dataset} />
                </Tab>
                <Tab eventKey="between" title="Between schemes">
                  <SchemesComparisonManagement projectSlug={projectName} dataset={dataset} />
                </Tab>
              </Tabs>
            </Tab>
          )}
        </Tabs>
      )}
    </ProjectPageLayout>
  );
};
