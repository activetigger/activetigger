import { FC, useState } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import { useParams } from 'react-router-dom';
import { BertopicPage } from '../pages/BertopicPage';

import { DataTabular } from '../components/DataTabular';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { ProjectionManagement } from '../components/ProjectionManagement';
import { useAppContext } from '../core/useAppContext';

/**
 * Component to display the export page
 */
export const ProjectExplorePage: FC = () => {
  const { projectName } = useParams();
  const {
    appContext: { currentScheme, currentProject: project, phase },
    setAppContext,
  } = useAppContext();
  const availableLabels =
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].labels
      : [];
  const [kindScheme] = useState<string>(
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].kind || 'multiclass'
      : 'multiclass',
  );
  const availableFeatures = project?.features.available ? project?.features.available : [];

  return (
    <ProjectPageLayout projectName={projectName} currentAction="explore">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <Tabs className="mt-3" defaultActiveKey="tabular">
              <Tab eventKey="tabular" title="Tabular view">
                <div className="explanations">Explore your data as a table</div>
                <DataTabular
                  projectSlug={projectName || ''}
                  currentScheme={currentScheme || ''}
                  availableLabels={availableLabels}
                  availableUsers={project?.users?.users ?? []}
                  kindScheme={kindScheme}
                  isValid={project?.params.valid || false}
                  isTest={project?.params.test || false}
                  currentDataset={phase}
                  setAppContext={setAppContext}
                />
              </Tab>
              <Tab eventKey="visualization" title="Visualization" unmountOnExit={true}>
                <div className="explanations">Explore your annotable data as a projection</div>
                <ProjectionManagement
                  projectName={projectName || null}
                  currentScheme={currentScheme || null}
                  availableFeatures={availableFeatures}
                  currentElementId={undefined}
                />
              </Tab>
              <Tab eventKey="bertopic" title="Topic model">
                <div className="explanations">
                  Explore the topics in your train data with{' '}
                  <a
                    href="https://maartengr.github.io/BERTopic/index.html"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Bertopic
                  </a>
                  . You can also find a complete tutorial on BERTopic{' '}
                  <a
                    href="https://www.css.cnrs.fr/the-general-inquirer-in-the-time-of-llms-a-bertopic-tutorial/"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    here
                  </a>
                  .
                </div>
                <BertopicPage />
              </Tab>
            </Tabs>
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
