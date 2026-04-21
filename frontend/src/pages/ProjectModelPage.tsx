import { FC, useState } from 'react';
import { Tab, Tabs } from 'react-bootstrap';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { useParams } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { ModelEvaluation } from '../components/ModelEvaluation';
import { ModelManagement } from '../components/ModelManagement';

/**
 * Component to manage model training
 */

export const ProjectModelPage: FC = () => {
  const { projectName: projectSlug } = useParams();

  const [activeKey, setActiveKey] = useState<string>('models');

  return (
    <ProjectPageLayout projectName={projectSlug} currentAction="model">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <Tabs
              id="panel"
              className="mt-3"
              activeKey={activeKey}
              onSelect={(k) => setActiveKey(k || 'models')}
            >
              <Tab eventKey="models" title="Training">
                <div className="explanations ms-3">Train quick and BERT models</div>
                <ModelManagement />
              </Tab>
              <Tab eventKey="evaluation" title="Evaluation">
                <div className="explanations ms-3">
                  Evaluate your models on annotations (train, eval and test){' '}
                  <a className="evaldataset">
                    <HiOutlineQuestionMarkCircle />
                  </a>
                  .
                </div>
                <Tooltip anchorSelect=".evaldataset" place="top">
                  Use validation statistics to choose the best model and test statistics for final
                  generalization scores of the best model (do not choose models based on this)
                  <br />
                </Tooltip>
                <ModelEvaluation />
              </Tab>
            </Tabs>
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
