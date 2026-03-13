import { FC, useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';

import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';

import { FeaturesManagement } from '../components/FeaturesManagement';
import { EvalSetsManagement } from '../components/forms/EvalSetsManagement';
import { ImportAnnotations } from '../components/forms/ImportAnnotations';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { useAppContext } from '../core/useAppContext';

import { ProjectHistory } from '../components/ProjectHistory';
import { ProjectParameters } from '../components/ProjectParameters';

/**
 * Component to display the project page
 */
export const ProjectSettingsPage: FC = () => {
  // get data
  const { projectName: projectSlug } = useParams();
  const {
    appContext: { currentScheme, currentProject: project, history },
    setAppContext,
  } = useAppContext();

  // define variables

  const availableLabels = useMemo(
    () =>
      currentScheme && project && project.schemes.available[currentScheme]
        ? project.schemes.available[currentScheme].labels || []
        : [],
    [currentScheme, project],
  );

  // redirect if at least 2 tags

  // get the fact that we come from the create page
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [fromProjectPage, setFromProjectPage] = useState<boolean>(false);
  if (!fromProjectPage && searchParams.get('fromProjectPage') === 'true') {
    setFromProjectPage(true);
  }

  // if conditions, navigate to the tag page
  useEffect(() => {
    if (currentScheme && fromProjectPage && availableLabels.length > 1) {
      navigate(`/projects/${projectSlug}/tag`);
      setFromProjectPage(false);
    }
  }, [fromProjectPage, availableLabels, navigate, projectSlug, currentScheme]);

  if (!projectSlug || !project) return;

  return (
    <ProjectPageLayout projectName={projectSlug} currentAction="settings">
      <Tabs id="panel" className="mt-3" defaultActiveKey="parameters">
        <Tab eventKey="parameters" title="Parameters">
          <ProjectParameters project={project} projectSlug={projectSlug} />
        </Tab>
        <Tab eventKey="features" title="Features">
          <div className="explanations">Available features</div>
          <FeaturesManagement />
        </Tab>

        <Tab eventKey="import" title="Import">
          <div className="explanations">Import data to this project</div>
          <ImportAnnotations
            projectName={project.params.project_slug}
            currentScheme={currentScheme || null}
          />
          <hr />
          <EvalSetsManagement
            projectSlug={projectSlug}
            currentScheme={currentScheme || ''}
            dataset={'valid'}
            exist={project?.params.valid}
          />
          <hr />
          <EvalSetsManagement
            projectSlug={projectSlug}
            currentScheme={currentScheme || ''}
            dataset={'test'}
            exist={project?.params.test}
          />
        </Tab>

        <Tab eventKey="session" title="Session history">
          <div className="explanations">History of the current session</div>
          <ProjectHistory
            projectSlug={projectSlug}
            history={history}
            setAppContext={setAppContext}
          />
        </Tab>
      </Tabs>
    </ProjectPageLayout>
  );
};
