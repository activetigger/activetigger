import { FC, useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';

import { LabelsManagement } from '../components/LabelsManagement';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { useAppContext } from '../core/useAppContext';

import { CodebookDisplay } from '../components/CodeBookDisplay';
import { SchemesManagement } from '../components/SchemesManagement';
import { useAuth } from '../core/useAuth';
import { reorderLabels } from '../core/utils';

/**
 * Component to display the project page
 */
export const ProjectPage: FC = () => {
  // get data
  const { projectName: projectSlug } = useParams();
  const {
    appContext: { currentScheme, currentProject: project, displayConfig },
    setAppContext,
  } = useAppContext();
  const { authenticatedUser } = useAuth();
  // define variables
  const kindScheme =
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].kind
      : '';
  const availableLabels = useMemo(
    () =>
      currentScheme && project && project.schemes.available[currentScheme]
        ? project.schemes.available[currentScheme].labels || []
        : [],
    [currentScheme, project],
  );

  // sort labels according to the displayConfig
  const availableLabelsSorted = reorderLabels(
    availableLabels as string[],
    displayConfig.labelsOrder || [],
  );

  // redirect if at least 2 tags

  // get the fact that we come from the create page
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [fromCreatePage, setFromCreatePage] = useState<boolean>(false);
  if (!fromCreatePage && searchParams.get('fromCreatePage') === 'true') {
    setFromCreatePage(true);
  }

  // if conditions, navigate to the tag page
  useEffect(() => {
    if (currentScheme && fromCreatePage && availableLabels.length > 1) {
      navigate(`/projects/${projectSlug}/tag`);
      setFromCreatePage(false);
    }
  }, [fromCreatePage, availableLabels, navigate, projectSlug, currentScheme]);

  if (!projectSlug || !project) return;

  return (
    <ProjectPageLayout projectName={projectSlug}>
      <div className="container-fluid d-flex justify-content-center">
        <SchemesManagement
          projectSlug={projectSlug}
          canEdit={displayConfig.interfaceType !== 'annotator'}
          username={authenticatedUser?.username || null}
        />
      </div>

      <CodebookDisplay
        projectSlug={projectSlug}
        currentScheme={currentScheme || null}
        canEdit={displayConfig.interfaceType !== 'annotator'}
      />

      {availableLabels.length === 0 && (
        <div className="alert alert-info col-12 mt-2">
          No labels available for this scheme. Add labels to start annotation.
        </div>
      )}

      <LabelsManagement
        projectSlug={projectSlug}
        currentScheme={currentScheme || null}
        availableLabels={availableLabelsSorted}
        kindScheme={kindScheme as string}
        setAppContext={setAppContext}
        canEdit={displayConfig.interfaceType !== 'annotator'}
      />
    </ProjectPageLayout>
  );
};
