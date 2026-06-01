import { FC, useEffect, useMemo } from 'react';

import { useGetProjectionData } from '../../core/api';
import { useAppContext } from '../../core/useAppContext';
import { useAuth } from '../../core/useAuth';
import { ElementOutModel } from '../../types';
import { ProjectionExplorer } from '../ProjectionExplorer';

interface DisplayProjectionProps {
  projectName: string | null;
  currentScheme: string | null;
  currentElement?: ElementOutModel | null;
}

// define the component
export const DisplayProjection: FC<DisplayProjectionProps> = ({
  projectName,
  currentScheme,
  currentElement,
}) => {
  // hook for all the parameters
  const {
    appContext: { currentProject: project, currentProjection, labelColorMapping, activeModel },
    setAppContext,
  } = useAppContext();
  const { authenticatedUser } = useAuth();

  // fetch projection data with the API (null if no model)
  const { projectionData, reFetchProjectionData } = useGetProjectionData(
    projectName,
    currentScheme,
    activeModel || null,
  );

  // available projections
  const availableProjections = useMemo(() => project?.projections, [project?.projections]);

  // scheme metadata used to enable the per-label focus selector on multilabel schemes
  const schemeInfo =
    currentScheme && project?.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme]
      : undefined;

  // fetch projection if needed and set it in the context
  useEffect(() => {
    // case a first projection is added
    if (
      authenticatedUser &&
      !currentProjection &&
      availableProjections?.available[authenticatedUser?.username]
    ) {
      reFetchProjectionData();
      setAppContext((prev) => ({ ...prev, currentProjection: projectionData || undefined }));
    }
  }, [
    availableProjections?.available,
    authenticatedUser,
    currentProjection,
    reFetchProjectionData,
    projectionData,
    setAppContext,
  ]);

  return (
    <div style={{ width: '80%' }}>
      {currentProjection ? (
        <ProjectionExplorer
          projectName={projectName}
          data={currentProjection}
          selectedId={currentElement?.element_id}
          labelColorMapping={labelColorMapping || {}}
          schemeKind={schemeInfo?.kind}
          availableLabels={(schemeInfo?.labels as string[] | undefined) ?? []}
        />
      ) : (
        <>No projection computed</>
      )}
    </div>
  );
};
