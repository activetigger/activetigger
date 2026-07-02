import { FC, useCallback, useEffect, useMemo } from 'react';

import { useGetProjectionData } from '../../core/api';
import { useAppContext } from '../../core/useAppContext';
import { ElementOutModel } from '../../types';
import { ModelsPillDisplay } from '../ModelsPillDisplay';
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
    appContext: {
      currentProject: project,
      currentProjection,
      currentProjectionName,
      labelColorMapping,
      activeModel,
    },
    setAppContext,
  } = useAppContext();

  // available projections (name -> id) from server state
  const availableProjections = useMemo(() => project?.projections, [project?.projections]);
  const availableNames = useMemo(
    () => Object.keys(availableProjections?.available || {}),
    [availableProjections?.available],
  );

  // pick a default active projection if none selected yet
  useEffect(() => {
    if (!currentProjectionName && availableNames.length > 0) {
      setAppContext((prev) => ({ ...prev, currentProjectionName: availableNames[0] }));
    } else if (currentProjectionName && !availableNames.includes(currentProjectionName)) {
      setAppContext((prev) => ({
        ...prev,
        currentProjectionName: availableNames[0] || null,
        currentProjection: undefined,
      }));
    }
  }, [availableNames, currentProjectionName, setAppContext]);

  const setCurrentProjectionName = useCallback(
    (nameOrUpdater: React.SetStateAction<string | null>) => {
      setAppContext((prev) => {
        const next =
          typeof nameOrUpdater === 'function'
            ? (nameOrUpdater as (v: string | null) => string | null)(
                prev.currentProjectionName || null,
              )
            : nameOrUpdater;
        return { ...prev, currentProjectionName: next, currentProjection: undefined };
      });
    },
    [setAppContext],
  );

  // fetch projection data with the API (null if no model)
  const { projectionData, reFetchProjectionData } = useGetProjectionData(
    projectName,
    currentScheme,
    currentProjectionName || null,
    activeModel || null,
  );

  // scheme metadata used to enable the per-label focus selector on multilabel schemes
  const schemeInfo =
    currentScheme && project?.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme]
      : undefined;

  // pick up freshly fetched projection into context
  useEffect(() => {
    if (!currentProjectionName) return;

    // no projection yet in context — take the fetched one
    if (!currentProjection && projectionData) {
      setAppContext((prev) => ({ ...prev, currentProjection: projectionData }));
      return;
    }

    // stored projection no longer matches the selected name — refetch
    const expectedId = availableProjections?.available[currentProjectionName];
    if (currentProjection && expectedId !== undefined && currentProjection.status !== expectedId) {
      reFetchProjectionData();
      if (projectionData) {
        setAppContext((prev) => ({ ...prev, currentProjection: projectionData }));
      }
    }
  }, [
    availableProjections?.available,
    currentProjection,
    currentProjectionName,
    projectionData,
    reFetchProjectionData,
    setAppContext,
  ]);

  return (
    <div style={{ width: '80%' }}>
      {availableNames.length > 0 && (
        <div className="d-flex mb-2 flex-wrap align-items-center" style={{ gap: 8 }}>
          <ModelsPillDisplay
            modelNames={availableNames}
            currentModelName={currentProjectionName || null}
            setCurrentModelName={setCurrentProjectionName}
          />
        </div>
      )}
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
