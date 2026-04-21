import { FC, useEffect } from 'react';
import { useParams } from 'react-router-dom';

import { isEqual } from 'lodash';
import { useProject } from '../core/api';
import { useAppContext } from '../core/useAppContext';
import { useAuth } from '../core/useAuth';

/**
 * Component to actualise project state in the context, called every N seconds.
 */
export const CurrentProjectState: FC = () => {
  const { projectName } = useParams();
  const { setAppContext, appContext, resetContext } = useAppContext();
  const { authenticatedUser } = useAuth();

  // api call
  const { project, reFetch } = useProject(projectName);

  // set a default scheme if there is none
  useEffect(() => {
    const availableSchemes = appContext.currentProject
      ? Object.keys(appContext.currentProject.schemes.available)
      : [];
    // case of there is no selected scheme and schemes are available
    if (!appContext.currentScheme && availableSchemes.length > 0) {
      setAppContext((state) => ({
        ...state,
        currentScheme: availableSchemes[0],
      }));
    }
  }, [appContext.currentScheme, setAppContext, appContext.currentProject]);

  // reset context if project change
  useEffect(() => {
    if (
      appContext.currentProject &&
      projectName !== appContext.currentProject.params.project_slug
    ) {
      resetContext();
    }
  }, [projectName, appContext.currentProject, resetContext]);

  // update isComputing context value
  useEffect(() => {
    const isComputing =
      project && authenticatedUser && authenticatedUser.username
        ? authenticatedUser.username in project.languagemodels.training ||
          authenticatedUser.username in project.quickmodel.training ||
          authenticatedUser.username in project.projections.training ||
          authenticatedUser.username in project.bertopic.training ||
          Object.values(project.features.training).length > 0
        : false;
    setAppContext((prev) => {
      if (!isEqual(prev.currentProject, project)) {
        return { ...prev, currentProject: project, isComputing };
      }
      if (prev.isComputing !== isComputing) return { ...prev, isComputing };
      return prev;
    });
  }, [project, setAppContext, authenticatedUser]);

  // get project state every time interval
  useEffect(() => {
    //expose refetch method into context
    setAppContext((prev) => ({ ...prev, reFetchCurrentProject: reFetch }));
    // execute a fetch call to update project data every 2000ms
    const intervalId = setInterval(reFetch, 2000);
    // useEffect can return a method which is executed when the component is unmounted
    return () => {
      clearInterval(intervalId);
    };
  }, [reFetch, setAppContext]);

  return null;
};
