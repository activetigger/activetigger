import { FC } from 'react';
import { FaStopCircle } from 'react-icons/fa';
import { Tooltip } from 'react-tooltip';
import { useGetServer, useStopProcesses } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { useAuth } from '../../core/useAuth';
import { ProjectStateModel } from '../../types';
import { ModalErrors } from '../ModalError';
import { PossibleProjectActions } from './ProjectPageLayout';

import { useAppContext } from '../../core/useAppContext';

/* define a component for project action bar 
with the project & the current action*/
export const StatusNotch: FC<{
  projectState: ProjectStateModel | null;
  currentProjectAction?: PossibleProjectActions;
  currentMode?: string;
  currentScheme?: string;
  currentUser: string;
  developmentMode?: boolean;
}> = ({
  projectState,
  currentUser,
  //  developmentMode,
}) => {
  // function to clear history
  const {
    appContext: { currentProject, displayConfig, history },
  } = useAppContext();

  // mean annotation time over the current session history
  const durations = history
    .map((h) => h.durationMs)
    .filter((d): d is number => typeof d === 'number' && d >= 0);
  const meanAnnotationTimeMs =
    durations.length > 0 ? durations.reduce((a, b) => a + b, 0) / durations.length : null;
  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${Math.round(ms)} ms`;
    const seconds = ms / 1000;
    if (seconds < 60) return `${seconds.toFixed(1)} s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds - minutes * 60);
    return `${minutes}m ${remainingSeconds}s`;
  };
  const { authenticatedUser } = useAuth();
  const { stopProcesses } = useStopProcesses(
    projectState ? projectState.params.project_slug : null,
  );
  // const nbUsers = projectState ? projectState.users.length : 0;

  // display the number of current processes on the server
  const { queueState, gpu } = useGetServer(currentProject || null);

  // 2 types of menu
  const canEdit = authenticatedUser?.status !== 'annotator';

  // test if computation is currently undergoing
  const currentComputation =
    projectState && projectState.languagemodels
      ? currentUser in projectState.languagemodels.training ||
        (projectState.nermodels ? currentUser in projectState.nermodels.training : false) ||
        currentUser in projectState.quickmodel.training ||
        currentUser in projectState.projections.training ||
        currentUser in projectState.bertopic.training ||
        Object.values(projectState.features.training).length > 0
      : false;

  const whatComutation = (projectState: ProjectStateModel) => {
    if (currentUser in projectState.languagemodels.training) {
      return 'BERT';
    } else if (projectState.nermodels && currentUser in projectState.nermodels.training) {
      return 'NER';
    } else if (currentUser in projectState.quickmodel.training) {
      return 'Quick model';
    } else if (currentUser in projectState.projections.training) {
      return 'Projection';
    } else if (Object.values(projectState.features.training).length > 0) {
      return 'Feature';
    } else if (projectState.bertopic.training) {
      return 'BERTopic';
    }
    return '';
  };

  // display the number of current processes on the server
  const { disk } = useGetServer(projectState || null);

  // notify if disk is full
  const { notify } = useNotifications();
  if (disk ? Number(disk['proportion']) > 98 : false) {
    notify({
      message: 'Disk is almost full, please delete some files or inform the admin team',
      type: 'warning',
    });
  }

  const errors = projectState?.errors.map((arr) => arr.join(' - ')) || [];
  // NOTE: Axel: I didn't do no css refactor here cause it's highly specific
  return (
    <div id="status-notch">
      {canEdit && (
        <div style={{ display: 'flex' }}>
          {/* Display size of project 2 version (computer and smartphones) */}
          <span className="d-none d-md-inline">
            Storage: {projectState?.memory ? `${projectState.memory.toFixed(1)} Mo` : ''}
          </span>
          <span className="d-md-none">
            Storage: {projectState?.memory ? `${projectState.memory.toFixed(1)} Mo` : ''}
          </span>
          {/* Display number of process running 1 version (computer) */}
          <span className="d-none d-md-inline">
            Global queue: {Object.values(queueState || []).length}
          </span>
          {/* Display GPU memory 1 version (computer) */}
          <span className="d-none d-md-inline">
            GPU Memory:
            {gpu
              ? ` ${(gpu['total_memory'] - gpu['available_memory']).toFixed(1)} / ${gpu['total_memory']} Go`
              : ' no GPU available'}
          </span>

          {displayConfig.displayMeanAnnotationTime && (
            <span className="d-none d-md-inline">
              <span style={{ textDecoration: 'overline' }}>annotation time</span>
              {meanAnnotationTimeMs !== null ? ` ${formatDuration(meanAnnotationTimeMs)}` : ' n/a'}
            </span>
          )}

          {/* Display GPU memory 1 version (computer) */}
          {currentComputation && (
            <>
              <span id="computing-span" style={{ display: 'flex', alignItems: 'center' }}>
                <div className="spinner-border spinner-border-sm text-warning" role="status">
                  <span className="visually-hidden">Computing</span>
                </div>
                <div>
                  Computing <i>{whatComutation(projectState as ProjectStateModel)}</i>
                </div>
                <a
                  id="stop-button"
                  onClick={() => stopProcesses('all')}
                  style={{ paddingBottom: '1.5px', cursor: 'pointer', marginLeft: '8px' }}
                >
                  <FaStopCircle style={{ color: 'red' }} />
                </a>
              </span>
              <Tooltip anchorSelect="#stop-button" place="top">
                Stop the current process
              </Tooltip>
            </>
          )}
          {/* Error pop up */}
          {projectState?.errors && projectState?.errors.length > 0 && (
            <ModalErrors errors={errors} />
          )}
        </div>
      )}
    </div>
  );
};
