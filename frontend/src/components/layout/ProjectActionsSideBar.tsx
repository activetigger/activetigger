import classNames from 'classnames';
import { FC } from 'react';
import { FaCloudDownloadAlt } from 'react-icons/fa';
import { HiMiniRectangleGroup } from 'react-icons/hi2';
import { IoBookSharp, IoSettingsSharp } from 'react-icons/io5';
import { MdModelTraining } from 'react-icons/md';
import { PiTagDuotone } from 'react-icons/pi';
import { RiAiGenerate } from 'react-icons/ri';
import { Link } from 'react-router-dom';
import { useGetServer } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { useAuth } from '../../core/useAuth';
import { ProjectStateModel } from '../../types';
import { AnnotationIcon } from '../Icons';
import { PossibleProjectActions } from './ProjectPageLayout';

/* define a component for project action bar 
with the project & the current action*/
export const ProjectActionsSidebar: FC<{
  projectState: ProjectStateModel | null;
  currentProjectAction?: PossibleProjectActions;
  currentMode?: string;
  currentScheme?: string;
  currentUser: string;
  developmentMode?: boolean;
}> = ({
  currentProjectAction,
  projectState,
  currentScheme,
  //  developmentMode,
}) => {
  const projectSlug = projectState ? projectState.params.project_slug : null;
  const projectName = projectState ? projectState.params.project_name : null;
  const { authenticatedUser } = useAuth();

  // 2 types of menu
  const canEdit = authenticatedUser?.status !== 'annotator';

  // display the number of current processes on the server
  const { disk } = useGetServer(projectState || null);

  // notify if disk is full
  const { notify } = useNotifications();
  if (disk ? Number(disk['proportion']) > 98 : false) {
    notify({
      message: 'Disk is almost full, please delete some files or notify the admin team.',
      type: 'warning',
    });
  }

  return (
    // NOTE: Axel: I didn't do no css refactor here cause it's highly specific
    <div className={`project-sidebar flex-shrink-0 bg-light sticky-top`}>
      {canEdit && (
        <ul className="nav nav-pills mb-auto ">
          <li className="nav-item  d-none d-md-inline w-100">
            <div className="nav-link d-inline-block rounded-pill px-3 py-1 bg-light w-100">
              <div className="fw-semibold text-dark text-truncate">{projectName}</div>
              <div className="small text-primary" title={currentScheme}>
                {currentScheme}
              </div>
            </div>
          </li>
          <li className="nav-item">
            <Link
              to={`/projects/${projectSlug}`}
              className={classNames('nav-link', !currentProjectAction && 'active')}
              aria-current="page"
              title="Access and modify your project parameters"
            >
              <IoBookSharp />
              <span className="ms-1 sidemenulabel">Codebook</span>
            </Link>
          </li>
          <li className="nav-item">
            <Link
              to={`/projects/${projectSlug}/explore`}
              className={classNames('nav-link', currentProjectAction === 'explore' && 'active')}
              aria-current="page"
              title="Explore your data"
            >
              <HiMiniRectangleGroup />
              <span className="ms-1 sidemenulabel">Explore</span>
            </Link>
          </li>
          <li className="nav-item">
            <Link
              to={`/projects/${projectSlug}/tag`}
              className={classNames('nav-link', currentProjectAction === 'tag' && 'active')}
              aria-current="page"
              title="Tag your data"
            >
              <AnnotationIcon />
              <span className="ms-1 sidemenulabel">Annotate</span>
            </Link>
          </li>
          <li className="nav-item">
            <Link
              to={`/projects/${projectSlug}/model`}
              className={classNames('nav-link', currentProjectAction === 'model' && 'active')}
              aria-current="page"
              title="Manage your models"
            >
              <MdModelTraining />
              <span className="ms-1 sidemenulabel">Model</span>
            </Link>
          </li>
          {/* <li className="nav-item">
            <Link
              to={`/projects/${projectSlug}/validate`}
              className={classNames('nav-link', currentProjectAction === 'validate' && 'active')}
              aria-current="page"
              title="Test your model"
            >
              <FaListCheck />

              <span className="ms-1 sidemenulabel">Evaluate</span>
            </Link>
          </li> */}
          <li className="nav-item">
            <Link
              to={`/projects/${projectSlug}/export`}
              className={classNames('nav-link', currentProjectAction === 'export' && 'active')}
              aria-current="page"
              title="Export everything"
            >
              <FaCloudDownloadAlt />
              <span className="ms-1 sidemenulabel">Export</span>
            </Link>
          </li>

          <li className="nav-item">
            <Link
              to={`/projects/${projectSlug}/generate`}
              className={classNames('nav-link', currentProjectAction === 'generate' && 'active')}
              aria-current="page"
              title="Use generative tools to annotate your data"
              style={{ color: '#e00eebff' }}
            >
              <RiAiGenerate />
              <span className="ms-1">Generative</span>
            </Link>
          </li>
          <li className="nav-item">
            <Link
              to={`/projects/${projectSlug}/settings`}
              className={classNames('nav-link', currentProjectAction === 'settings' && 'active')}
              aria-current="page"
              title="Project settings"
            >
              <IoSettingsSharp />
              <span className="ms-1 sidemenulabel">Settings</span>
            </Link>
          </li>
        </ul>
      )}
      {!canEdit && (
        <ul className="nav nav-pills flex-column mb-auto">
          <li className="nav-item  d-none d-md-inline">
            <div
              className="nav-link d-inline-block rounded-pill px-3 py-1 bg-light"
              style={{ lineHeight: '1.1' }}
            >
              <div className="fw-semibold text-dark text-truncate">{projectSlug}</div>
              <div
                className="small text-primary"
                style={{
                  marginTop: '-2px',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
                title={currentScheme}
              >
                {currentScheme && currentScheme.length > 15
                  ? `${currentScheme.substring(0, 15)}…`
                  : currentScheme}
              </div>
            </div>
          </li>
          <li className="nav-item">
            <Link
              to={`/projects/${projectSlug}`}
              className={classNames('nav-link', !currentProjectAction && 'active')}
              aria-current="page"
              title="Access and modify your project parameters"
            >
              <IoBookSharp />
              <span className="ms-1 sidemenulabel">Codebook</span>
            </Link>
          </li>
          <li className="nav-item">
            <Link
              to={`/projects/${projectSlug}/tag`}
              className={classNames('nav-link', currentProjectAction === 'tag' && 'active')}
              aria-current="page"
              title="Tag your data"
            >
              <PiTagDuotone />
              <span className="ms-1 sidemenulabel">Annotate</span>
            </Link>
          </li>
        </ul>
      )}
    </div>
  );
};
