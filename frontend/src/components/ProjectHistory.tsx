import { Dispatch, FC, SetStateAction } from 'react';

import { useGetLogs } from '../core/api';
import { displayTime } from '../core/utils';

import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { Tooltip } from 'react-tooltip';
import { AppContextValue } from '../core/context';
import { useAnnotationSessionHistory } from '../core/useHistory';
import { ElementHistoryPoint } from '../types';

interface ProjectHistoryProps {
  projectSlug: string;
  history: ElementHistoryPoint[];
  setAppContext: Dispatch<SetStateAction<AppContextValue>>;
}

interface Row {
  time: string;
  user: string;
  project: string;
  action: string;
}

export const ProjectHistory: FC<ProjectHistoryProps> = ({ projectSlug, history }) => {
  // get logs
  const { logs } = useGetLogs(projectSlug, 100);

  const { clearAnnotationSessionHistory } = useAnnotationSessionHistory();

  // const columns: readonly Column<Row>[] = [
  //   {
  //     name: 'Time',
  //     key: 'time',
  //     resizable: true,
  //   },
  //   {
  //     name: 'User',
  //     key: 'user',
  //     resizable: true,
  //   },
  //   {
  //     name: 'Project',
  //     key: 'project',
  //   },
  //   {
  //     name: 'Action',
  //     key: 'action',
  //   },
  // ];
  return (
    <>
      <div className="horizontal">
        <a className="history">
          <HiOutlineQuestionMarkCircle />
        </a>
        Session counter: {history.length}
        <Tooltip anchorSelect=".history" place="right" style={{ zIndex: 99 }}>
          Number of elements that have been annotated during the current session. Sessions are used
          <br />
          to prevent users from annotating elements twice. Clear history if you want to be able to
          <br />
          re-annotate.
        </Tooltip>
        <button
          onClick={clearAnnotationSessionHistory}
          className="btn-danger"
          style={{ marginLeft: '10px' }}
        >
          Clear history
        </button>
      </div>
      <h3 className="subsection">Activity on this project</h3>
      <table id="history-table">
        <thead>
          <tr>
            <th id="time">Time</th>
            <th id="user">User</th>
            <th id="project">Project</th>
            <th id="action">Action</th>
          </tr>
        </thead>
        <tbody>
          {((logs as unknown as Row[]) || []).map((row, index) => (
            <tr className={index % 2 === 0 ? 'darker' : ''} key={index}>
              <td id="time">{displayTime(row.time)}</td>
              <td id="user">{row.user}</td>
              <td id="project">{row.project}</td>
              <td id="action">{row.action}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
};
