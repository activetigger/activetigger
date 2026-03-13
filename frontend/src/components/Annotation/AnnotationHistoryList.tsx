import classNames from 'classnames';
import { truncate } from 'lodash';
import { FC } from 'react';
import { FaRegTrashAlt } from 'react-icons/fa';
import { HiOutlineEye, HiOutlineViewGrid } from 'react-icons/hi';
import { HiOutlineTableCells } from 'react-icons/hi2';
import { Link, useParams } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import { useAppContext } from '../../core/useAppContext';
import { useAnnotationSessionHistory } from '../../core/useHistory';
import { displayTime } from '../../core/utils';
import { ElementHistoryPoint } from '../../types';
import { AnnotationIcon, NoAnnotationIcon } from '../Icons';
import { MiddleEllipsis } from './MiddleEllipsis';

const dateTimeFormat = new Intl.DateTimeFormat('en-GB', {
  year: 'numeric',
  month: 'numeric',
  day: 'numeric',
  hour: 'numeric',
  minute: 'numeric',
});
const sameDayTimeFormat = new Intl.DateTimeFormat('en-GB', {
  hour: 'numeric',
  minute: 'numeric',
});

const AnnotationHistoryEntry: FC<{ elementHistoryPoint: ElementHistoryPoint }> = ({
  elementHistoryPoint,
}) => {
  const { projectName, elementId: currentElementId } = useParams();

  const date = elementHistoryPoint.time ? new Date(elementHistoryPoint.time) : undefined;
  const now = new Date();
  const sameDay = date
    ? now.getDate() === date.getDate() &&
      now.getMonth() === date.getMonth() &&
      now.getFullYear() === date.getFullYear()
    : undefined;

  return (
    <Link
      className={classNames(
        'history-element',
        elementHistoryPoint.element_id === currentElementId && 'selected',
      )}
      to={`/projects/${projectName}/tag/${elementHistoryPoint.element_id}`}
    >
      <p>{truncate(elementHistoryPoint.element_text, { length: 100 })}</p>
      <div className="d-flex gap-1 flex-wrap position-relative w-100">
        {elementHistoryPoint.label !== undefined && (
          <span className="badge d-flex align-center gap-1">
            {elementHistoryPoint.label ? (
              <>
                <AnnotationIcon className="flex-shrink-0" />{' '}
                <MiddleEllipsis label={elementHistoryPoint.label} />
              </>
            ) : (
              <NoAnnotationIcon />
            )}
          </span>
        )}
        <span className="badge ">
          <MiddleEllipsis label={'id:' + elementHistoryPoint.element_id} />{' '}
        </span>
        {date && (
          <span className="badge ">
            {(sameDay ? sameDayTimeFormat : dateTimeFormat).format(date)}
          </span>
        )}
      </div>
    </Link>
  );
};

const AnnotationHistoryTable: FC<{ items: ElementHistoryPoint[] }> = ({ items }) => {
  const { projectName, elementId: currentElementId } = useParams();

  return (
    <table id="history-table">
      <thead>
        <tr>
          <th>Time</th>
          <th>Label</th>
          <th>Text</th>
          <th>Element ID</th>
        </tr>
      </thead>
      <tbody>
        {items.map((hp, index) => (
          <tr
            className={classNames(index % 2 === 0 ? 'darker' : '', {
              'fw-bold': hp.element_id === currentElementId,
            })}
            key={`${hp.element_id}-${index}`}
          >
            <td>{hp.time ? displayTime(hp.time) : ''}</td>
            <td>{hp.label || ''}</td>
            <td>
              <Link to={`/projects/${projectName}/tag/${hp.element_id}`}>
                {truncate(hp.element_text, { length: 80 })}
              </Link>
            </td>
            <td>{hp.element_id}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export const AnnotationHistoryList: FC = () => {
  const { appContext, setAppContext } = useAppContext();
  const { history, phase, currentProject, displayConfig } = appContext;

  const { clearAnnotationSessionHistory } = useAnnotationSessionHistory();

  const toggleViewMode = () => {
    setAppContext((prev) => ({
      ...prev,
      displayConfig: {
        ...prev.displayConfig,
        displayFormat: prev.displayConfig.displayFormat === 'table' ? 'cards' : 'table',
      },
    }));
  };

  const filteredHistory = history
    .slice(0, 100)
    .filter(
      (hp) => hp.dataset === phase && hp.project_slug === currentProject?.params.project_slug,
    );

  return (
    <div className="horizontal center flex-column">
      <div className="d-flex justify-content-start gap-4 w-100 mb-4 align-items-center">
        <button
          className="btn-secondary-action d-flex align-items-center gap-2 clearhistory"
          onClick={() => {
            clearAnnotationSessionHistory();
          }}
        >
          <FaRegTrashAlt /> Clear history ({history.length} items)
        </button>
        <Tooltip anchorSelect=".clearhistory" style={{ zIndex: 99 }}>
          Clear history to be able to see again elements during this session.
        </Tooltip>
        <button
          className="btn btn-link p-0"
          onClick={toggleViewMode}
          title={
            displayConfig.displayFormat === 'cards' ? 'Switch to table view' : 'Switch to card view'
          }
        >
          {displayConfig.displayFormat === 'cards' ? (
            <HiOutlineTableCells size={20} />
          ) : (
            <HiOutlineViewGrid size={20} />
          )}
        </button>
        <button
          className="btn btn-link p-0"
          onClick={() => {
            setAppContext((prev) => ({
              ...prev,
              displayConfig: {
                ...displayConfig,
                displayHistory: false,
              },
            }));
          }}
          title="Hide history"
        >
          <HiOutlineEye size={20} />
        </button>
      </div>
      {displayConfig.displayFormat === 'cards' ? (
        <div className="annotation-history">
          {filteredHistory.map((historyPoint, i) => {
            return (
              <AnnotationHistoryEntry
                key={`${historyPoint.element_id}-${i}`}
                elementHistoryPoint={historyPoint}
              />
            );
          })}
        </div>
      ) : (
        <div style={{ width: '100%', overflowX: 'auto' }}>
          <AnnotationHistoryTable items={filteredHistory} />
        </div>
      )}
    </div>
  );
};
