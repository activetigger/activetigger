import { FC, useEffect, useState } from 'react';
import DataGrid, { Column, RenderEditCellProps } from 'react-data-grid';
import 'react-data-grid/lib/styles.css';
import { useBlocker } from 'react-router-dom';

import Highlighter from 'react-highlight-words';
import { MdSkipNext, MdSkipPrevious } from 'react-icons/md';

import { Modal } from 'react-bootstrap';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { Link } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import { useAddTableAnnotations, useTableElements } from '../core/api';
import { AppContextValue } from '../core/context';
import { AnnotationModel } from '../types';
import { TableFilterState, TableTagFilterSelect } from './TableTagFilterSelect';

interface Row {
  id_internal: string;
  id_external: string;
  timestamp: string;
  labels: string;
  text: string;
  user: string;
}

interface DataTabularModel {
  projectSlug: string;
  currentScheme: string;
  availableLabels: string[];
  availableUsers: string[];
  kindScheme: string;
  isValid: boolean;
  isTest: boolean;
  currentDataset: string;
  setAppContext: React.Dispatch<React.SetStateAction<AppContextValue>>;
}

export const DataTabular: FC<DataTabularModel> = ({
  projectSlug,
  currentScheme,
  availableLabels,
  availableUsers,
  kindScheme,
  isValid,
  isTest,
  currentDataset,
  setAppContext,
}) => {
  // data modification management
  const [modifiedRows, setModifiedRows] = useState<Record<string, AnnotationModel>>({});

  // change the dataset of the context
  const changeDataSet = (dataset: string) => {
    setAppContext((prev: AppContextValue) => ({
      ...prev,
      phase: dataset,
    }));
  };

  const blocker = useBlocker(({ currentLocation, nextLocation }) => {
    return (
      currentLocation.pathname !== nextLocation.pathname && Object.values(modifiedRows).length > 0
    );
  });

  //   const availableLabels =
  //     (currentScheme && project?.schemes?.available?.[currentScheme]?.labels) ?? [];

  // selection elements
  const [page, setPage] = useState<number | null>(1);
  const [contains, setContains] = useState<string | null>(null);
  const [pageSize, setPageSize] = useState(20);
  const [filter, setFilter] = useState<TableFilterState>({
    recent: false,
    labels: [],
    users: [],
  });
  const [changeTrigger, setChangeTrigger] = useState<boolean>(false);

  // get API elements when table shape change
  const {
    table,
    getPage,
    total: totalElement,
  } = useTableElements(
    projectSlug,
    currentScheme,
    page,
    pageSize,
    currentDataset,
    contains,
    filter.recent,
    filter.labels,
    filter.users,
  );

  const [rows, setRows] = useState<Row[]>([]);

  // update rows only when a even trigger the update table
  useEffect(() => {
    if (table) {
      setRows(table as Row[]);
    }
  }, [table, currentDataset]);

  useEffect(() => {
    if (page !== null) getPage({ pageIndex: page, pageSize });
  }, [page, pageSize, getPage, currentDataset, changeTrigger]);

  // define table
  const columns: readonly Column<Row>[] = [
    {
      key: 'id',
      name: 'id',
      resizable: true,
      width: 180,
      renderCell: (props) => (
        <div className={props.row.id_internal in modifiedRows ? 'modified-cell' : ''}>
          <Link to={`/projects/${projectSlug}/tag/${props.row.id_internal}`}>
            {props.row.id_external}
          </Link>
        </div>
      ),
    },
    {
      key: 'labels',
      name: kindScheme === 'multiclass' ? 'Label ✎' : 'Label',
      resizable: true,

      renderCell: (props) => (
        <div
          style={{
            maxHeight: '100%',
            width: '100%',
            whiteSpace: 'wrap',
            overflowY: 'auto',
            userSelect: 'none',
          }}
        >
          {props.row.labels}
        </div>
      ),
      renderEditCell: kindScheme === 'multiclass' ? renderDropdown : undefined,
      width: 100,
    },
    {
      key: 'user',
      name: 'User',
      resizable: true,
      renderCell: (props) => (
        <div style={{ textAlign: 'center', width: '100%' }}>{props.row.user}</div>
      ),
      width: 100,
    },
    {
      key: 'text',
      name: 'Text',
      resizable: true,

      renderCell: (props) => (
        <div
          style={{
            maxHeight: '100%',
            width: '100%',
            whiteSpace: 'wrap',
            overflowY: 'auto',
            userSelect: 'none',
          }}
        >
          <Highlighter
            highlightClassName="Search"
            searchWords={contains && isValidRegex(contains) ? [contains.replace('ALL:', '')] : []}
            autoEscape={false}
            textToHighlight={props.row.text}
            highlightStyle={{
              backgroundColor: 'yellow',
              margin: '0px',
              padding: '0px',
            }}
            caseSensitive={true}
          />
        </div>
      ),
    },
    {
      key: 'comment',
      name: 'Comment',
      resizable: true,
      width: 100,
    },
    { key: 'timestamp', name: 'Changed', resizable: true, width: 100 },
  ];

  // specific function to have a select component
  function renderDropdown({ row, onRowChange }: RenderEditCellProps<Row>) {
    return (
      <select
        value={row.labels}
        onChange={(event) => {
          onRowChange({ ...row, labels: event.target.value }, true);
          setModifiedRows((prevState) => ({
            ...prevState,
            [row.id_internal]: {
              element_id: row.id_internal,
              label: event.target.value,
              scheme: currentScheme as string,
              project_slug: projectSlug as string,
              dataset: currentDataset,
            },
          }));
        }}
        autoFocus
      >
        <option></option>
        {(availableLabels as string[]).map((l) => (
          <option key={l} value={l}>
            {l}
          </option>
        ))}
      </select>
    );
  }

  // send changes
  const { addTableAnnotations } = useAddTableAnnotations(
    projectSlug || null,
    currentScheme || null,
    currentDataset || null,
  );
  function validateChanges() {
    addTableAnnotations(Object.values(modifiedRows)); // send the modifications
    setModifiedRows({}); // reset modified rows
  }

  function range(start: number, end: number) {
    return Array.from({ length: end - start }, (_, i) => start + i);
  }

  const isValidRegex = (pattern: string) => {
    try {
      new RegExp(pattern);
      return true;
    } catch (e) {
      return false;
    }
  };

  return (
    <>
      <div className="horizontal wrap" id="tabular-view-control-panel">
        <div>
          <label>Dataset</label>
          <select
            className="form-select"
            value={currentDataset}
            onChange={(e) => {
              setPage(1);
              changeDataSet(e.target.value);
            }}
          >
            <option value="train">train</option>
            {isValid && <option value="valid">validation</option>}
            {isTest && <option value="test">test</option>}
          </select>
        </div>
        <div style={{ minWidth: 220 }}>
          <label>Filter by Tag/Users</label>
          <TableTagFilterSelect
            availableLabels={availableLabels}
            availableUsers={availableUsers}
            onChange={(f) => {
              setPage(1);
              setFilter(f);
            }}
          />
        </div>
        <div>
          <label htmlFor="filter-input">
            Filter by content
            <HiOutlineQuestionMarkCircle className="search" />
            {!isValidRegex(contains || '') && <span className="badge danger">Regex not valid</span>}
          </label>
          <input
            placeholder="Regex filter on text"
            id="filter-input"
            onChange={(e) => {
              setPage(1);
              setContains(e.target.value);
            }}
          />
        </div>
      </div>
      <div className="horizontal wrap" id="tabular-view-page-control">
        <div>
          <label>Page size</label>
          <select
            onChange={(e) => {
              setPage(1);
              setPageSize(Number(e.target.value));
            }}
            className="form-select"
            value={pageSize}
          >
            {[10, 20, 50].map((e) => (
              <option key={e}>{e}</option>
            ))}
          </select>
        </div>
        <div>
          <label>Page</label>
          <select
            onChange={(e) => {
              setPage(Number(e.target.value));
            }}
            value={page || '1'}
          >
            {range(1, totalElement > 0 ? Math.ceil(totalElement / pageSize) + 1 : 1).map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="horizontal center">
        {Object.keys(modifiedRows).length > 0 && (
          <button
            className="btn-primary-action"
            onClick={() => {
              validateChanges();
              setChangeTrigger(!changeTrigger);
            }}
          >
            Validate changes
          </button>
        )}
      </div>
      <span className="explanations">Total: {totalElement}</span>

      <DataGrid
        className="fill-grid rdg-light"
        style={{ backgroundColor: 'white' }}
        columns={columns}
        rows={rows}
        rowHeight={80}
        onRowsChange={(e) => {
          setRows(e);
        }}
      />
      <div className="horizontal center">
        <button
          className="transparent-background"
          onClick={() => (page && page > 1 ? setPage(page - 1) : setPage(1))}
        >
          <MdSkipPrevious size={30} />
        </button>{' '}
        Change page{' '}
        <button
          className="transparent-background"
          onClick={() => (page ? setPage(page + 1) : setPage(1))}
        >
          <MdSkipNext size={30} />
        </button>
      </div>

      <Modal show={blocker.state === 'blocked'} onHide={blocker.reset}>
        <Modal.Header>
          <Modal.Title>You are leaving the page</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          Do you want to change the page ? You will lose all your changes if you proceed
        </Modal.Body>
        <Modal.Footer>
          <button onClick={blocker.reset}>No</button>
          <button onClick={blocker.proceed}>Yes</button>
        </Modal.Footer>
      </Modal>
      <Tooltip anchorSelect=".search">
        Special search: ALL= in all text/label, COMMENT= in comments, HAS_COMMENT
      </Tooltip>
    </>
  );
};
