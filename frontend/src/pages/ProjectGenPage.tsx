import { FC, useEffect, useMemo, useState } from 'react';
import { Modal, Tab, Tabs } from 'react-bootstrap';
import DataGrid, { Column } from 'react-data-grid';
import { FaDownload, FaPlusCircle, FaRegTrashAlt } from 'react-icons/fa';
import { HiOutlineSparkles } from 'react-icons/hi';
import { useParams } from 'react-router-dom';
import PulseLoader from 'react-spinners/PulseLoader';
import { GenPipelineForm } from '../components/forms/GenPipelineForm';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { ModelsPillDisplay } from '../components/ModelsPillDisplay';
import {
  useConvertGenRun,
  useDeleteGenPipeline,
  useDeleteGenRun,
  useGenPipelines,
  useGenRuns,
  useGetGenerationsFile,
  useSandboxGenPipeline,
  useStartGenRun,
  useStopProcesses,
} from '../core/api';
import { useAppContext } from '../core/useAppContext';
import { useAuth } from '../core/useAuth';
import { GeneratedRow, GenPipeline } from '../types';

interface GeneratedTableRow {
  element_id: string;
  predicted?: string | null;
  raw: string;
  error?: string | null;
  [key: string]: unknown;
}

const scrollableCell = (content: string | null | undefined) => (
  <div style={{ maxHeight: '100%', whiteSpace: 'wrap', overflowY: 'auto', userSelect: 'none' }}>
    {content}
  </div>
);

const generatedColumns: readonly Column<GeneratedTableRow>[] = [
  { name: 'Element', key: 'element_id', resizable: true, width: '12%' },
  {
    name: 'Predicted',
    key: 'predicted',
    resizable: true,
    width: '15%',
    renderCell: ({ row }) =>
      row.predicted === null || row.predicted === undefined ? (
        <span className="badge bg-warning text-dark">NA</span>
      ) : (
        scrollableCell(row.predicted)
      ),
  },
  {
    name: 'Raw output',
    key: 'raw',
    resizable: true,
    width: '40%',
    renderCell: ({ row }) => scrollableCell(row.raw),
  },
  {
    name: 'Error',
    key: 'error',
    resizable: true,
    renderCell: ({ row }) => scrollableCell(row.error),
  },
];

/**
 * Panel to build and use generative pipelines (issue #1100):
 * pipeline pills, creation modal, sandbox, runs on the dataset.
 */
export const GenPage: FC = () => {
  const { projectName } = useParams() as { projectName: string };
  const { authenticatedUser } = useAuth();
  const {
    appContext: { currentProject },
  } = useAppContext();

  // pipelines
  const [pipelinesRefresh, setPipelinesRefresh] = useState(0);
  const { pipelines } = useGenPipelines(projectName, pipelinesRefresh);
  const [currentPipelineName, setCurrentPipelineName] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const { deleteGenPipeline } = useDeleteGenPipeline(projectName);

  // sandbox
  const { sandboxGenPipeline } = useSandboxGenPipeline(projectName);
  const [sandboxN, setSandboxN] = useState(5);
  const [sandboxRows, setSandboxRows] = useState<GeneratedRow[] | null>(null);
  const [sandboxNa, setSandboxNa] = useState<number>(0);
  const [sandboxLoading, setSandboxLoading] = useState(false);

  // runs
  const { startGenRun } = useStartGenRun(projectName);
  const [runsRefresh, setRunsRefresh] = useState(0);
  const { genRuns } = useGenRuns(projectName, runsRefresh);
  const { deleteGenRun } = useDeleteGenRun(projectName);
  const { getGenerationsFile } = useGetGenerationsFile(projectName);
  const { stopProcesses } = useStopProcesses(projectName);
  const [runDataset, setRunDataset] = useState('train');
  const [runMode, setRunMode] = useState('all');
  const [runN, setRunN] = useState<string>('');
  const [runWorkers, setRunWorkers] = useState(1);
  const { convertGenRun } = useConvertGenRun(projectName);
  const [runToConvert, setRunToConvert] = useState<number | null>(null);
  const [newSchemeName, setNewSchemeName] = useState('');

  const currentPipeline: GenPipeline | undefined = useMemo(
    () => (pipelines || []).find((p) => p.name === currentPipelineName),
    [pipelines, currentPipelineName],
  );

  const schemes = useMemo(
    () =>
      Object.entries(currentProject?.schemes.available || {}).map(([name, scheme]) => ({
        name,
        kind: (scheme as { kind?: string }).kind || 'multiclass',
      })),
    [currentProject],
  );

  // generation in progress for the current user (from the project state)
  const training = authenticatedUser?.username
    ? currentProject?.generations.training?.[authenticatedUser.username]
    : undefined;
  const isGenerating = training !== undefined;

  // refresh the runs table when a generation ends
  useEffect(() => {
    if (!isGenerating) setRunsRefresh((k) => k + 1);
  }, [isGenerating]);

  // the sandbox results belong to the selected pipeline
  useEffect(() => {
    setSandboxRows(null);
  }, [currentPipelineName]);

  const runSandbox = async () => {
    if (!currentPipeline) return;
    setSandboxLoading(true);
    setSandboxRows(null);
    const result = await sandboxGenPipeline(currentPipeline.id, {
      n_elements: sandboxN,
      mode: 'all',
      dataset: 'train',
    });
    setSandboxLoading(false);
    if (result) {
      setSandboxRows(result.rows);
      setSandboxNa(result.n_na);
    }
  };

  const launchRun = async () => {
    if (!currentPipeline) return;
    const wholeDataset = runDataset === 'all';
    const started = await startGenRun(currentPipeline.id, {
      dataset: runDataset,
      mode: wholeDataset ? 'all' : runMode,
      n_elements: wholeDataset || runN === '' ? null : Number(runN),
      n_workers: runWorkers,
    });
    if (started !== null) setRunsRefresh((k) => k + 1);
  };

  return (
    <ProjectPageLayout projectName={projectName} currentAction="generate">
      <div className="container-fluid mt-3">
        <div className="explanations">
          Build generative pipelines: a model + a prompt + post-treatments turning raw outputs into
          labels of a scheme (or free text). Test in the sandbox, then run on the dataset.
        </div>

        <Modal
          show={showCreateModal}
          id="createpipeline-modal"
          size="xl"
          onHide={() => setShowCreateModal(false)}
        >
          <Modal.Header closeButton>
            <Modal.Title>New generative pipeline</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <GenPipelineForm
              projectSlug={projectName}
              schemes={schemes}
              contextColumns={currentProject?.params.cols_context || []}
              onCreated={() => {
                setShowCreateModal(false);
                setPipelinesRefresh((k) => k + 1);
              }}
              cancel={() => setShowCreateModal(false)}
            />
          </Modal.Body>
        </Modal>

        <ModelsPillDisplay
          modelNames={(pipelines || []).map((p) => p.name)}
          currentModelName={currentPipelineName}
          setCurrentModelName={setCurrentPipelineName}
          deleteModelFunction={async (name) => {
            const pipeline = (pipelines || []).find((p) => p.name === name);
            if (pipeline && (await deleteGenPipeline(pipeline.id))) {
              setCurrentPipelineName(null);
              setPipelinesRefresh((k) => k + 1);
              setRunsRefresh((k) => k + 1);
            }
          }}
        >
          <button
            onClick={() => setShowCreateModal(true)}
            className="model-pill create-pill"
            id="create-new"
          >
            <FaPlusCircle size={20} /> New pipeline
          </button>
        </ModelsPillDisplay>

        <Tabs id="generation-panel" className="mt-3" defaultActiveKey="pipelines">
          <Tab eventKey="pipelines" title="Create & sandbox">
            {currentPipeline ? (
              <>
                <div className="card my-3">
                  <div className="card-body">
                    <h5 className="card-title">{currentPipeline.name}</h5>
                    <div className="small">
                      <b>Model</b> {currentPipeline.model_slug} <b>via</b>{' '}
                      {currentPipeline.credentials_name} ({currentPipeline.endpoint}) —{' '}
                      <b>Scheme</b>{' '}
                      {currentPipeline.scheme_name || <em>free generation (no scheme)</em>}
                    </div>
                    <div className="small mt-1">
                      <b>Parameters</b>{' '}
                      {Object.entries(currentPipeline.parameters || {})
                        .filter(([, value]) => value !== null && value !== undefined)
                        .map(([key, value]) => `${key}=${value}`)
                        .join(', ') || 'provider defaults'}
                    </div>
                    <div className="small mt-1">
                      <b>Post-treatment</b>{' '}
                      {currentPipeline.postprocess.length > 0
                        ? currentPipeline.postprocess.map((s) => s.name).join(' → ')
                        : 'none'}
                    </div>
                    <details className="mt-1">
                      <summary className="small">Prompt</summary>
                      <pre className="small bg-light p-2 mt-1">{currentPipeline.prompt}</pre>
                    </details>
                  </div>
                </div>

                <div className="card my-3">
                  <div className="card-body">
                    <h5 className="card-title">
                      <HiOutlineSparkles /> Sandbox
                    </h5>
                    <div className="d-flex align-items-end gap-2 mb-2">
                      <div>
                        <label className="form-label mb-0" htmlFor="sandbox-n">
                          Elements
                        </label>
                        <input
                          id="sandbox-n"
                          type="number"
                          min={1}
                          max={10}
                          className="form-control"
                          style={{ width: '6em' }}
                          value={sandboxN}
                          onChange={(e) => setSandboxN(Number(e.target.value))}
                        />
                      </div>
                      <button
                        className="btn btn-primary"
                        onClick={runSandbox}
                        disabled={sandboxLoading}
                      >
                        {sandboxLoading ? (
                          <PulseLoader size={8} color="white" />
                        ) : (
                          'Test on a sample'
                        )}
                      </button>
                      {sandboxRows && (
                        <span className="badge bg-warning text-dark mb-2">
                          {sandboxNa} / {sandboxRows.length} NA
                        </span>
                      )}
                    </div>
                    {sandboxRows && (
                      <DataGrid
                        className="fill-grid"
                        columns={generatedColumns}
                        rows={sandboxRows as unknown as GeneratedTableRow[]}
                        rowHeight={80}
                      />
                    )}
                  </div>
                </div>
              </>
            ) : (
              <div className="text-muted my-3">
                Select a pipeline or create a new one to test it in the sandbox
              </div>
            )}
          </Tab>

          <Tab eventKey="runs" title="Run & results">
            {currentPipeline ? (
              <div className="card my-3">
                <div className="card-body">
                  <h5 className="card-title">Run {currentPipeline.name} on the dataset</h5>
                  {isGenerating ? (
                    <div className="d-flex align-items-center gap-3">
                      <PulseLoader />
                      <span>
                        Generating with <b>{training?.pipeline_name}</b> — progress{' '}
                        {training?.progress ?? 0}%
                      </span>
                      <button
                        className="btn btn-danger"
                        onClick={() => stopProcesses('generation')}
                      >
                        Stop
                      </button>
                    </div>
                  ) : (
                    <div className="d-flex align-items-end gap-2 flex-wrap">
                      <div>
                        <label className="form-label mb-0" htmlFor="run-dataset">
                          Dataset
                        </label>
                        <select
                          id="run-dataset"
                          className="form-select"
                          value={runDataset}
                          onChange={(e) => setRunDataset(e.target.value)}
                        >
                          <option value="train">train</option>
                          <option value="annotable">annotable (train+valid+test)</option>
                          <option value="all">all (complete dataset)</option>
                        </select>
                      </div>
                      <div>
                        <label className="form-label mb-0" htmlFor="run-mode">
                          Elements
                        </label>
                        <select
                          id="run-mode"
                          className="form-select"
                          value={runMode}
                          disabled={runDataset === 'all'}
                          onChange={(e) => setRunMode(e.target.value)}
                        >
                          <option value="all">all</option>
                          <option value="tagged">tagged</option>
                          <option value="untagged">untagged</option>
                        </select>
                      </div>
                      <div>
                        <label className="form-label mb-0" htmlFor="run-n">
                          Limit (empty = all)
                        </label>
                        <input
                          id="run-n"
                          type="number"
                          min={1}
                          className="form-control"
                          style={{ width: '8em' }}
                          value={runN}
                          disabled={runDataset === 'all'}
                          onChange={(e) => setRunN(e.target.value)}
                        />
                      </div>
                      <div>
                        <label className="form-label mb-0" htmlFor="run-workers">
                          Parallel calls
                        </label>
                        <input
                          id="run-workers"
                          type="number"
                          min={1}
                          max={10}
                          className="form-control"
                          style={{ width: '6em' }}
                          value={runWorkers}
                          onChange={(e) => setRunWorkers(Number(e.target.value))}
                        />
                      </div>
                      <button className="btn btn-primary" onClick={launchRun}>
                        Start generation
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-muted my-3">Select a pipeline to run it on the dataset</div>
            )}

            <div className="card my-3">
              <div className="card-body">
                <h5 className="card-title">Runs</h5>
                {(genRuns || []).length === 0 ? (
                  <em className="text-muted">No generation run yet</em>
                ) : (
                  <table className="table table-sm align-middle">
                    <thead>
                      <tr>
                        <th>Pipeline</th>
                        <th>Time</th>
                        <th>By</th>
                        <th>Selection</th>
                        <th>Elements</th>
                        <th>Status</th>
                        <th>NA</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {(genRuns || []).map((run) => (
                        <tr key={run.id}>
                          <td>{run.pipeline_name}</td>
                          <td>{new Date(run.time).toLocaleString()}</td>
                          <td>{run.user_name}</td>
                          <td>
                            {run.dataset} / {run.mode}
                          </td>
                          <td>{run.n_elements}</td>
                          <td>
                            <span
                              className={
                                'badge ' +
                                (run.status === 'done'
                                  ? 'bg-success'
                                  : run.status === 'running'
                                    ? 'bg-info'
                                    : 'bg-danger')
                              }
                            >
                              {run.status}
                            </span>
                          </td>
                          <td>{run.n_na ?? ''}</td>
                          <td className="text-end">
                            <button
                              className="btn btn-sm btn-outline-primary me-1"
                              disabled={run.status !== 'done'}
                              onClick={() => {
                                setNewSchemeName('');
                                setRunToConvert(run.id);
                              }}
                            >
                              To scheme
                            </button>
                            <button
                              className="btn btn-sm btn-outline-secondary me-1"
                              disabled={run.status === 'running'}
                              onClick={() => getGenerationsFile(run.id)}
                            >
                              <FaDownload />
                            </button>
                            <button
                              className="btn btn-sm btn-outline-danger"
                              disabled={run.status === 'running'}
                              onClick={async () => {
                                if (await deleteGenRun(run.id)) setRunsRefresh((k) => k + 1);
                              }}
                            >
                              <FaRegTrashAlt />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          </Tab>
        </Tabs>

        <Modal show={runToConvert !== null} onHide={() => setRunToConvert(null)}>
          <Modal.Header closeButton>
            <Modal.Title>Create a scheme from this run</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <p className="text-muted small">
              The predicted labels of the run become annotations in a new scheme (NA outputs are
              skipped)
            </p>
            <input
              type="text"
              className="form-control"
              placeholder="Name of the new scheme"
              value={newSchemeName}
              onChange={(e) => setNewSchemeName(e.target.value)}
            />
            <button
              className="btn btn-primary mt-2"
              disabled={!newSchemeName}
              onClick={async () => {
                if (runToConvert !== null && (await convertGenRun(runToConvert, newSchemeName))) {
                  setRunToConvert(null);
                }
              }}
            >
              Create scheme
            </button>
          </Modal.Body>
        </Modal>
      </div>
    </ProjectPageLayout>
  );
};
