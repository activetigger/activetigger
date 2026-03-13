import { FC, useEffect, useState } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import DataGrid, { Column } from 'react-data-grid';
import Select from 'react-select';
import { SendMessage } from '../components/forms/SendMessage';
import { PageLayout } from '../components/layout/PageLayout';
import { ManageMessages } from '../components/ManageMessages';
import {
  useGetLogs,
  useGetMonitoringData,
  useGetMonitoringMetrics,
  useGetServer,
  useGetUserStatistics,
  useRestartQueue,
  useStopProcesses,
  useUsers,
} from '../core/api';
import { useAuth } from '../core/useAuth';

interface Computation {
  unique_id: string;
  user: string;
  time: string;
  kind: string;
}

interface Row {
  time: string;
  user: string;
  action: string;
}

type ModelStats = {
  name: string;
  n: number;
  mean: number;
  std: number;
};

export function ModelStatsTable({ rows }: { rows: ModelStats[] }) {
  return (
    <table style={{ borderCollapse: 'collapse', width: '100%' }}>
      <thead>
        <tr>
          <th>Model</th>
          <th>N</th>
          <th>Mean</th>
          <th>Std</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.name}>
            <td>{row.name}</td>
            <td>{row.n}</td>
            <td>{row.mean.toFixed(2)}</td>
            <td>{row.std.toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

type ApiResponse = Record<
  string,
  {
    n: number;
    mean: number;
    std: number;
  }
>;

function normalizeStats(data: ApiResponse): ModelStats[] {
  return Object.entries(data).map(([name, stats]) => ({
    name,
    ...stats,
  }));
}

type ProcessEvent = {
  start: string;
  end: string;
  duration: number;
  order: number;
};

type ProcessRow = {
  process_name: string;
  kind: string;
  time: string;
  parameters: Record<string, unknown>;
  events: Record<string, ProcessEvent>;
  project_slug: string;
  user_name: string;
  duration: number;
};

type Props = {
  rows: ProcessRow[];
};

export function ProcessTable({ rows }: Props) {
  const displayAllEvent = (events: Record<string, ProcessEvent>) => {
    console.log(events);
    return (
      <ul>
        {Object.entries(events)
          .filter((event) => !['start', 'end'].includes(event[0])) // Is meant to disappear, this is to prevent problems with events created before refactor
          .sort((entryA, entryB) => (entryA[1].order > entryB[1].order ? 1 : 0))
          .map((entry) => {
            if (entry[0] == 'global') {
              const start = new Date(entry[1].start);
              const end = new Date(entry[1].end);
              return (
                <li key={entry[0]}>
                  {start.getHours()}:{start.getMinutes()} — {end.getHours()}:{end.getMinutes()}
                </li>
              );
            } else {
              return (
                <li key={entry[0]}>
                  {entry[0]} : {Math.round(entry[1].duration * 100) / 100} s
                </li>
              );
            }
          })}
      </ul>
    );
  };
  return (
    <table style={{ borderCollapse: 'collapse', width: '100%' }}>
      <thead>
        <tr>
          <th>Process</th>
          <th>Kind</th>
          <th>Project</th>
          <th>User</th>
          <th>Events</th>
          <th>Duration (s)</th>
        </tr>
      </thead>
      <tbody>
        {(rows || []).map((row) => (
          <tr key={row.process_name}>
            <td title={row.process_name}>{row.process_name.slice(0, 8)}…</td>
            <td>{row.kind}</td>
            <td>{row.project_slug}</td>
            <td>{row.user_name}</td>
            <td>{displayAllEvent(row.events)}</td>
            <td>{row.duration.toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/**
 * MonitorPage component displays server monitoring information including logs, resources, active projects, and user statistics.
 */

export const MonitorPage: FC = () => {
  const { authenticatedUser } = useAuth();
  const { activeProjects, gpu, cpu, memory, disk, reFetchQueueState } = useGetServer(null);
  const { restartQueue } = useRestartQueue();
  const { stopProcesses } = useStopProcesses(null);
  const { logs } = useGetLogs('all', 500);
  const [currentUser, setCurrentUser] = useState<string | null>(null);
  const { userStatistics, reFetchStatistics } = useGetUserStatistics(currentUser);
  const { metrics } = useGetMonitoringMetrics();
  const { data } = useGetMonitoringData('all');
  useEffect(() => {
    reFetchStatistics();
  }, [currentUser, reFetchStatistics]);
  const { users } = useUsers();
  const userOptions = users
    ? Object.keys(users).map((userKey) => ({
        value: userKey,
        label: userKey,
      }))
    : [];

  const columns: readonly Column<Row>[] = [
    {
      name: 'Time',
      key: 'time',
      resizable: true,
    },
    {
      name: 'User',
      key: 'user',
      resizable: true,
    },
    {
      name: 'Project',
      key: 'project',
    },
    {
      name: 'Action',
      key: 'action',
    },
  ];

  if (authenticatedUser?.username !== 'root') {
    return (
      <div className="d-flex flex-column align-items-center justify-content-center vh-100 bg-light text-center">
        <div className="p-4 bg-white shadow rounded">
          <h1 className="display-1 fw-bold text-danger mb-3">403</h1>
          <h2 className="h4 mb-3">Access Forbidden</h2>
          <p className="text-muted mb-4">You don’t have permission to access this page.</p>
          <button className="btn btn-primary" onClick={() => window.history.back()}>
            <i className="bi bi-arrow-left me-2"></i> Go Back
          </button>
        </div>
      </div>
    );
  }

  return (
    <PageLayout currentPage="monitor">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <Tabs id="panel2" className="mt-3" defaultActiveKey="active">
              <Tab eventKey="active" title="Active Projects">
                <h2 className="subtitle">Monitor active projects</h2>

                <button className="btn btn-danger m-1" onClick={restartQueue}>
                  Restart memory & queue
                </button>

                {Object.keys(activeProjects || {}).map((project) => (
                  <div key={project}>
                    <div>
                      <table className="table-statistics">
                        <thead>
                          <tr>
                            <th>Project</th>
                            <th colSpan={3} className="table-primary text-primary text-center">
                              {project}
                            </th>
                          </tr>
                          <tr>
                            <th>User</th>
                            <th>Time</th>
                            <th>Kind</th>
                            <th>Kill process</th>
                          </tr>
                        </thead>
                        <tbody>
                          {activeProjects &&
                            Object.values(activeProjects[project] as unknown as Computation[]).map(
                              (e) => (
                                <tr key={e.unique_id}>
                                  <td>{e.user}</td>
                                  <td>{e.time}</td>
                                  <td>{e.kind}</td>
                                  <td>
                                    <button
                                      onClick={() => {
                                        stopProcesses('all', e.unique_id);
                                        reFetchQueueState();
                                      }}
                                      className="btn btn-danger"
                                    >
                                      kill
                                    </button>
                                  </td>
                                </tr>
                              ),
                            )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ))}
              </Tab>
              <Tab eventKey="statistics" title="Statistics">
                {<ModelStatsTable rows={normalizeStats(metrics || {})} />}
                {<ProcessTable rows={data as unknown as ProcessRow[]} />}
              </Tab>
              <Tab eventKey="messages" title="Messages">
                <div className="col-md-6">
                  <h3 className="subtitle">Send message</h3>
                  <SendMessage />
                </div>
                <div className="col-md-8 mt-3">
                  <h3 className="subtitle">Manage messages</h3>
                  <ManageMessages />
                </div>
              </Tab>
              <Tab eventKey="logs" title="Logs">
                <h2 className="subtitle">Recent activity on all projects</h2>
                {logs ? (
                  <DataGrid<Row>
                    className="fill-grid rdg-light mt-2"
                    columns={columns}
                    rows={(logs as unknown as Row[]) || []}
                  />
                ) : (
                  <div>No rights</div>
                )}
              </Tab>
              <Tab eventKey="ressources" title="Ressouces">
                <h2 className="subtitle">Monitor ressources</h2>
                <table className="table-statistics">
                  <thead>
                    <tr>
                      <th colSpan={2} className="table-primary text-primary text-center">
                        Type
                      </th>
                      <th className="table-primary text-primary text-center">Ressources</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td colSpan={2}>GPU</td>
                      <td>{JSON.stringify(gpu)}</td>
                    </tr>
                    <tr>
                      <td colSpan={2}>CPU</td>
                      <td>{JSON.stringify(cpu)}</td>
                    </tr>
                    <tr>
                      <td colSpan={2}>Memory</td>
                      <td>{JSON.stringify(memory)}</td>
                    </tr>
                    <tr>
                      <td colSpan={2}>Disk</td>
                      <td>{JSON.stringify(disk)}</td>
                    </tr>
                  </tbody>
                </table>
                <hr />
              </Tab>

              <Tab eventKey="users" title="Users">
                <h2 className="subtitle">Monitor users</h2>
                <Select
                  id="select-user"
                  className="form-select"
                  options={userOptions}
                  onChange={(selectedOption) => {
                    setCurrentUser(selectedOption ? selectedOption.value : null);
                  }}
                  isClearable
                  placeholder="Select a user"
                />
                <table className="table-statistics">
                  <thead>
                    <tr>
                      <th colSpan={2} className="table-primary text-primary text-center">
                        User
                      </th>
                      <th className="table-primary text-primary text-center">Statistics</th>
                    </tr>
                  </thead>
                  <tbody>
                    {userStatistics &&
                      Object.entries(userStatistics).map(([key, value]) => (
                        <tr key={key}>
                          <td colSpan={2}>{key}</td>
                          <td>{JSON.stringify(value)}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </Tab>
            </Tabs>
          </div>
        </div>
      </div>{' '}
    </PageLayout>
  );
};
