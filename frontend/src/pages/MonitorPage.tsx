import { FC, useEffect, useState } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import DataGrid, { Column } from 'react-data-grid';
import Select from 'react-select';
import { VictoryAxis, VictoryBar, VictoryChart, VictoryTheme, VictoryTooltip } from 'victory';
import { SendMessage } from '../components/forms/SendMessage';
import { PageLayout } from '../components/layout/PageLayout';
import { ManageMessages } from '../components/ManageMessages';
import {
  useAddSelfAsManager,
  useGetAllProjects,
  useGetLogs,
  useGetMonitoringActivity,
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

type ActivityPoint = {
  hour: string;
  annotations: number;
  active_users: number;
};

function ActivityTimeline({ points }: { points: ActivityPoint[] }) {
  if (!points || points.length === 0) {
    return <div className="alert alert-info m-3">No activity in the selected period.</div>;
  }

  const data = points.map((p, i) => ({
    i,
    date: new Date(p.hour),
    annotations: p.annotations,
    active_users: p.active_users,
  }));

  // One tick per day (every 24 hours), labelled with the day boundary
  const dayTickValues = data.filter((d) => d.date.getUTCHours() === 0).map((d) => d.i);
  const dayTickFormat = (i: number) => {
    const d = data[i]?.date;
    return d ? `${d.getUTCMonth() + 1}/${d.getUTCDate()}` : '';
  };

  const annotationsTotal = data.reduce((s, d) => s + d.annotations, 0);
  const activeUsersPeak = data.reduce((m, d) => Math.max(m, d.active_users), 0);

  return (
    <div>
      <div className="mb-2">
        <span className="badge bg-primary me-2">Annotations (7d): {annotationsTotal}</span>
        <span className="badge bg-success">Peak distinct users / hour: {activeUsersPeak}</span>
      </div>

      <div>
        <h3 className="subtitle mt-1 mb-0">Annotations per hour</h3>
        <VictoryChart
          theme={VictoryTheme.material}
          domainPadding={{ x: 5 }}
          width={1000}
          height={170}
          padding={{ top: 10, bottom: 35, left: 55, right: 15 }}
        >
          <VictoryAxis
            tickValues={dayTickValues}
            tickFormat={dayTickFormat}
            style={{ tickLabels: { fontSize: 10 } }}
          />
          <VictoryAxis
            dependentAxis
            label="Annotations"
            style={{ axisLabel: { padding: 40, fontSize: 12 }, tickLabels: { fontSize: 10 } }}
          />
          <VictoryBar
            data={data}
            x="i"
            y="annotations"
            style={{ data: { fill: '#0072B2' } }}
            labels={({ datum }) =>
              `${datum.date.toISOString().slice(0, 13)}:00\nAnnotations: ${datum.annotations}`
            }
            labelComponent={<VictoryTooltip />}
          />
        </VictoryChart>
      </div>

      <div>
        <h3 className="subtitle mt-1 mb-0">Active users per hour</h3>
        <VictoryChart
          theme={VictoryTheme.material}
          domainPadding={{ x: 5 }}
          width={1000}
          height={170}
          padding={{ top: 10, bottom: 35, left: 55, right: 15 }}
        >
          <VictoryAxis
            tickValues={dayTickValues}
            tickFormat={dayTickFormat}
            style={{ tickLabels: { fontSize: 10 } }}
          />
          <VictoryAxis
            dependentAxis
            label="Active users"
            style={{ axisLabel: { padding: 40, fontSize: 12 }, tickLabels: { fontSize: 10 } }}
          />
          <VictoryBar
            data={data}
            x="i"
            y="active_users"
            style={{ data: { fill: '#D55E00' } }}
            labels={({ datum }) =>
              `${datum.date.toISOString().slice(0, 13)}:00\nUsers: ${datum.active_users}`
            }
            labelComponent={<VictoryTooltip />}
          />
        </VictoryChart>
      </div>
    </div>
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
  const { allProjects, reFetchAllProjects } = useGetAllProjects();
  const { activity } = useGetMonitoringActivity(7);
  const { addSelfAsManager } = useAddSelfAsManager(reFetchAllProjects);
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
            <Tabs id="panel2" className="mt-3" defaultActiveKey="activity">
              <Tab eventKey="activity" title="Activity">
                <h2 className="subtitle">Instance activity — last 7 days (hourly)</h2>
                <ActivityTimeline points={(activity?.activity as ActivityPoint[]) || []} />
              </Tab>
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
              <Tab eventKey="projects" title="All projects">
                <h2 className="subtitle">All existing projects</h2>
                <table className="table-statistics">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Type</th>
                      <th>Slug</th>
                      <th>Creator</th>
                      <th>Created</th>
                      <th>Size (MB)</th>
                      <th>Last activity</th>
                      <th>My rights</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(allProjects || []).map((p) => (
                      <tr key={p.project_slug}>
                        <td>{p.parameters?.project_name}</td>
                        <td>{p.parameters?.kind ?? 'text'}</td>
                        <td>{p.project_slug}</td>
                        <td>{p.created_by}</td>
                        <td>{p.created_at}</td>
                        <td>{p.size?.toFixed(1) ?? '-'}</td>
                        <td>{p.last_activity ?? '-'}</td>
                        <td>{p.user_right}</td>
                        <td>
                          {p.user_right === 'none' ? (
                            <button
                              className="btn btn-sm btn-primary"
                              onClick={() => addSelfAsManager(p.project_slug)}
                            >
                              Add me as manager
                            </button>
                          ) : (
                            <span className="text-muted">—</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
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
              <Tab eventKey="ressources" title="Resources">
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
