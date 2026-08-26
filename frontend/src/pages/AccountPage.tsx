import { FC, useEffect, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { AddCredentials } from '../components/forms/AddCredentials';
import { ChangeEmail } from '../components/forms/ChangeEmail';
import { ChangePassword } from '../components/forms/ChangePassword';
import { PageLayout } from '../components/layout/PageLayout';
import { UserActivityChart } from '../components/UserActivityChart';
import {
  useCurrentUser,
  useDeleteUserCredentials,
  useGetUserStatistics,
  useUserCredentials,
} from '../core/api';
import { useAuth } from '../core/useAuth';

const formatDuration = (seconds: number): string => {
  if (!seconds || seconds <= 0) return '0s';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.round(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
};

export const AccountPage: FC = () => {
  const { authenticatedUser } = useAuth();
  const [refreshKey, setRefreshKey] = useState(0);
  const { currentUser } = useCurrentUser(refreshKey);

  const [credentialsRefreshKey, setCredentialsRefreshKey] = useState(0);
  const { userCredentials } = useUserCredentials(credentialsRefreshKey);
  const { deleteUserCredentials } = useDeleteUserCredentials();

  const [showPasswordModal, setShowPasswordModal] = useState(false);
  const [showEmailModal, setShowEmailModal] = useState(false);
  const [showCredentialsModal, setShowCredentialsModal] = useState(false);

  const handleDeleteCredentials = async (name: string) => {
    const ok = await deleteUserCredentials(name);
    if (ok) setCredentialsRefreshKey((k) => k + 1);
  };

  const username = currentUser?.username ?? authenticatedUser?.username;
  const status = currentUser?.status ?? authenticatedUser?.status;
  const contact = currentUser?.contact ?? '';

  const { userStatistics, reFetchStatistics } = useGetUserStatistics(username ?? null);
  useEffect(() => {
    reFetchStatistics();
  }, [username, reFetchStatistics]);

  const activity = userStatistics?.annotation_activity || [];
  const annotationsLast7Days = activity.reduce((sum, point) => sum + point.annotations, 0);

  return (
    <PageLayout currentPage="account">
      <div className="container">
        {username && (
          <div className="row">
            <div className="col-0 col-sm-2 col-md-3" />
            <div className="col-12 col-sm-8 col-md-6">
              <h3 className="mt-3 mb-3">Account</h3>

              <div className="card mb-3">
                <div className="card-body">
                  <div className="mb-2">
                    <strong>Username:</strong> {username}
                  </div>
                  <div className="mb-2">
                    <strong>Status:</strong> {status}
                  </div>
                  <div className="mb-2">
                    <strong>Email:</strong>{' '}
                    {contact ? contact : <em className="text-muted">not set</em>}
                  </div>
                </div>
              </div>

              <div className="d-flex gap-2 flex-wrap">
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => setShowPasswordModal(true)}
                >
                  Change password
                </button>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => setShowEmailModal(true)}
                >
                  Change email
                </button>
              </div>

              <h4 className="mt-4 mb-2">API credentials</h4>
              <div className="card mb-3">
                <div className="card-body">
                  <p className="text-muted small mb-2">
                    Saved endpoint/key pairs for generative APIs. Keys are stored encrypted and can
                    be reused when configuring a generative model, but never displayed again.
                  </p>
                  {(userCredentials || []).length === 0 ? (
                    <em className="text-muted">No saved credentials</em>
                  ) : (
                    <table className="table table-sm align-middle mb-2">
                      <thead>
                        <tr>
                          <th>Name</th>
                          <th>API</th>
                          <th>Endpoint</th>
                          <th></th>
                        </tr>
                      </thead>
                      <tbody>
                        {(userCredentials || []).map((credential) => (
                          <tr key={credential.name}>
                            <td>{credential.name}</td>
                            <td>{credential.api}</td>
                            <td>{credential.endpoint || <em className="text-muted">none</em>}</td>
                            <td className="text-end">
                              <button
                                type="button"
                                className="btn btn-outline-danger btn-sm"
                                onClick={() => handleDeleteCredentials(credential.name)}
                              >
                                Delete
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={() => setShowCredentialsModal(true)}
                  >
                    Add credentials
                  </button>
                </div>
              </div>

              <h4 className="mt-4 mb-2">Statistics</h4>
              <div className="card mb-3">
                <div className="card-body">
                  {userStatistics ? (
                    <>
                      <div className="mb-2">
                        <strong>Projects:</strong> {Object.keys(userStatistics.projects).length}
                      </div>
                      <div className="mb-2">
                        <strong>Total annotations:</strong> {userStatistics.total_annotations}
                      </div>
                      <div className="mb-2">
                        <strong>Annotations (last 7 days):</strong> {annotationsLast7Days}
                      </div>
                      <div className="mb-2">
                        <strong>Median time per annotation:</strong>{' '}
                        {userStatistics.median_annotation_time_seconds != null ? (
                          `${userStatistics.median_annotation_time_seconds.toFixed(1)} s`
                        ) : (
                          <em className="text-muted">not enough data</em>
                        )}
                      </div>
                      <div className="mb-2">
                        <strong>GPU time:</strong> {formatDuration(userStatistics.gpu_time_seconds)}
                      </div>
                      <div className="mb-2">
                        <strong>Compute time:</strong>{' '}
                        {formatDuration(userStatistics.compute_time_seconds)}
                      </div>
                    </>
                  ) : (
                    <em className="text-muted">Loading statistics…</em>
                  )}
                </div>
              </div>
            </div>
            <div className="col-0 col-sm-2 col-md-3" />
          </div>
        )}
        {username && userStatistics && (
          <div className="row">
            <div className="col-12 col-lg-10 mx-auto">
              <h4 className="mt-2 mb-0">Recent activity — last 7 days (hourly)</h4>
              <UserActivityChart points={activity} />
            </div>
          </div>
        )}
      </div>

      <Modal show={showPasswordModal} onHide={() => setShowPasswordModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Change password</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <ChangePassword onSuccess={() => setShowPasswordModal(false)} />
        </Modal.Body>
      </Modal>

      <Modal show={showCredentialsModal} onHide={() => setShowCredentialsModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Add API credentials</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <AddCredentials
            onSuccess={() => {
              setShowCredentialsModal(false);
              setCredentialsRefreshKey((k) => k + 1);
            }}
          />
        </Modal.Body>
      </Modal>

      <Modal show={showEmailModal} onHide={() => setShowEmailModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Change email</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <ChangeEmail
            currentEmail={contact}
            onSuccess={() => {
              setShowEmailModal(false);
              setRefreshKey((k) => k + 1);
            }}
          />
        </Modal.Body>
      </Modal>
    </PageLayout>
  );
};
