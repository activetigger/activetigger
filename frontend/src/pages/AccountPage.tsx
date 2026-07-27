import { FC, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { AddCredentials } from '../components/forms/AddCredentials';
import { ChangeEmail } from '../components/forms/ChangeEmail';
import { ChangePassword } from '../components/forms/ChangePassword';
import { PageLayout } from '../components/layout/PageLayout';
import { useCurrentUser, useDeleteUserCredentials, useUserCredentials } from '../core/api';
import { useAuth } from '../core/useAuth';

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
            </div>
            <div className="col-0 col-sm-2 col-md-3" />
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
