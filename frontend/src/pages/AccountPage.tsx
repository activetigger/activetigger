import { FC, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { ChangeEmail } from '../components/forms/ChangeEmail';
import { ChangePassword } from '../components/forms/ChangePassword';
import { PageLayout } from '../components/layout/PageLayout';
import { useCurrentUser } from '../core/api';
import { useAuth } from '../core/useAuth';

export const AccountPage: FC = () => {
  const { authenticatedUser } = useAuth();
  const [refreshKey, setRefreshKey] = useState(0);
  const { currentUser } = useCurrentUser(refreshKey);

  const [showPasswordModal, setShowPasswordModal] = useState(false);
  const [showEmailModal, setShowEmailModal] = useState(false);

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
