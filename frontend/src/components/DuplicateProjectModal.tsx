import { FC, useEffect, useState } from 'react';
import { Button, Modal, Spinner } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { getProjectStatus, useDuplicateProject } from '../core/api';
import { useNotifications } from '../core/notifications';
import { AvailableProjectsModel } from '../types';

interface DuplicateProjectModalProps {
  show: boolean;
  onHide: () => void;
  projects: AvailableProjectsModel[];
}

const POLL_INTERVAL_MS = 1500;

export const DuplicateProjectModal: FC<DuplicateProjectModalProps> = ({
  show,
  onHide,
  projects,
}) => {
  const duplicateProject = useDuplicateProject();
  const { notify } = useNotifications();
  const navigate = useNavigate();
  const [selectedSlug, setSelectedSlug] = useState<string>('');
  const [busy, setBusy] = useState<boolean>(false);
  const [status, setStatus] = useState<string>('');

  useEffect(() => {
    if (show) {
      setSelectedSlug(projects[0]?.parameters.project_slug ?? '');
      setBusy(false);
      setStatus('');
    }
  }, [show, projects]);

  const handleConfirm = async () => {
    if (!selectedSlug) return;
    setBusy(true);
    setStatus('Starting duplication…');
    let targetSlug: string;
    try {
      targetSlug = await duplicateProject(selectedSlug);
    } catch {
      setBusy(false);
      setStatus('');
      return;
    }

    setStatus(`Copying files into ${targetSlug}…`);
    // poll the backend until the new project finishes loading or errors out
    const poll = async (): Promise<void> => {
      const s = await getProjectStatus(targetSlug);
      if (s === 'existing') {
        notify({ type: 'success', message: 'Project duplicated.' });
        onHide();
        navigate(0);
        return;
      }
      if (typeof s === 'string' && s.startsWith('error')) {
        notify({ type: 'error', message: s });
        setBusy(false);
        setStatus('');
        return;
      }
      setStatus(s === 'duplicating' ? `Copying files into ${targetSlug}…` : `Status: ${s}`);
      setTimeout(poll, POLL_INTERVAL_MS);
    };
    poll();
  };

  return (
    <Modal show={show} onHide={busy ? undefined : onHide}>
      <Modal.Header closeButton={!busy}>
        <Modal.Title>Duplicate a project</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <p className="text-muted small mb-2">
          A copy of the selected project will be created with the handle{' '}
          <code>{selectedSlug ? `${selectedSlug}-copy` : '<slug>-copy'}</code>. Files and database
          entries (schemes, annotations, features, models, generations) are duplicated. You become
          the manager of the new project.
          <br />
          <br />
          <strong>
            If your project is large, it will double the storage usage. Please avoid too many
            duplications.
          </strong>
          <br />
        </p>
        <label htmlFor="duplicate-project-select" className="form-label">
          Project to duplicate
        </label>
        <select
          id="duplicate-project-select"
          className="form-select"
          value={selectedSlug}
          onChange={(e) => setSelectedSlug(e.target.value)}
          disabled={busy}
        >
          {projects.map((p) => (
            <option key={p.parameters.project_slug} value={p.parameters.project_slug}>
              {p.parameters.project_name} ({p.parameters.project_slug})
            </option>
          ))}
        </select>
        {busy && status && (
          <div className="mt-3 d-flex align-items-center text-muted small">
            <Spinner size="sm" animation="border" className="me-2" />
            {status}
          </div>
        )}
      </Modal.Body>
      <Modal.Footer>
        <Button variant="primary" onClick={handleConfirm} disabled={!selectedSlug || busy}>
          {busy ? (
            <>
              <Spinner size="sm" animation="border" className="me-2" />
              Duplicating…
            </>
          ) : (
            'Duplicate'
          )}
        </Button>
        <Button variant="secondary" onClick={onHide} disabled={busy}>
          Cancel
        </Button>
      </Modal.Footer>
    </Modal>
  );
};
