import { FC, useEffect, useState } from 'react';

import { marked } from 'marked';
import {
  useDeleteMessage,
  useGetCodebookMessages,
  useGetSchemeCodebook,
  usePostSchemeCodebook,
} from '../core/api';

import { CodebookManagement } from '../components/CodeBookManagement';

import MDEditor from '@uiw/react-md-editor';
import { Modal } from 'react-bootstrap';
import { FaBookOpen, FaCloudDownloadAlt, FaRegTrashAlt } from 'react-icons/fa';
import { MdDriveFileRenameOutline } from 'react-icons/md';

interface CodebookDisplayProps {
  projectSlug: string | null;
  currentScheme: string | null;
  canEdit?: boolean;
}

export const CodebookDisplay: FC<CodebookDisplayProps> = ({
  projectSlug,
  currentScheme,
  canEdit,
}) => {
  // get codebook
  const { codebook, time, reFetchCodebook } = useGetSchemeCodebook(
    projectSlug || null,
    currentScheme || null,
  );

  // project messages addressed to the current user
  const { codebookMessages, reFetchCodebookMessages } = useGetCodebookMessages(projectSlug || null);
  const { deleteMessage } = useDeleteMessage();
  const onDeleteMessage = async (id: number) => {
    await deleteMessage(id);
    reFetchCodebookMessages();
  };
  // hooks and states to modify the codebook
  const { postCodebook } = usePostSchemeCodebook(projectSlug || null, currentScheme || null);
  const [modifiedCodebook, setModifiedCodebook] = useState<string | undefined>(undefined);

  // reset modified codebook when scheme changes
  useEffect(() => {
    setModifiedCodebook(undefined);
  }, [currentScheme]);

  const saveCodebook = async () => {
    await postCodebook(modifiedCodebook || '', time || '');
    setModifiedCodebook(undefined);
    reFetchCodebook();
  };

  // open codebook edition
  const [showCodebookModal, setShowCodebookModal] = useState(false);
  const openAsHTML = () => {
    const htmlContent = `
    <html>
      <head>
        <title>Codebook</title>
        <meta charset="UTF-8" />
        <style>
          body { font-family: sans-serif; padding: 2em; }
        </style>
      </head>
      <body>
        <div>${marked.parse(codebook || '')}</div>
      </body>
    </html>
  `;
    const blob = new Blob([htmlContent], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    window.open(url, '_blank');
  };

  const downloadMarkdown = () => {
    const blob = new Blob([codebook || ''], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'codebook.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <div id="codebook">
        {/* Header fin et discret */}
        <div id="header">
          {' '}
          <span style={{ fontWeight: 'bold' }}>📘 Guidelines</span>
          {canEdit && (
            <div id="edit-buttons-group" role="group">
              <button
                onClick={() => {
                  setModifiedCodebook(undefined);
                  setShowCodebookModal(true);
                }}
                title="Edit codebook"
                className="btn btn-link p-0"
              >
                <MdDriveFileRenameOutline size={20} />
              </button>
              <button onClick={openAsHTML} title="Open" className="btn btn-link p-0">
                <FaBookOpen size={20} />
              </button>
              <button onClick={downloadMarkdown} title="Download" className="btn btn-link p-0">
                <FaCloudDownloadAlt size={20} />
              </button>
            </div>
          )}
        </div>

        {(codebookMessages || []).length > 0 && (
          <div id="project-messages" style={{ margin: '0.5rem 0 1rem 0' }}>
            {(codebookMessages || []).map((m) => (
              <div
                key={m.id}
                style={{
                  background: '#f8f9fa',
                  border: '1px solid #e9ecef',
                  borderLeft: '4px solid #6f9bd1',
                  borderRadius: '0.5rem',
                  padding: '0.5rem 0.75rem',
                  marginBottom: '0.5rem',
                }}
              >
                <div className="d-flex justify-content-between align-items-start gap-2">
                  <div style={{ fontSize: '0.75rem', color: '#666' }}>
                    From <span className="fw-bold">{m.created_by}</span>
                    {m.time && (
                      <>
                        {' · '}
                        <span style={{ color: '#999' }}>{new Date(m.time).toLocaleString()}</span>
                      </>
                    )}
                  </div>
                  <button
                    type="button"
                    className="btn btn-link p-0 text-muted"
                    title="Delete this message"
                    onClick={() => onDeleteMessage(m.id)}
                  >
                    <FaRegTrashAlt />
                  </button>
                </div>
                <div
                  style={{ fontSize: '0.9rem', color: '#333', lineHeight: '1.5' }}
                  dangerouslySetInnerHTML={{ __html: marked.parse(m.content) as string }}
                />
              </div>
            ))}
          </div>
        )}

        {/* Corps du codebook avec scroll */}
        <div id="content" data-color-mode="light">
          <MDEditor.Markdown
            source={codebook}
            style={{
              backgroundColor: 'transparent',
              fontSize: '0.95rem',
              lineHeight: '1.6',
              maxWidth: '100%',
            }}
          />
        </div>
      </div>

      <Modal
        show={showCodebookModal}
        onHide={() => {
          setShowCodebookModal(false);
          saveCodebook();
        }}
        id="codebook-modal"
        size="xl"
      >
        <Modal.Header closeButton>
          <Modal.Title>Edit your current codebook</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <CodebookManagement
            codebook={codebook}
            time={time}
            modifiedCodebook={modifiedCodebook}
            setModifiedCodebook={setModifiedCodebook}
            saveCodebook={saveCodebook}
            callbackOnClose={setShowCodebookModal}
          />
        </Modal.Body>
      </Modal>
    </>
  );
};
