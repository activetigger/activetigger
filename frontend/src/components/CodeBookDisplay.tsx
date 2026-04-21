import { FC, useEffect, useState } from 'react';

import { marked } from 'marked';
import { useGetSchemeCodebook, usePostSchemeCodebook } from '../core/api';

import { CodebookManagement } from '../components/CodeBookManagement';

import MDEditor from '@uiw/react-md-editor';
import { Modal } from 'react-bootstrap';
import { FaBookOpen, FaCloudDownloadAlt } from 'react-icons/fa';
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
