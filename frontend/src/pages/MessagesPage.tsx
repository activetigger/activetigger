import { marked } from 'marked';
import { FC, useMemo, useState } from 'react';
import { FaRegTrashAlt } from 'react-icons/fa';
import { HiOutlineInbox } from 'react-icons/hi';
import { PageLayout } from '../components/layout/PageLayout';
import { useDeleteMessage, useGetInbox, useSendMessage, useUserProjects } from '../core/api';
import { useAuth } from '../core/useAuth';

type Target = 'user' | 'project' | 'all';

const MessageBubble: FC<{
  id: number;
  content: string;
  time?: string;
  createdBy: string;
  forProject?: string | null;
  onDelete: (id: number) => void;
}> = ({ id, content, time, createdBy, forProject, onDelete }) => {
  const html = useMemo(() => {
    marked.setOptions({ breaks: true });
    return marked.parse(content) as string;
  }, [content]);

  return (
    <div
      style={{
        background: '#f8f9fa',
        border: '1px solid #e9ecef',
        borderLeft: forProject ? '4px solid #6f9bd1' : '4px solid #ff9a3c',
        borderRadius: '0.5rem',
        padding: '0.75rem 1rem',
        marginBottom: '0.75rem',
        textAlign: 'left',
      }}
    >
      <div className="d-flex justify-content-between align-items-start gap-2">
        <div style={{ fontSize: '0.8rem', color: '#666' }}>
          From <span className="fw-bold">{createdBy}</span>
          {forProject && (
            <>
              {' '}
              ·{' '}
              <span className="badge text-bg-light" style={{ border: '1px solid #dee2e6' }}>
                project: {forProject}
              </span>
            </>
          )}
        </div>
        <button
          type="button"
          className="btn btn-link p-0 text-muted"
          title="Delete this message"
          onClick={() => onDelete(id)}
        >
          <FaRegTrashAlt />
        </button>
      </div>
      <div
        className="message-content mt-1"
        style={{ fontSize: '0.9rem', color: '#333', lineHeight: '1.5' }}
        dangerouslySetInnerHTML={{ __html: html }}
      />
      {time && (
        <div style={{ fontSize: '0.75rem', color: '#999', marginTop: '0.25rem' }}>
          {new Date(time).toLocaleString()}
        </div>
      )}
    </div>
  );
};

export const MessagesPage: FC = () => {
  const { inbox, reFetchInbox } = useGetInbox();
  const { sendMessage } = useSendMessage();
  const { deleteMessage } = useDeleteMessage();
  const { projects } = useUserProjects();
  const { authenticatedUser } = useAuth();
  const isRoot = authenticatedUser?.username === 'root';

  const [target, setTarget] = useState<Target>('user');
  const [forUser, setForUser] = useState<string>('');
  const [forProject, setForProject] = useState<string>('');
  const [content, setContent] = useState<string>('');
  const [submitting, setSubmitting] = useState<boolean>(false);

  const recipient = forUser.trim();

  const send = async () => {
    if (!content.trim()) return;
    if (target === 'user' && !recipient) return;
    if (target === 'project' && !forProject) return;
    setSubmitting(true);
    const opts =
      target === 'user'
        ? { for_user: recipient }
        : target === 'project'
          ? { for_project: forProject }
          : undefined;
    const ok = await sendMessage(content, target, opts);
    setSubmitting(false);
    if (ok) {
      setContent('');
      reFetchInbox();
    }
  };

  const onDelete = async (id: number) => {
    await deleteMessage(id);
    reFetchInbox();
  };

  const messages = inbox || [];

  return (
    <PageLayout currentPage="messages">
      <div className="container-fluid">
        <div className="row">
          <div className="col-0 col-lg-3" />
          <div className="col-12 col-lg-6">
            <div className="d-flex align-items-center gap-2 mt-3">
              <HiOutlineInbox size={24} />
              <h2 className="m-0">Messages</h2>
            </div>

            <div className="card mt-3">
              <div className="card-body">
                <h6 className="card-title">Send a new message</h6>

                <div className="d-flex flex-wrap gap-3 mb-2">
                  <label className="form-check-label">
                    <input
                      type="radio"
                      className="form-check-input me-1"
                      checked={target === 'user'}
                      onChange={() => setTarget('user')}
                    />
                    To a user
                  </label>
                  <label className="form-check-label">
                    <input
                      type="radio"
                      className="form-check-input me-1"
                      checked={target === 'project'}
                      onChange={() => setTarget('project')}
                    />
                    To all members of a project
                  </label>
                  {isRoot && (
                    <label className="form-check-label">
                      <input
                        type="radio"
                        className="form-check-input me-1"
                        checked={target === 'all'}
                        onChange={() => setTarget('all')}
                      />
                      To all users of the instance
                    </label>
                  )}
                </div>

                {target === 'all' && (
                  <div className="alert alert-warning py-2 mb-2" style={{ fontSize: '0.85rem' }}>
                    This message will be delivered to every active user's inbox.
                  </div>
                )}

                {target === 'user' && (
                  <input
                    type="text"
                    className="form-control mb-2"
                    placeholder="Recipient username"
                    value={forUser}
                    onChange={(e) => setForUser(e.target.value)}
                    autoComplete="off"
                  />
                )}
                {target === 'project' && (
                  <select
                    className="form-select mb-2"
                    value={forProject}
                    onChange={(e) => setForProject(e.target.value)}
                  >
                    <option value="">Select a project…</option>
                    {(projects || []).map((p) => (
                      <option key={p.parameters.project_slug} value={p.parameters.project_slug}>
                        {p.parameters.project_name}
                      </option>
                    ))}
                  </select>
                )}

                <textarea
                  className="form-control mb-2"
                  placeholder="Message (markdown supported)"
                  rows={3}
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                />

                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={
                    submitting ||
                    !content.trim() ||
                    (target === 'user' && !recipient) ||
                    (target === 'project' && !forProject)
                  }
                  onClick={send}
                >
                  Send
                </button>
              </div>
            </div>

            <div className="mt-4">
              <h6>Your inbox</h6>
              {messages.length === 0 && <div className="text-muted">No messages.</div>}
              {messages.map((m) => (
                <MessageBubble
                  key={m.id}
                  id={m.id}
                  content={m.content}
                  time={m.time}
                  createdBy={m.created_by}
                  forProject={m.for_project}
                  onDelete={onDelete}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </PageLayout>
  );
};
