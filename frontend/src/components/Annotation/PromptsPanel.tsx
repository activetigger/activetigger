import { FC, useState } from 'react';
import { LuRefreshCw, LuTrash2 } from 'react-icons/lu';

import { useAddImagePrompt, useDeleteImagePrompt, useListImagePrompts } from '../../core/api';
import { PromptsProjectStateModel } from '../../types';

interface PromptsPanelProps {
  projectSlug: string;
  state?: PromptsProjectStateModel | null;
}

/**
 * Multimodal prompt management — contents of the prompts modal.
 * Create / list / delete prompts used with the "prompt" selection mode
 * to rank images by cosine similarity against the prompt embedding.
 * See docs/multimodal-prompt-selection.md.
 */
export const PromptsPanel: FC<PromptsPanelProps> = ({ projectSlug, state }) => {
  const addPrompt = useAddImagePrompt(projectSlug);
  const deletePrompt = useDeleteImagePrompt(projectSlug);
  const { prompts, reFetchImagePrompts } = useListImagePrompts(projectSlug);

  const [text, setText] = useState<string>('');
  const [featureName, setFeatureName] = useState<string>('');

  const bindable = state?.bindable_features ?? [];
  const training = Object.values(state?.training ?? {});

  // default to the first bindable feature as soon as one appears
  if (!featureName && bindable.length > 0) {
    setFeatureName(bindable[0]);
  }

  const onSubmit = async () => {
    if (!text.trim() || !featureName) return;
    const res = await addPrompt(text.trim(), featureName);
    if (res) {
      setText('');
      // the embedding happens on the GPU worker — refetch will pick it up
      // once update_processes persists the row.
      setTimeout(() => reFetchImagePrompts(), 1500);
    }
  };

  return (
    <div>
      {bindable.length === 0 ? (
        <div className="alert alert-warning mb-0" role="alert">
          No multimodal-embeddings feature yet. Compute one from the Features page first.
        </div>
      ) : (
        <div className="mb-3">
          <label htmlFor="prompt_text" className="form-label small-gray">
            New prompt
          </label>
          <textarea
            id="prompt_text"
            className="form-control mb-2"
            rows={2}
            placeholder='e.g. "a red car parked in front of a house"'
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div className="d-flex gap-2 align-items-center">
            {bindable.length > 1 && (
              <select
                className="form-select form-select-sm"
                style={{ maxWidth: '260px' }}
                value={featureName}
                onChange={(e) => setFeatureName(e.target.value)}
              >
                {bindable.map((f) => (
                  <option key={f} value={f}>
                    {f}
                  </option>
                ))}
              </select>
            )}
            <button className="btn btn-primary btn-sm" onClick={onSubmit} disabled={!text.trim()}>
              Save prompt
            </button>
          </div>
        </div>
      )}

      {training.length > 0 && (
        <div className="alert alert-info py-2 mb-3">
          <strong>Encoding…</strong>
          <ul className="mb-0 mt-1">
            {training.map((p, i) => (
              <li key={i}>
                <em>{p.text}</em>
                {p.progress ? ` (${p.progress}%)` : ''}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="d-flex justify-content-between align-items-center mb-1">
        <span className="small-gray">Saved prompts ({prompts.length})</span>
        <button
          type="button"
          className="btn btn-sm btn-light"
          title="Refresh"
          onClick={() => reFetchImagePrompts()}
        >
          <LuRefreshCw size={14} />
        </button>
      </div>

      {prompts.length === 0 ? (
        <p className="text-muted small mb-0">No prompts saved yet.</p>
      ) : (
        <div className="table-responsive">
          <table className="table table-sm table-striped table-hover align-middle mb-0">
            <thead>
              <tr>
                <th>Prompt</th>
                <th style={{ width: '30%' }}>Feature</th>
                <th style={{ width: '15%' }}>User</th>
                <th style={{ width: '56px' }}></th>
              </tr>
            </thead>
            <tbody>
              {prompts.map((p) => (
                <tr key={p.prompt_id}>
                  <td>{p.text}</td>
                  <td className="text-muted small">{p.feature_name}</td>
                  <td className="text-muted small">{p.user}</td>
                  <td className="text-end">
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-danger"
                      title="Delete prompt"
                      onClick={async () => {
                        const ok = await deletePrompt(p.prompt_id);
                        if (ok) reFetchImagePrompts();
                      }}
                    >
                      <LuTrash2 size={14} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
