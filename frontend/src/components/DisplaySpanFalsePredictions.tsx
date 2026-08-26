import chroma from 'chroma-js';
import { FC, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { AnnotateBlendTag, TextAnnotateBlend } from 'react-text-annotate-blend';

type Span = { start: number; end: number; tag: string };
export type SpanFalsePredictionDoc = {
  id: string;
  text: string;
  gold_spans: Span[];
  pred_spans: Span[];
  // disagreements is still populated by the backend (kind classification),
  // but intentionally not surfaced here — the user wants a clean inline
  // view, no per-document error labels.
};

interface Props {
  falsePredictions: SpanFalsePredictionDoc[];
  // Stable tag palette source. Falls back to the union of observed tags.
  labels?: string[];
  // Required to build the annotation-panel deep-link.
  projectSlug?: string | null;
  // "train" / "valid" / "test" — passed through as ?dataset=... so the
  // annotation page lands on the right split.
  dataset?: string;
}

const PAGE_SIZE = 25;

/**
 * Inline-highlighted wrong-prediction viewer for span schemes.
 *
 * Each document renders the text twice: once with gold spans (Gold row),
 * once with predicted spans (Predicted row). Colors are stable per tag
 * across both rows, so a wrong-tag disagreement reads as the same
 * character range painted differently in the two rows.
 *
 * The id is rendered as a link to the annotation panel for that element,
 * mirroring the classification false-predictions table.
 */
export const DisplaySpanFalsePredictions: FC<Props> = ({
  falsePredictions,
  labels,
  projectSlug,
  dataset,
}) => {
  const [page, setPage] = useState(0);

  const colorMap = useMemo<Record<string, string>>(() => {
    let tagList = labels;
    if (!tagList || tagList.length === 0) {
      const seen = new Set<string>();
      for (const d of falsePredictions) {
        for (const s of [...d.gold_spans, ...d.pred_spans]) seen.add(s.tag);
      }
      tagList = Array.from(seen).sort();
    }
    const palette = chroma.scale('Paired').colors(Math.max(tagList.length, 2));
    return Object.fromEntries(tagList.map((t, i) => [t, palette[i] || '#bbbbbb']));
  }, [labels, falsePredictions]);

  const pageDocs = falsePredictions.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = Math.max(1, Math.ceil(falsePredictions.length / PAGE_SIZE));

  const toAnnotateTags = (spans: Span[]): AnnotateBlendTag[] =>
    spans.map((s) => ({
      start: s.start,
      end: s.end,
      tag: s.tag,
      color: colorMap[s.tag] ?? '#bbbbbb',
    }));

  // Normalize the dataset value to one of the annotation page's expected
  // querystring tokens. Same shape as DisplayScores' datasetClean.
  const datasetClean = (() => {
    const d = (dataset || '').toLowerCase();
    if (d.includes('test')) return 'test';
    if (d.includes('valid')) return 'valid';
    return 'train';
  })();

  if (falsePredictions.length === 0) {
    return (
      <div className="text-muted">No disagreements — predictions match gold on every span.</div>
    );
  }

  return (
    <div>
      <div className="d-flex flex-wrap gap-3 mb-3 small">
        <span className="text-muted me-2">
          {falsePredictions.length} document{falsePredictions.length === 1 ? '' : 's'} with
          disagreements
        </span>
        {Object.entries(colorMap).map(([tag, color]) => (
          <span key={tag} className="d-inline-flex align-items-center">
            <span
              style={{
                display: 'inline-block',
                width: 14,
                height: 14,
                background: color,
                marginRight: 4,
                borderRadius: 2,
              }}
            />
            {tag}
          </span>
        ))}
      </div>

      {pageDocs.map((doc) => (
        <div key={doc.id} className="card mb-3">
          <div className="card-body">
            <div className="text-muted small mb-2">
              id:{' '}
              {projectSlug ? (
                <Link to={`/projects/${projectSlug}/tag/${doc.id}?dataset=${datasetClean}`}>
                  {doc.id}
                </Link>
              ) : (
                doc.id
              )}
            </div>
            <div className="mb-2">
              <div className="text-muted small mb-1">Gold</div>
              <TextAnnotateBlend
                content={doc.text || ''}
                value={toAnnotateTags(doc.gold_spans)}
                onChange={() => {
                  /* read-only */
                }}
                style={{ fontSize: '1rem', whiteSpace: 'pre-wrap' }}
              />
            </div>
            <div>
              <div className="text-muted small mb-1">Predicted</div>
              <TextAnnotateBlend
                content={doc.text || ''}
                value={toAnnotateTags(doc.pred_spans)}
                onChange={() => {
                  /* read-only */
                }}
                style={{ fontSize: '1rem', whiteSpace: 'pre-wrap' }}
              />
            </div>
          </div>
        </div>
      ))}

      {totalPages > 1 && (
        <div className="d-flex justify-content-center align-items-center gap-2">
          <button
            className="btn btn-sm btn-outline-secondary"
            disabled={page === 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
          >
            ← Prev
          </button>
          <span className="small text-muted">
            Page {page + 1} / {totalPages}
          </span>
          <button
            className="btn btn-sm btn-outline-secondary"
            disabled={page >= totalPages - 1}
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
};
