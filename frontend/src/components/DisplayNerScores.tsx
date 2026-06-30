import { FC, useEffect, useState } from 'react';
import { MLStatisticsModel } from '../types';
import { DisplayScores } from './DisplayScores';

type NerFlavor = 'exact' | 'partial';

type NerScoreBundle = {
  training_kind?: string;
  exact?: MLStatisticsModel;
  partial?: MLStatisticsModel;
  // "type" is still written to the JSON file by the backend for power users
  // but intentionally not surfaced in the UI.
};

interface DisplayNerScoresProps {
  title: string | null;
  scores: NerScoreBundle | null | undefined;
  modelName?: string;
  projectSlug?: string | null;
  dataset?: string;
}

const FLAVORS: Array<{ key: NerFlavor; label: string }> = [
  { key: 'exact', label: 'Exact (same boundaries + tag)' },
  { key: 'partial', label: 'Partial (same tag, any overlap)' },
];

/**
 * NER-specific wrapper around DisplayScores: same layout as multilabel /
 * multiclass (Macro F1 line + confusion matrix and per-label P/R/F1 table
 * side-by-side), with a selector that swaps between exact and partial
 * matching.
 */
export const DisplayNerScores: FC<DisplayNerScoresProps> = ({
  title,
  scores,
  modelName,
  projectSlug,
  dataset,
}) => {
  const [flavor, setFlavor] = useState<NerFlavor>('exact');
  // If the selected flavor isn't populated for this split, fall back to
  // whichever one is present so the table doesn't disappear silently.
  useEffect(() => {
    if (scores && !scores[flavor] && scores.exact) setFlavor('exact');
    else if (scores && !scores[flavor] && scores.partial) setFlavor('partial');
  }, [scores, flavor]);

  if (!scores) return null;
  const current = scores[flavor];

  return (
    <div className="my-3">
      {title && <h5 className="subsection">{title}</h5>}
      <div className="horizontal">
        <label htmlFor="ner-matching" style={{ marginRight: '10px' }}>
          Matching{' '}
        </label>
        <select
          id="ner-matching"
          value={flavor}
          onChange={(e) => setFlavor(e.target.value as NerFlavor)}
          style={{ maxWidth: '200px' }}
        >
          {FLAVORS.map((f) => (
            <option key={f.key} value={f.key}>
              {f.label}
            </option>
          ))}
        </select>
      </div>
      {current ? (
        <DisplayScores
          title={null}
          scores={current as MLStatisticsModel}
          modelName={modelName}
          projectSlug={projectSlug}
          dataset={dataset}
        />
      ) : (
        <div className="text-muted">No metrics for this flavor.</div>
      )}
    </div>
  );
};
