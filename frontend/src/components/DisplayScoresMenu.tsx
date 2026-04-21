import { FC, useEffect, useState } from 'react';
import { MLStatisticsModel } from '../types';
import { DisplayScores } from './DisplayScores';

interface Scores {
  [key: string]: MLStatisticsModel;
}

interface DisplayScoresMenuPropos {
  projectSlug?: string | null;
  scores: Scores;
  modelName?: string;
  displayTitle?: boolean;
  skip?: string[];
  exclude_labels?: string[];
}

export const DisplayScoresMenu: FC<DisplayScoresMenuPropos> = ({
  scores,
  modelName,
  displayTitle,
  skip,
  projectSlug,
  exclude_labels,
}) => {
  const allowedScores = Object.entries(scores)
    .filter(([_, value]) => value != null)
    .filter(([key]) => !skip?.includes(key));
  const scoreKeys = allowedScores.map(([key]) => key);
  const [currentScore, setCurrentScore] = useState<string>(scoreKeys[0] || '');
  // Ensure currentScore is still valid when scores change
  useEffect(() => {
    if (!scoreKeys.includes(currentScore)) {
      setCurrentScore(Object.keys(scoreKeys)[0] || '');
    }
  }, [scores, currentScore, allowedScores, scoreKeys]);

  if (!scores || Object.keys(scores).length === 0) {
    return <div>No scores available</div>;
  }

  const displayScores: { [key: string]: string } = {
    train_scores: 'Train (all)',
    outofsample_scores: 'Train (eval)',
    valid_scores: 'Validation set',
    test_scores: 'Test set',
  };

  return (
    <>
      <div className="horizontal">
        <label htmlFor="statistics" style={{ marginRight: '10px' }}>
          Scores{' '}
        </label>
        <select
          id="statistics"
          value={currentScore}
          onChange={(e) => setCurrentScore(e.target.value)}
          style={{ maxWidth: '200px' }}
        >
          {allowedScores.map(([key]) => (
            <option key={key} value={key}>
              {displayScores[key] || key}
            </option>
          ))}
        </select>
      </div>
      {scores && (
        <DisplayScores
          title={displayTitle ? currentScore : null}
          scores={scores[currentScore]}
          modelName={modelName}
          projectSlug={projectSlug}
          dataset={currentScore}
          exclude_labels={exclude_labels}
        />
      )}
    </>
  );
};
