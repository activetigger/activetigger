import { FC } from 'react';
import { DisplayScores } from './DisplayScores';

interface QuickModelDisplayProps {
  projectSlug: string;
  currentModel?: Record<string, never>;
}

// NOTE: Axel; Not used
export const QuickModelDisplay: FC<QuickModelDisplayProps> = ({ currentModel, projectSlug }) => {
  // if no model, return nothing
  if (!currentModel) return null;

  return (
    <div>
      <h5 className="subsection">Current {currentModel.model} model</h5>
      <details className="m-2">
        <summary>Model parameters</summary>
        <table className="table table-striped table-hover">
          <tbody>
            {currentModel.params &&
              Object.entries(currentModel.params).map(([key, value], i) => (
                <tr key={i}>
                  <td>{key}</td>
                  <td>
                    {Array.isArray(value)
                      ? value.join(', ') // or use bullets if you prefer
                      : typeof value === 'object' && value !== null
                        ? JSON.stringify(value, null, 2)
                        : String(value)}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </details>
      <div>
        <DisplayScores
          title=""
          scores={currentModel.statistics_test as unknown as Record<string, number>}
          projectSlug={projectSlug}
        />
        {currentModel.statistics_cv10 && (
          <DisplayScores
            title="Cross validation CV10"
            scores={currentModel.statistics_cv10 as unknown as Record<string, number>}
            projectSlug={projectSlug}
          />
        )}
      </div>
    </div>
  );
};
