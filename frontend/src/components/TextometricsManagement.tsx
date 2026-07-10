import { FC, useEffect, useMemo, useState } from 'react';
import { VictoryAxis, VictoryBar, VictoryChart, VictoryTheme } from 'victory';

import { useComputeTextometrics, useGetTextometrics } from '../core/api';
import { useAppContext } from '../core/useAppContext';
import { DistributionModel } from '../types';
import { StopProcessButton } from './StopProcessButton';

interface TextometricsManagementProps {
  projectSlug: string | null;
}

const summaryLabels: [keyof DistributionModel['summary'], string][] = [
  ['count', 'Documents'],
  ['mean', 'Mean'],
  ['std', 'Std'],
  ['min', 'Min'],
  ['q25', 'Q25'],
  ['median', 'Median'],
  ['q75', 'Q75'],
  ['max', 'Max'],
];

const DistributionDisplay: FC<{ title: string; distribution: DistributionModel }> = ({
  title,
  distribution,
}) => {
  // histogram bars positioned on bin centers
  const bars = useMemo(() => {
    const edges = distribution.histogram.bin_edges;
    return distribution.histogram.counts.map((count, i) => ({
      x: (edges[i] + edges[i + 1]) / 2,
      y: count,
    }));
  }, [distribution.histogram]);
  const binWidth =
    distribution.histogram.bin_edges.length > 1
      ? distribution.histogram.bin_edges[1] - distribution.histogram.bin_edges[0]
      : 1;

  return (
    <div>
      <h4 className="subsection">{title}</h4>
      <table className="table table-sm w-auto">
        <tbody>
          {summaryLabels.map(([key, label]) => (
            <tr key={key}>
              <td>{label}</td>
              <td>{distribution.summary[key] ?? '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <VictoryChart theme={VictoryTheme.material} width={600} height={300} domainPadding={10}>
        <VictoryAxis label={title} style={{ axisLabel: { padding: 30 } }} />
        <VictoryAxis dependentAxis label="Documents" style={{ axisLabel: { padding: 40 } }} />
        <VictoryBar
          data={bars}
          barWidth={Math.max(2, 550 / bars.length - 2)}
          style={{ data: { fill: '#0072B2' } }}
        />
      </VictoryChart>
      <span className="explanations">Bin width: {binWidth.toFixed(1)}</span>
    </div>
  );
};

/**
 * Textometry statistics of the train dataset: computed on demand by a
 * background task, then displayed (distributions + most frequent words)
 */
export const TextometricsManagement: FC<TextometricsManagementProps> = ({ projectSlug }) => {
  const {
    appContext: { currentProject: project },
  } = useAppContext();

  const isTraining = Object.keys(project?.textometrics?.training || {}).length > 0;
  const { textometrics, reFetchTextometrics } = useGetTextometrics(projectSlug);
  const computeTextometrics = useComputeTextometrics(projectSlug);
  const [nWordsDisplayed, setNWordsDisplayed] = useState<string>('20');

  // when the computation ends (training goes back to empty), refetch the results
  useEffect(() => {
    if (!isTraining) reFetchTextometrics();
  }, [isTraining, reFetchTextometrics]);

  if (isTraining)
    return (
      <div className="col-12 my-3">
        <div>Computing textometrics...</div>
        <StopProcessButton projectSlug={projectSlug} kind="textometrics" />
      </div>
    );

  if (!textometrics)
    return (
      <div className="col-12 my-3">
        <button className="btn btn-primary" onClick={computeTextometrics}>
          Compute textometrics
        </button>
      </div>
    );

  return (
    <div className="col-12 my-3">
      <div className="d-flex align-items-center gap-3 mb-3">
        <span className="explanations">
          Computed on {new Date(textometrics.computed_at).toLocaleString()} (tokenizer:{' '}
          {textometrics.parameters.tokenizer})
        </span>
        <button className="btn btn-secondary btn-sm" onClick={computeTextometrics}>
          Recompute
        </button>
      </div>
      <div className="row">
        <div className="col-md-6">
          <DistributionDisplay
            title="Words per document"
            distribution={textometrics.statistics.words_per_doc}
          />
        </div>
        <div className="col-md-6">
          <DistributionDisplay
            title="Tokens per document"
            distribution={textometrics.statistics.tokens_per_doc}
          />
        </div>
      </div>
      <div className="row mt-3">
        <div className="col-md-6">
          <h4 className="subsection">Most frequent words (stopwords excluded)</h4>
          <div className="d-flex align-items-center gap-2 mb-2">
            <label htmlFor="n-words-displayed">Words displayed</label>
            <select
              id="n-words-displayed"
              className="form-select form-select-sm w-auto"
              value={nWordsDisplayed}
              onChange={(e) => setNWordsDisplayed(e.target.value)}
            >
              <option value="20">20</option>
              <option value="50">50</option>
              <option value="all">All</option>
            </select>
          </div>
          <table className="table table-sm table-striped w-auto">
            <thead>
              <tr>
                <th>Word</th>
                <th>Count</th>
              </tr>
            </thead>
            <tbody>
              {(nWordsDisplayed === 'all'
                ? textometrics.statistics.most_frequent_words
                : textometrics.statistics.most_frequent_words.slice(0, Number(nWordsDisplayed))
              ).map((element) => (
                <tr key={element.word}>
                  <td>{element.word}</td>
                  <td>{element.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
