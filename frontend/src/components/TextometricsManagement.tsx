import { FC, ReactNode, useEffect, useMemo, useState } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import { Link } from 'react-router-dom';
import Select from 'react-select';
import { VictoryAxis, VictoryBar, VictoryChart, VictoryTheme } from 'victory';

import { useComputeTextometrics, useGetTextometrics } from '../core/api';
import { useAppContext } from '../core/useAppContext';
import { DistributionModel } from '../types';
import { StopProcessButton } from './StopProcessButton';

interface TextometricsManagementProps {
  projectSlug: string | null;
}

// light yellow pill for counts/scores
const countBadgeStyle = { backgroundColor: '#fff3cd', color: '#664d03' };

/**
 * Compact multicolumn ranked list: entries flow over as many columns as the
 * width allows, with the count/score displayed as a pill
 */
const CompactRankedList: FC<{
  items: { key: string; label: ReactNode; value: number | string }[];
}> = ({ items }) => (
  <ul className="list-unstyled mb-0" style={{ columnWidth: '14em', columnGap: '2.5em' }}>
    {items.map((item, index) => (
      <li
        key={item.key}
        className="d-flex justify-content-between border-bottom py-1"
        style={{ breakInside: 'avoid' }}
      >
        <span>
          <span className="text-muted me-2">{index + 1}.</span>
          {item.label}
        </span>
        <span className="badge align-self-center" style={countBadgeStyle}>
          {item.value}
        </span>
      </li>
    ))}
  </ul>
);

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

const DistributionSummaryTable: FC<{
  distributions: { label: string; distribution: DistributionModel }[];
}> = ({ distributions }) => (
  <table className="table table-sm w-auto">
    <thead>
      <tr>
        <th></th>
        {distributions.map(({ label }) => (
          <th key={label}>{label}</th>
        ))}
      </tr>
    </thead>
    <tbody>
      {summaryLabels.map(([key, label]) => (
        <tr key={key}>
          <td>{label}</td>
          {distributions.map(({ label: column, distribution }) => (
            <td key={column}>{distribution.summary[key] ?? '-'}</td>
          ))}
        </tr>
      ))}
    </tbody>
  </table>
);

const DistributionHistogram: FC<{ title: string; distribution: DistributionModel }> = ({
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
    <div style={{ maxWidth: '420px' }}>
      <VictoryChart theme={VictoryTheme.material} width={600} height={220} domainPadding={10}>
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

  // tf-idf explorer state
  const [selectedWord, setSelectedWord] = useState<string | null>(null);
  const [documentFilter, setDocumentFilter] = useState<string>('');
  const tfidfWords = textometrics?.statistics.tfidf_words;
  const tfidfDocuments = textometrics?.statistics.tfidf_documents;
  const wordOptions = useMemo(
    () =>
      (tfidfWords || []).map((element) => ({
        value: element.word,
        label: `${element.word} (${element.n_documents} docs)`,
      })),
    [tfidfWords],
  );
  const currentWord = useMemo(
    () => (tfidfWords || []).find((element) => element.word === selectedWord),
    [tfidfWords, selectedWord],
  );
  const filteredDocuments = useMemo(() => {
    const documents = tfidfDocuments || [];
    return (
      documentFilter
        ? documents.filter((element) => element.element_id.includes(documentFilter))
        : documents
    ).slice(0, 50);
  }, [tfidfDocuments, documentFilter]);

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
        <div className="col-12">
          <h4 className="subsection">Document length</h4>
        </div>
        <div className="col-md-4">
          <DistributionSummaryTable
            distributions={[
              { label: 'Words', distribution: textometrics.statistics.words_per_doc },
              { label: 'Tokens', distribution: textometrics.statistics.tokens_per_doc },
            ]}
          />
        </div>
        <div className="col-md-8">
          <DistributionHistogram
            title="Words per document"
            distribution={textometrics.statistics.words_per_doc}
          />
          <DistributionHistogram
            title="Tokens per document"
            distribution={textometrics.statistics.tokens_per_doc}
          />
        </div>
      </div>
      <div className="row mt-3">
        <div className="col-12">
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
          <CompactRankedList
            items={(nWordsDisplayed === 'all'
              ? textometrics.statistics.most_frequent_words
              : textometrics.statistics.most_frequent_words.slice(0, Number(nWordsDisplayed))
            ).map((element) => ({
              key: element.word,
              label: element.word,
              value: element.count,
            }))}
          />
        </div>
      </div>
      <div className="row mt-3">
        <div className="col-12">
          <h4 className="subsection">TF-IDF explorer</h4>
          {!tfidfWords ? (
            <div className="alert alert-info">
              Recompute textometrics to get the TF-IDF statistics.
            </div>
          ) : (
            <Tabs defaultActiveKey="byword" className="mb-2">
              <Tab eventKey="byword" title="By word">
                <div className="explanations">
                  Select a word to display the documents where it is the most distinctive (highest
                  TF-IDF score)
                </div>
                <div className="col-md-4 mb-2">
                  <Select
                    options={wordOptions}
                    value={wordOptions.find((option) => option.value === selectedWord) || null}
                    onChange={(option) => setSelectedWord(option ? option.value : null)}
                    isClearable
                    placeholder="Select a word..."
                  />
                </div>
                {currentWord && (
                  <CompactRankedList
                    items={currentWord.top_documents.map((element) => ({
                      key: element.element_id,
                      label: (
                        <Link to={`/projects/${projectSlug}/tag/${element.element_id}`}>
                          {element.element_id}
                        </Link>
                      ),
                      value: element.score,
                    }))}
                  />
                )}
              </Tab>
              <Tab eventKey="bydocument" title="By document">
                {!tfidfDocuments ? (
                  <div className="alert alert-info">
                    The train set is larger than the limit (
                    {textometrics.parameters.tfidf_max_documents} documents): the per-document
                    TF-IDF view is not stored to keep the statistics file small.
                  </div>
                ) : (
                  <>
                    <div className="explanations">
                      Most distinctive words (highest TF-IDF scores) of each document
                    </div>
                    <div className="col-md-4 mb-2">
                      <input
                        type="text"
                        className="form-control form-control-sm"
                        placeholder="Filter by document id..."
                        value={documentFilter}
                        onChange={(e) => setDocumentFilter(e.target.value)}
                      />
                    </div>
                    <table className="table table-sm table-striped">
                      <thead>
                        <tr>
                          <th>Document</th>
                          <th>Most distinctive words</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredDocuments.map((element) => (
                          <tr key={element.element_id}>
                            <td>
                              <Link to={`/projects/${projectSlug}/tag/${element.element_id}`}>
                                {element.element_id}
                              </Link>
                            </td>
                            <td>
                              {element.top_words.map((word) => (
                                <span
                                  key={word.word}
                                  className="badge me-1"
                                  style={countBadgeStyle}
                                >
                                  {word.word} {word.score}
                                </span>
                              ))}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {tfidfDocuments.length > filteredDocuments.length && (
                      <span className="explanations">
                        Showing the first {filteredDocuments.length} documents — use the filter to
                        narrow down
                      </span>
                    )}
                  </>
                )}
              </Tab>
            </Tabs>
          )}
        </div>
      </div>
    </div>
  );
};
