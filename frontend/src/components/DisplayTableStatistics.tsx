import cx from 'classnames';
import { FC, useMemo } from 'react';
import { useWindowSize } from 'usehooks-ts';
import { useAppContext } from '../core/useAppContext';
import { reorderLabels } from '../core/utils';
import { MLStatisticsModel } from '../types';

export interface DisplayTableStatisticsProps {
  scores: MLStatisticsModel;
  title?: string | null;
}

interface TableModel {
  index: string[];
  columns: string[];
  data: number[][];
}

export const DisplayTableStatistics: FC<DisplayTableStatisticsProps> = ({ scores }) => {
  const { width: widthWindow } = useWindowSize();
  const {
    appContext: { displayConfig },
  } = useAppContext();
  const table = scores.table ? (scores.table as unknown as TableModel) : null;

  // sort labels and build a permutation to reorder data rows/columns accordingly
  const labels = useMemo<string[]>(
    () =>
      reorderLabels(
        (scores?.table?.index as unknown as string[]) || [],
        displayConfig.labelsOrder || [],
      ),
    [displayConfig.labelsOrder, scores],
  );
  const nLabels = Object.entries(labels).length;

  // permutation: for each position in `labels`, the original index in table.index
  const perm = useMemo<number[]>(() => {
    if (!table) return [];
    return labels.map((label) => table.index.indexOf(label));
  }, [labels, table]);

  // reordered data: rows and columns permuted to match `labels` order
  const reorderedData = useMemo<number[][]>(() => {
    if (!table) return [];
    return perm.map((origRow) => perm.map((origCol) => table.data[origRow][origCol]));
  }, [table, perm]);

  const isDiag = (colIndex: number, rowIndex: number) => {
    return colIndex < nLabels - 1 && colIndex === rowIndex;
  };
  const isTotal = (colIndex: number, rowIndex: number) => {
    return colIndex === nLabels - 1 || rowIndex === nLabels - 1;
  };

  const displayScore = (score: number | undefined) => {
    if (typeof score === 'number') {
      if (score === 0) {
        return '0.00';
      } else if (score === 1) {
        return '1.00';
      } else {
        return (String(score) + '0').slice(0, 4);
      }
    }
    return '';
  };

  const limitLabelSize = (label: string, type: string) => {
    let maxLength = 100;
    if (type == 'col') {
      maxLength = widthWindow > 1000 ? 6 : 4;
    } else if (type === 'row') {
      maxLength = widthWindow > 1000 ? 15 : 13;
    }

    if (label) {
      if (label.length > maxLength) {
        return label.slice(0, maxLength - 4) + '..' + label.slice(-2);
      } else {
        return label;
      }
    } else {
      return 'Loading...';
    }
  };

  return (
    <>
      {table && (
        <div id={`DisplayTableStatistics-${nLabels}`}>
          <div className="row">
            <div className="main row" style={{ justifyContent: 'flex-end' }}>
              <div id="truth-container" style={{ height: `${(nLabels + 2) * 30}px` }}>
                <span style={{ height: `${nLabels * 30}px` }}>Truth</span>
              </div>
              <div id="horizontal-labels-container" style={{ height: `${(nLabels + 2) * 30}px` }}>
                <div style={{ height: `${nLabels * 30}px` }}>
                  {labels.map((label, rowIndex) => (
                    <div
                      className={cx(
                        'table-cell label-column-left label-name',
                        isTotal(-1, rowIndex) ? ' total-cell' : '',
                      )}
                      key={label}
                      title={label}
                    >
                      {limitLabelSize(label, 'row')}
                    </div>
                  ))}
                </div>
              </div>
              <div className="table">
                <div className="row">
                  {/* Predicted overlay row */}
                  <div
                    className="table-cell"
                    style={{ flex: `${nLabels} 1 auto` }}
                    id="overlay-label"
                  >
                    Predicted
                  </div>
                </div>
                <div className="row">
                  {/* Labels row */}
                  {labels.map((label, colIndex) => (
                    <div
                      className={cx(
                        'table-cell label-name',
                        isTotal(colIndex, -1) ? ' total-cell' : '',
                      )}
                      key={label}
                      title={label}
                    >
                      {limitLabelSize(label, 'col')}
                    </div>
                  ))}
                </div>
                {reorderedData.map((row, rowIndex) => (
                  // All data
                  <div className="row" key={rowIndex}>
                    {row.map((cell, colIndex) => (
                      <div
                        className={cx(
                          'table-cell number-cell',
                          isDiag(colIndex, rowIndex) ? ' diag-cell' : '',
                          isTotal(colIndex, rowIndex) ? ' total-cell' : '',
                        )}
                        key={colIndex}
                      >
                        {cell}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
            <div className="score-left row">
              <div className="table">
                <div className="row">
                  <div className="table-cell" style={{ flex: `0 .5 auto` }} id="overlay-label">
                    Scores
                  </div>
                </div>
                <div className="row">
                  <div className="table-cell">{limitLabelSize('Recall', 'col')}</div>
                  <div className="table-cell">{limitLabelSize('F1', 'col')}</div>
                </div>
                {labels.map((label, rowIndex) => (
                  <div className="row" key={rowIndex}>
                    <div className="table-cell number-cell">
                      {scores.recall_label && displayScore(scores.recall_label[label])}
                    </div>
                    <div className="table-cell number-cell">
                      {scores.f1_label && displayScore(scores.f1_label[label])}
                    </div>
                  </div>
                ))}
              </div>
              <div id="horizontal-labels-container" style={{ height: `${(nLabels + 2) * 30}px` }}>
                <div style={{ height: `${nLabels * 30}px` }}>
                  {labels.map((label) => (
                    <div
                      className="table-cell label-column-right label-name recall"
                      key={label}
                      title={label}
                    >
                      {label !== 'Total' ? limitLabelSize(label, 'row') : ''}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          <div className="score-bottom row" style={{ justifyContent: 'flex-end' }}>
            <div id="truth-container" style={{ height: `${2 * 30}px` }}>
              <span style={{ height: `${2 * 30}px` }}>Scores</span>
            </div>
            <div id="horizontal-labels-container" style={{ height: '60px' }}>
              <div style={{ height: '60px' }}>
                <div className="table-cell label-column-left">
                  {limitLabelSize('Precision', 'row')}
                </div>
                <div className="table-cell label-column-left">{limitLabelSize('F1', 'row')}</div>
                <div className="table-cell label-column-left"></div>
              </div>
            </div>
            <div className="table">
              <div className="row">
                {labels.map((label, colIndex) => (
                  <div key={colIndex} className="table-cell number-cell">
                    {scores.precision_label && displayScore(scores.precision_label[label])}
                  </div>
                ))}
              </div>
              <div className="row">
                {labels.map((label, colIndex) => (
                  <div key={colIndex} className="table-cell number-cell">
                    {scores.f1_label && displayScore(scores.f1_label[label])}
                  </div>
                ))}
              </div>
              <div className="row">
                {labels.map((label, colIndex) => (
                  <div key={colIndex} className="table-cell label-name recall" title={label}>
                    {label !== 'Total' ? limitLabelSize(label, 'col') : ''}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
