import { FC, useCallback, useMemo } from 'react';
import DataTable, { TableColumn, TableStyles } from 'react-data-table-component';
import { CSSObject } from 'styled-components';
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

type rowData = Record<string, string | number>;

export const DisplayTableStatisticsReact: FC<DisplayTableStatisticsProps> = ({ scores }) => {
  const {
    appContext: { displayConfig },
  } = useAppContext();
  // Table format and styling
  const rowHeightPX = 48;
  const labelStyle: CSSObject = useMemo(
    () => ({
      fontWeight: 'bold',
      fontSize: '.8rem',
      whiteSpace: 'nowrap',
      overflow: 'hidden',
      textOverflow: 'ellipsis',
    }),
    [],
  );
  const customTableStyle: TableStyles = {
    rows: { style: { height: `${rowHeightPX}px` } },
    headCells: { style: { ...labelStyle, height: `${rowHeightPX}px` } },
  };
  const genericCellStyle = useMemo(
    () => ({
      minWidth: 'fit-content',
      center: true,
      grow: 1,
    }),
    [],
  );
  const genericLabelCellStyle = useMemo(
    () => ({
      minWidth: '8rem',
      style: labelStyle,
      grow: 2,
    }),
    [labelStyle],
  );
  const cellFormat = useCallback(
    (row: rowData, _rowIndex: number, column: TableColumn<rowData>) => {
      const columnName: string = column.name as unknown as string;
      const content = row[columnName];

      // Display labels vertical
      if (columnName.length === 0) return <>{row['_rowLabel']}</>;

      // Display total values in grey
      if (columnName === 'Total' || row['_rowLabel'] === 'Total')
        return <p style={{ margin: 'auto', color: '#909090' }}>{content}</p>;

      // Display diagonal values in green
      if (row['_rowLabel'] == columnName)
        return (
          <p
            style={{
              backgroundColor: '#90909040',
              margin: 'auto',
              padding: '.5rem 1rem',
              borderRadius: '.8rem',
              color: 'green',
              fontWeight: 'bold',
            }}
          >
            {content}
          </p>
        );
      return <p style={{ margin: 'auto', color: 'red', fontWeight: 'bold' }}>{content}</p>;
    },
    [],
  );

  // Data Management
  const table = scores.table ? (scores.table as unknown as TableModel) : null;
  const reorderedLabels = useMemo<string[]>(
    () =>
      reorderLabels(
        (scores?.table?.index as unknown as string[]) || [],
        displayConfig.labelsOrder || [],
      ),
    [displayConfig.labelsOrder, scores],
  );
  const reorderedConfusionMatrix = useMemo<number[][]>(() => {
    if (!table) return [];
    const permutations: number[] = reorderedLabels.map((label) => table.index.indexOf(label));
    return permutations.map((origRow) =>
      permutations.map((origCol) => table.data[origRow][origCol]),
    );
  }, [table, reorderedLabels]);

  const [confusionMatrixColumns, confusionMatrixData] = useMemo<
    [TableColumn<rowData>[], rowData[]]
  >(() => {
    const tableColumns: TableColumn<rowData>[] = [
      {
        name: '',
        selector: (row: Record<string, string | number>) => row['_rowLabel'],
        ...genericLabelCellStyle,
        cell: cellFormat,
      },
      ...reorderedLabels.map((label) => {
        return {
          name: label,
          selector: (row: Record<string, string | number>) => row[label],
          ...genericCellStyle,
          cell: cellFormat,
        };
      }),
    ];
    const tableData = reorderedLabels.map((rowLabel, iRow) => {
      return {
        _rowLabel: rowLabel,
        ...Object.fromEntries(
          reorderedLabels.map((columnLabel, iCol) => {
            return [columnLabel, reorderedConfusionMatrix[iRow][iCol]];
          }),
        ),
      } as rowData;
    });
    return [tableColumns, tableData];
  }, [
    reorderedLabels,
    reorderedConfusionMatrix,
    cellFormat,
    genericCellStyle,
    genericLabelCellStyle,
  ]);

  const [scoresColumns, scoresData] = useMemo<[TableColumn<rowData>[], rowData[]]>(() => {
    const tableColumns: TableColumn<rowData>[] = [
      {
        name: '',
        selector: (row) => row['_rowLabel'],
        ...genericLabelCellStyle,
      },
      ...['Recall', 'Precision', 'F1'].map((score) => {
        return {
          name: score,
          selector: (row: Record<string, string | number>) => row[score],
          ...genericCellStyle,
        };
      }),
    ];
    const scoreData: rowData[] = reorderedLabels
      .filter((label) => label !== 'Total')
      .map((rowLabel) => {
        return {
          _rowLabel: rowLabel,
          Recall: scores.recall_label ? scores.recall_label[rowLabel] : 'NaN',
          Precision: scores.precision_label ? scores.precision_label[rowLabel] : 'NaN',
          F1: scores.f1_label ? scores.f1_label[rowLabel] : 'NaN',
        } as rowData;
      });
    return [tableColumns, scoreData];
  }, [reorderedLabels, scores, genericCellStyle, genericLabelCellStyle]);

  return (
    <>
      {table && (
        <>
          <div id="table-statistics-react">
            <div id="confusion-matrix-container">
              <div id="truth-container">
                <span style={{ height: rowHeightPX * reorderedLabels.length + 'px' }}>Truth</span>
              </div>
              <div style={{ flex: '1 1 50px' }}>
                <div id="predicted-container">
                  <span id="gap-span"></span>
                  <span
                    id="predicted-span"
                    style={{ flex: `${reorderedLabels.length + 1} 1 50px` }}
                  >
                    Predicted
                  </span>
                </div>
                <DataTable<rowData>
                  columns={confusionMatrixColumns}
                  data={confusionMatrixData}
                  customStyles={customTableStyle}
                />
              </div>
            </div>
            <div id="scores-table-container">
              <DataTable<rowData>
                columns={scoresColumns}
                data={scoresData}
                customStyles={customTableStyle}
              />
            </div>
          </div>
        </>
      )}
    </>
  );
};
