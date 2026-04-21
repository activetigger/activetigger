import { FC } from 'react';

export interface DisplayMatrixProps {
  labels: string[];
  matrix: number[][];
  f1Labels?: Record<string, string>;
}
// NOTE: Axel: I believe it is not used. Delete?
export const DisplayMatrix: FC<DisplayMatrixProps> = ({ matrix, labels }) => {
  // Calculate row totals
  const rowTotals = matrix.map((row) => row.reduce((sum, val) => sum + val, 0));

  // Calculate column totals
  const columnTotals = labels.map((_, colIndex) =>
    matrix.reduce((sum, row) => sum + row[colIndex], 0),
  );

  // Calculate grand total
  const grandTotal = columnTotals.reduce((sum, val) => sum + val, 0);

  return (
    <div className="overflow-x-auto p-4">
      <table className="border-collapse border border-gray-300 w-full text-center">
        <thead>
          <tr>
            <th rowSpan={2} className="border border-gray-300 p-2 bg-gray-200"></th>
            <th colSpan={labels.length} className="border border-gray-300 p-2 bg-gray-300">
              Predicted
            </th>
            <th rowSpan={2} className="border border-gray-300 p-2 bg-gray-300">
              Total
            </th>
          </tr>
          <tr className="bg-gray-200">
            {labels.map((label, i) => (
              <th key={i} scope="col" className="border border-gray-300 p-2">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i} className="hover:bg-blue-100 transition">
              <th scope="row" className="border border-gray-300 p-2 bg-gray-200">
                {labels[i]}
              </th>
              {row.map((val, j) => (
                <td key={j} className="border border-gray-300 p-2">
                  {val}
                </td>
              ))}
              {/* Row total */}
              <td className="border border-gray-300 p-2 font-bold bg-gray-100">{rowTotals[i]}</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr className="bg-gray-300">
            <th className="border border-gray-300 p-2">Total</th>
            {columnTotals.map((total, i) => (
              <td key={i} className="border border-gray-300 p-2 font-bold">
                {total}
              </td>
            ))}
            {/* Grand Total */}
            <td className="border border-gray-300 p-2 font-bold bg-gray-200">{grandTotal}</td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
};
