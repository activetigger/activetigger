import { FC } from 'react';
import DataGrid, { Column } from 'react-data-grid';

export interface Row {
  Topic: number;
  Count: number;
  Name: string;
  Representation: string[];
  Representative_Docs: string[];
}

interface TableTopicsProps {
  data: Row[];
}

export const DisplayTableTopics: FC<TableTopicsProps> = ({ data }) => {
  const columns: readonly Column<Row>[] = [
    { key: 'Topic', name: 'Topic', width: 80 },
    { key: 'Count', name: 'Count', width: 90 },
    { key: 'Name', name: 'Name', resizable: true },
    {
      key: 'Representation',
      name: 'Representation',
      resizable: true,
      renderCell: (props) => (
        <div
          style={{
            maxHeight: '100%',
            width: '100%',
            whiteSpace: 'wrap',
            overflowY: 'auto',
            userSelect: 'none',
          }}
        >
          {props.row.Representation}
        </div>
      ),
    },
    {
      key: 'Representative_Docs',
      name: 'Representative Docs',
      resizable: true,
      renderCell: (props) => (
        <div
          style={{
            maxHeight: '100%',
            width: '100%',
            whiteSpace: 'wrap',
            overflowY: 'auto',
            userSelect: 'none',
          }}
        >
          {props.row.Representative_Docs}
        </div>
      ),
    },
  ];

  return (
    <div
      style={{
        inlineSize: '100%',
        minInlineSize: 600,
        blockSize: 80 * data.length,
        display: 'grid',
        gap: 8,
      }}
    >
      <div style={{ blockSize: '100%' }}>
        <DataGrid columns={columns} rows={data} className="fill-grid rdg-light" rowHeight={80} />
      </div>
    </div>
  );
};
