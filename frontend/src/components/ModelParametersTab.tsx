import { ReactNode } from 'react';

function renderValue(value: unknown): ReactNode {
  if (value === null) {
    return <em>null</em>;
  }

  if (value === undefined) {
    return <em>undefined</em>;
  }

  if (typeof value === 'string') {
    return <span>{value}</span>;
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return <code>{String(value)}</code>;
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <em>[]</em>;
    }

    return (
      <ul style={{ margin: 0, paddingLeft: '1.2em' }}>
        {value.map((item, i) => (
          <li key={i}>{renderValue(item)}</li>
        ))}
      </ul>
    );
  }

  if (typeof value === 'object') {
    return (
      <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{JSON.stringify(value, null, 2)}</pre>
    );
  }

  return <code>{String(value)}</code>;
}

import { FC } from 'react';

interface ModelParametersTabProps {
  params: Record<string, unknown>;
  sortKeys?: boolean;
}

export const ModelParametersTab: FC<ModelParametersTabProps> = ({ params, sortKeys = true }) => {
  const entries = Object.entries(params || {});
  if (sortKeys) {
    entries.sort(([keyA], [keyB]) => {
      if (keyA === 'base_model') return -1;
      if (keyB === 'base_model') return 1;
      return keyA.localeCompare(keyB);
    });
  }
  return (
    <table id="parameter-tables-thin">
      <tbody>
        {entries.map(([key, value], index) => (
          <tr key={key} className={index % 2 === 0 ? 'dark' : ''}>
            <td>
              <code>{key}</code>
            </td>
            <td>{renderValue(value)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};
