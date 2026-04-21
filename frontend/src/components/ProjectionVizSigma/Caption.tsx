import { toPairs } from 'lodash';
import { FC } from 'react';

// Node caption
// export const Caption: FC<{ labelColorMapping: { [key: string]: string } }> = ({
//   labelColorMapping,
// }) => {
//   return (
//     <div className="p-1">
//       {toPairs(labelColorMapping).map(([label, color]) => (
//         <div className="d-flex align-items-center" key={label}>
//           <div className="node me-1" style={{ backgroundColor: color }} /> <span>{label}</span>
//         </div>
//       ))}
//     </div>
//   );
// };
export const Caption: FC<{ labelColorMapping: { [key: string]: string } }> = ({
  labelColorMapping,
}) => {
  const entries = toPairs(labelColorMapping);
  // const twoColumns = entries.length > 10;
  const twoColumns = false;
  const fontSize = entries.length > 10 ? '0.75rem' : '0.875rem';

  return (
    <div className="p-1">
      <div
        className={`d-inline-flex flex-column ${twoColumns ? 'flex-wrap' : ''}`}
        style={twoColumns ? { maxHeight: '200px' } : {}}
      >
        {entries.map(([label, color]) => (
          <div className="d-flex align-items-center me-3 mb-1" key={label}>
            <div
              className="node me-1"
              style={{ backgroundColor: color, width: 12, height: 12, borderRadius: '50%' }}
              aria-hidden="true"
            />
            <span style={{ fontSize }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
