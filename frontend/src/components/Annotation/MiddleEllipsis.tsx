import { FC } from 'react';

export const MiddleEllipsis: FC<{ label: string; forceComplete?: boolean }> = ({
  label,
  forceComplete = false,
}) => {
  if (forceComplete) {
    return (
      <span className="label-full" title={label}>
        {label}
      </span>
    );
  }
  const splitIndex = Math.ceil(label.length / 2);
  return (
    <span className="truncate-middle" title={label}>
      <span>{label.slice(0, splitIndex || 1)}</span>
      <span>{label.slice(splitIndex)}</span>
    </span>
  );
};
