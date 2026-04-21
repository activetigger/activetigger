import { FC } from 'react';
import PulseLoader from 'react-spinners/PulseLoader';
import { useStopProcesses } from '../core/api';

interface stopProcessButtonProps {
  projectSlug: string | null;
  kind?: string;
}

export const StopProcessButton: FC<stopProcessButtonProps> = ({ projectSlug, kind = 'all' }) => {
  const { stopProcesses } = useStopProcesses(projectSlug);
  return (
    <button key="stop" className="btn-stop-process my-2" onClick={() => stopProcesses(kind)}>
      <PulseLoader color={'white'} className="mx-2" /> Stop
    </button>
  );
};
