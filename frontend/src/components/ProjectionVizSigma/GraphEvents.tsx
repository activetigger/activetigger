import { useRegisterEvents } from '@react-sigma/core';
import { Dispatch, SetStateAction, useEffect } from 'react';
import { SigmaCursorTypes } from '.';

const GraphEvents: React.FC<{
  setSelectedIdAfterClick: (id?: string) => void;
  setSigmaCursor: Dispatch<SetStateAction<SigmaCursorTypes>>;
  setClusterHighlightAfterDoubleClick: (id?: string) => void;
}> = ({ setSelectedIdAfterClick, setSigmaCursor, setClusterHighlightAfterDoubleClick }) => {
  const registerEvents = useRegisterEvents();

  useEffect(() => {
    // Register the events
    registerEvents({
      // (un)Select node
      clickNode: (event) => {
        setSelectedIdAfterClick(event.node);
      },
      clickStage: () => setSelectedIdAfterClick(undefined),
      // pointer cursor
      enterNode: () => {
        setSigmaCursor('pointer');
      },
      leaveNode: () => {
        setSigmaCursor(undefined);
      },
      // grabbing cursor
      mousedown: () => {
        setSigmaCursor('grabbing');
      },
      mouseup: () => {
        setSigmaCursor(undefined);
      },
      doubleClickNode: (event) => {
        setClusterHighlightAfterDoubleClick(event.node);
        console.log(event.node);
      },
      doubleClick: (event) => {
        // Prevent zooming in behaviour https://github.com/jacomyal/sigma.js/issues/910
        event.preventSigmaDefault();
      },
      doubleClickStage: () => {
        setClusterHighlightAfterDoubleClick(undefined);
      },
    });
  }, [
    registerEvents,
    setClusterHighlightAfterDoubleClick,
    setSelectedIdAfterClick,
    setSigmaCursor,
  ]);

  return null;
};

export default GraphEvents;
