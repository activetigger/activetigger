import { useRegisterEvents, useSigma } from '@react-sigma/core';
import { pick } from 'lodash';
import { Dispatch, FC, SetStateAction, useCallback, useEffect, useRef, useState } from 'react';
import { Coordinates } from 'sigma/types';

import classNames from 'classnames';
import { PiCursorFill, PiSelectionBold } from 'react-icons/pi';
import { SigmaToolsType } from '.';

export interface MarqueBoundingBox {
  x: { min: number; max: number };
  y: { min: number; max: number };
}

export const MarqueeController: FC<{
  setBbox: Dispatch<SetStateAction<MarqueBoundingBox | undefined>>;
  validateBoundingBox: (boundingBox?: MarqueBoundingBox) => void;
  setActiveTool: Dispatch<SetStateAction<SigmaToolsType>>;
}> = ({ setBbox, validateBoundingBox, setActiveTool }) => {
  // sigma hooks
  const sigma = useSigma();
  const registerEvents = useRegisterEvents();

  // internal state
  const [selectionState, setSelectionState] = useState<
    | { type: 'off' }
    | { type: 'idle' }
    | {
        type: 'marquee';
        startCorner: Coordinates;
        mouseCorner: Coordinates;
      }
  >({ type: 'off' });

  // Use a ref so event handlers always see the latest selectionState
  // without needing to re-register events on every state change
  const selectionStateRef = useRef(selectionState);
  selectionStateRef.current = selectionState;

  // cleaning state when marquee closes
  const backToIdle = useCallback(() => {
    console.log('closing marquee');
    sigma.getCamera().enable();
    setSelectionState({ type: 'idle' });
  }, [sigma]);

  const closeMarkee = useCallback(() => {
    console.log('stop marquee');
    setBbox((prev) => {
      setTimeout(() => validateBoundingBox(prev), 0);
      return prev;
    });
    backToIdle();
  }, [validateBoundingBox, setBbox, backToIdle]);

  // Keyboard events
  useEffect(() => {
    const keyDownHandler = (e: KeyboardEvent) => {
      const state = selectionStateRef.current;
      if (state.type === 'idle') return;
      if (state.type === 'marquee' && e.key === 'Escape') {
        setBbox(undefined);
        validateBoundingBox(undefined);
        backToIdle();
      }
    };

    window.document.body.addEventListener('keydown', keyDownHandler);
    return () => {
      window.document.body.removeEventListener('keydown', keyDownHandler);
    };
  }, [backToIdle, validateBoundingBox, setBbox]);

  useEffect(() => {
    registerEvents({
      mousemovebody: (e) => {
        const state = selectionStateRef.current;
        // update bbox if ongoing marquee drawing
        if (state.type === 'marquee') {
          const mousePosition = pick(e, 'x', 'y') as Coordinates;

          const start = sigma.viewportToGraph(state.startCorner);
          const end = sigma.viewportToGraph(mousePosition);

          const minX = Math.min(start.x, end.x);
          const minY = Math.min(start.y, end.y);
          const maxX = Math.max(start.x, end.x);
          const maxY = Math.max(start.y, end.y);

          // update bbox state to update marquee display
          setBbox({ x: { min: minX, max: maxX }, y: { min: minY, max: maxY } });

          setSelectionState({
            ...state,
            mouseCorner: mousePosition,
          });
        }
      },
      clickStage: (e) => {
        const state = selectionStateRef.current;
        // start / stop Marquee drawing
        if (state.type !== 'off') {
          e.preventSigmaDefault();

          if (state.type === 'idle') {
            console.log('start marquee');
            const mousePosition: Coordinates = pick(e.event, 'x', 'y');

            setSelectionState({
              type: 'marquee',
              startCorner: mousePosition,
              mouseCorner: mousePosition,
            });
            sigma.getCamera().disable();
          } else {
            closeMarkee();
          }
        }
      },
      click: (e) => {
        const state = selectionStateRef.current;
        // to make sure a click elsewhere than stage closes the marquee
        if (state.type === 'marquee') {
          e.preventSigmaDefault();
          closeMarkee();
        }
      },
    });
  }, [registerEvents, sigma, setBbox, closeMarkee]);

  return (
    <>
      <div className="react-sigma-control">
        {/* normal zoom-pan tool activation button*/}
        <button
          className={classNames(
            selectionState.type === 'off' ? 'bg-primary text-light' : 'cursor-pointer',
          )}
          disabled={selectionState.type === 'off'}
          onClick={() => {
            if (selectionState.type !== 'off') {
              setSelectionState({ type: 'off' });
              setActiveTool('panZoom');
            }
          }}
        >
          <PiCursorFill />
        </button>
      </div>
      <div className="react-sigma-control">
        {/* normal marquee tool activation button */}
        <button
          className={classNames(
            selectionState.type !== 'off' ? 'bg-primary text-light' : 'cursor-pointer',
          )}
          disabled={selectionState.type !== 'off'}
          onClick={() => {
            if (selectionState.type === 'off') {
              setSelectionState({
                type: 'idle',
              });
              setActiveTool('marquee');
            }
          }}
        >
          <PiSelectionBold />
        </button>{' '}
      </div>
    </>
  );
};
