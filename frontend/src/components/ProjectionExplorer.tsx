import classNames from 'classnames';
import { FC, ReactNode, useCallback, useEffect, useState } from 'react';

import { FaLock } from 'react-icons/fa';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { useNavigate } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import { useGetElementById } from '../core/api';
import { useAppContext } from '../core/useAppContext';
import { ElementOutModel, ProjectionOutModel } from '../types';
import { ProjectionVizSigma } from './ProjectionVizSigma';
import { MarqueBoundingBox } from './ProjectionVizSigma/MarqueeController';

interface ProjectionExplorerProps {
  projectName: string | null;
  data: ProjectionOutModel;
  selectedId?: string;
  labelColorMapping: Record<string, string>;
  containerClassName?: string;
  vizClassName?: string;
  panelClassName?: string;
  children?: (element: ElementOutModel, clearSelection: () => void) => ReactNode;
}

/**
 * Shared wrapper around ProjectionVizSigma that handles:
 * - element selection (click a node -> fetch element -> display panel)
 * - frame bbox selection (marquee -> update selectionConfig in context)
 * - "lock on selection" checkbox
 * - selected element display (text, history, navigate link)
 *
 * Extra content for the selected element panel (e.g. annotation form) can be
 * injected via the `children` render prop.
 */
export const ProjectionExplorer: FC<ProjectionExplorerProps> = ({
  projectName,
  data,
  selectedId: externalSelectedId,
  labelColorMapping,
  containerClassName,
  vizClassName,
  panelClassName,
  children,
}) => {
  const {
    appContext: { selectionConfig },
    setAppContext,
  } = useAppContext();
  const navigate = useNavigate();

  // element selection state
  const { getElementById } = useGetElementById();
  const [selectedElement, setSelectedElement] = useState<ElementOutModel | null>(null);

  const setSelectedId = useCallback(
    (id?: string) => {
      if (id)
        getElementById(id, 'train').then((element) => {
          setSelectedElement(element || null);
        });
      else setSelectedElement(null);
    },
    [getElementById],
  );

  const clearSelection = useCallback(() => {
    setSelectedElement(null);
  }, []);

  // sync with external selection (e.g. currentElement from parent)
  useEffect(() => {
    if (externalSelectedId) {
      setSelectedId(externalSelectedId);
    }
  }, [externalSelectedId, setSelectedId]);

  // frame bbox callback
  const setFrameBbox = useCallback(
    (bbox?: MarqueBoundingBox) => {
      setAppContext((prev) => ({
        ...prev,
        selectionConfig: {
          ...prev.selectionConfig,
          frame: bbox ? [bbox.x.min, bbox.x.max, bbox.y.min, bbox.y.max] : undefined,
        },
      }));
    },
    [setAppContext],
  );

  return (
    <>
      {(selectionConfig.frame || []).length > 0 && (
        <div className="my-2">
          <label style={{ display: 'block' }}>
            <input
              type="checkbox"
              checked={selectionConfig.frameSelection}
              onChange={() => {
                setAppContext((prev) => ({
                  ...prev,
                  selectionConfig: {
                    ...prev.selectionConfig,
                    frameSelection: !prev.selectionConfig.frameSelection,
                  },
                }));
              }}
            />
            <span className="lock">
              <FaLock /> Lock on selection
            </span>
            <a className="lockhelp">
              <HiOutlineQuestionMarkCircle />
            </a>
            <Tooltip anchorSelect=".lockhelp" place="top">
              Once a vizualisation computed, you can use the square tool to select an area (or
              remove the square).<br></br> Then you can lock the selection, and only elements in the
              selected area will be available for annoation.
            </Tooltip>
          </label>
        </div>
      )}

      <div className={containerClassName || 'd-flex flex-column'}>
        <div className={vizClassName}>
          <ProjectionVizSigma
            data={data}
            selectedId={selectedElement?.element_id || externalSelectedId}
            setSelectedId={setSelectedId}
            frame={selectionConfig.frame}
            setFrameBbox={setFrameBbox}
            labelColorMapping={labelColorMapping}
          />
        </div>

        <div className={classNames(panelClassName, selectedElement && 'active')}>
          {selectedElement ? (
            <div
              style={{
                overflowY: 'auto',
                maxHeight: '80vh',
              }}
              className="mx-4"
            >
              <a
                className="badge m-0 p-1"
                onClick={() =>
                  navigate(`/projects/${projectName}/tag/${selectedElement.element_id}?tab=tag`)
                }
                style={{ cursor: 'pointer' }}
              >
                Text {selectedElement.element_id}
              </a>
              <div>{selectedElement.text}</div>
              <details>
                <summary>Previous annotations:</summary>
                <ul>
                  {selectedElement.history?.map((e) => {
                    return (
                      <li key={`${e.time}-${e.user}`}>
                        label: {e.label ? e.label : 'label removed'} ({e.time} by {e.user})
                        <br />
                      </li>
                    );
                  })}
                </ul>
              </details>
              {children && children(selectedElement, clearSelection)}
            </div>
          ) : (
            <div className="explanations horizontal center" style={{ flex: '1 1 auto' }}>
              Click on an element to display its content
            </div>
          )}
        </div>
      </div>
    </>
  );
};
