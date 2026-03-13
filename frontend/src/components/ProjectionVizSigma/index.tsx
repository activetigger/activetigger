import { ControlsContainer, SigmaContainer, ZoomControl } from '@react-sigma/core';
import '@react-sigma/core/lib/style.css';
import classNames from 'classnames';
import Graph from 'graphology';
import { FC, useCallback, useMemo, useState } from 'react';
import { PiSelectionSlashBold } from 'react-icons/pi';
import { Settings } from 'sigma/settings';
import { NodeDisplayData } from 'sigma/types';
import { COLORS } from '../../core/colors';
import { ProjectionOutModel } from '../../types';
import { Caption } from './Caption';
import GraphEvents from './GraphEvents';
import { MarqueBoundingBox, MarqueeController } from './MarqueeController';
import { MarqueeDisplay } from './MarqueeDisplay';

interface Props {
  data: ProjectionOutModel;
  // bbox
  // frameBbox?: MarqueBoundingBox;
  frame?: number[];
  setFrameBbox: (bbox?: MarqueBoundingBox) => void;
  // selection
  selectedId?: string;
  setSelectedId: (id?: string) => void;
  // color
  labelColorMapping: { [key: string]: string };
}

const sigmaStyle = { height: '100%', width: '100%' };

export type SigmaCursorTypes = 'crosshair' | 'pointer' | 'grabbing' | undefined;
export type SigmaToolsType = 'panZoom' | 'marquee';
interface NodeAttributesType {
  node_id: string;
  x: number;
  y: number;
  // Explanation: we want the label displayed to be the node id and not
  // the label as in the annotation
  label: string;
  true_label: string;
  size?: number;
}

// function to quantify point size
const getPointSize = (n: number) => {
  if (n <= 100) {
    return 8;
  } else if (n <= 500) {
    return 5;
  } else if (n <= 1000) {
    return 3;
  } else if (n <= 5000) {
    return 2;
  } else {
    return 1;
  }
};

// Create the Component that listen to all events
export const ProjectionVizSigma: FC<Props> = ({
  data,
  // get/set frame from/to app state
  frame,
  setFrameBbox,
  // manage node selection
  selectedId,
  setSelectedId,
  // color dictionary
  labelColorMapping,
}) => {
  // internal bbox used by marquee. This state will be updated with setFrameBbox once drawing is done.
  // app state is used as default value
  // transform frame type to bbox type
  const frameBbox = useMemo(
    () =>
      frame
        ? {
            x: { min: frame[0], max: frame[1] },
            y: { min: frame[2], max: frame[3] },
          }
        : undefined,
    [frame],
  );

  const [bbox, setBbox] = useState<MarqueBoundingBox | undefined>(frameBbox);
  //// Action if double click
  const [clusterHighlight, setClusterHighlight] = useState<string | undefined>(undefined);
  const setClusterHighlightAfterDoubleClick = useCallback(
    (id?: string) => {
      if (id && data) {
        const selected_node = data.nodes.find((o) => o.node_id === id);
        if (selected_node) {
          setClusterHighlight(selected_node.label.toString());
          return;
        }
      }
      setClusterHighlight(undefined);
    },
    [data],
  );

  labelColorMapping['NA'] = COLORS.NA;

  // Special cursor to help interactivity affordances
  const [sigmaCursor, setSigmaCursor] = useState<SigmaCursorTypes>(undefined);
  const [activeTool, setActiveTool] = useState<SigmaToolsType>('panZoom');

  // column to use for color mapping
  const [selectedColumn, setSelectedColumn] = useState<'labels' | 'predictions'>('labels');

  // prepare graph for sigma from data props
  const graph = useMemo(() => {
    console.log('compute graph');
    const graph = new Graph<NodeAttributesType>();
    if (data) {
      //TODO: refine those quick heuristics
      const size = getPointSize(data.nodes.length);
      data.nodes.forEach((node) => {
        graph.addNode(node.node_id, {
          node_id: node.node_id,
          x: node.x,
          y: node.y,
          label: node.node_id,
          true_label: node.label,
          size,
        });
      });
      return graph;
    }
    return undefined;
  }, [data]);

  // nodeReducer change node appearance from colorMapping and selection state
  const nodeReducer = useCallback(
    (_node: string, node: NodeAttributesType): Partial<NodeDisplayData> => {
      const res: Partial<NodeDisplayData> = { ...node };

      // apply color for nodes
      if (clusterHighlight) {
        if (clusterHighlight === node.true_label.toString()) {
          res.color = labelColorMapping[node.true_label];
        } else {
          res.color = labelColorMapping['NA'];
        }
      } else {
        res.color = labelColorMapping[node.true_label];
      }

      // only the label is displayed, so we set up the label as the node id
      res.label = node.node_id;

      if (selectedId === node.node_id) {
        // built-in appearance in Sigma which forces showing the label
        res.highlighted = true;
        res.color = 'black'; // highlight color
      }
      return res;
    },
    [selectedId, labelColorMapping, clusterHighlight],
  );
  const settings: Partial<Settings<NodeAttributesType>> = useMemo(
    () => ({
      // Settings from sigma
      allowInvalidContainer: true,
      nodeReducer,
    }),
    [nodeReducer],
  );

  return (
    <div>
      {/* // Weird euristic but if there are predictions, they will be as strings */}
      {data.nodes.length > 0 && data.nodes[0].predictions && (
        <>
          <label>Color by: </label>
          <select
            value={selectedColumn}
            onChange={(event) => {
              setSelectedColumn(event.target.value as 'labels' | 'predictions');
            }}
          >
            <option value="labels">Annotated elements</option>
            <option value="predictions">Predicted elements for {data.active_model?.value}</option>
          </select>
        </>
      )}
      <div
        style={{
          width: '100%',
          height: 'clamp(300px, 75vh, 1000px)',
        }}
      >
        <SigmaContainer
          className={classNames(
            sigmaCursor ? `cursor-${sigmaCursor}` : activeTool === 'marquee' && 'cursor-crosshair',
          )}
          style={sigmaStyle}
          graph={graph}
          settings={settings}
        >
          <GraphEvents
            setSelectedIdAfterClick={setSelectedId}
            setSigmaCursor={setSigmaCursor}
            setClusterHighlightAfterDoubleClick={setClusterHighlightAfterDoubleClick}
          />
          {Object.keys(labelColorMapping).length < 15 && (
            <ControlsContainer position="top-left">
              <Caption labelColorMapping={labelColorMapping} />
            </ControlsContainer>
          )}
          <ControlsContainer position={'top-right'}>
            <div className="border-bottom">
              {/* Active tools (zoom-pan or marquee)) buttons are managed by the marquee controller */}
              <MarqueeController
                setBbox={setBbox}
                validateBoundingBox={setFrameBbox}
                setActiveTool={setActiveTool}
              />
            </div>
            <ZoomControl />
            {/* delete bbox button */}
            {bbox !== undefined && (
              <div className="react-sigma-control">
                <button
                  onClick={() => {
                    setBbox(undefined);
                    setFrameBbox(undefined);
                  }}
                >
                  <PiSelectionSlashBold />
                </button>
              </div>
            )}
          </ControlsContainer>
          {/* show a dashed line rectangle to render the current bbox */}
          <MarqueeDisplay bbox={bbox} />
        </SigmaContainer>
      </div>
    </div>
  );
};
