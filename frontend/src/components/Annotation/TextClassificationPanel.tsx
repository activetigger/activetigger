import { motion } from 'framer-motion';
import Highlighter from 'react-highlight-words';
import { DisplayConfig, ElementOutModel } from '../../types';

import { FC, LegacyRef } from 'react';
import { Tooltip } from 'react-tooltip';
import { CSSProperties } from 'styled-components';
import { AnnotationIcon, NoAnnotationIcon, UserIcon } from '../Icons';

interface ClassificationPanelProps {
  element: ElementOutModel | undefined;
  displayConfig: DisplayConfig;
  textInFrame: string;
  textOutFrame: string;
  validHighlightText: string[];
  elementId: string;
  lastTag: string;
  phase: string;
  frameRef: HTMLDivElement;
}

export const TextClassificationPanel: FC<ClassificationPanelProps> = ({
  element,
  displayConfig,
  textInFrame,
  textOutFrame,
  validHighlightText,
  elementId,

  phase,
  frameRef,
}) => {
  return (
    <>
      <div
        className="annotation-frame"
        style={
          {
            '--height': `${displayConfig.textFrameHeight}vh`,
          } as CSSProperties
        }
        ref={frameRef as unknown as LegacyRef<HTMLDivElement>}
      >
        {element?.history && element.history[0] && element.history[0].label && (
          <span className="position-absolute end-0 top-0 me-1">
            <AnnotationIcon title={element.history[0].label} />
          </span>
        )}

        <p className="element-text">
          <motion.span
            animate={elementId ? { backgroundColor: ['#ea6b1f70', '#ffffff00'] } : {}}
            transition={{ duration: 1 }}
          >
            <Highlighter
              highlightClassName="Search"
              searchWords={validHighlightText}
              autoEscape={false}
              textToHighlight={textInFrame}
              highlightStyle={{
                backgroundColor: 'yellow',
                margin: '0px',
                padding: '2px 1px',
              }}
              caseSensitive={true}
            />
            {/* text out of frame */}
            <span className="text-out-context" title="Outside context window ">
              <Highlighter
                highlightClassName="Search"
                searchWords={validHighlightText}
                autoEscape={false}
                textToHighlight={textOutFrame}
                highlightStyle={{
                  backgroundColor: 'yellow',
                  margin: '0px',
                  padding: '2px 1px',
                }}
                caseSensitive={true}
              />
            </span>
            <Tooltip anchorSelect=".text-out-context" place="bottom" style={{ zIndex: 99 }}>
              Outside context window.
              <br />
              Go to config menu to change context window according to the model you plan to use
            </Tooltip>
          </motion.span>
        </p>

        <div className="footer">
          {
            //display context if available
            phase != 'test' &&
              displayConfig.displayContext &&
              Object.entries(element?.context || {}).length > 0 && (
                <details>
                  <summary>Context </summary>
                  <dl>
                    {Object.entries(element?.context || { None: 'None' }).map(([k, v]) => {
                      return (
                        <div className="dl-item" key={k}>
                          <dt>{k}</dt>
                          <dd>{v as string}</dd>
                        </div>
                      );
                    })}
                  </dl>
                </details>
              )
          }
          {
            //display history
            phase != 'test' && displayConfig.displayElementHistory && (
              <details>
                <summary>History</summary>
                <div>
                  {(element?.history || []).map((h, i) => {
                    return (
                      <div className="badge" key={i} title={h.comment || ''}>
                        {h.label ? (
                          <span className="me-1">
                            <AnnotationIcon />
                            {h.label}
                          </span>
                        ) : (
                          <NoAnnotationIcon className="me-1" />
                        )}{' '}
                        <span>
                          <UserIcon /> {h.user}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </details>
            )
          }
        </div>
      </div>
    </>
  );
};
