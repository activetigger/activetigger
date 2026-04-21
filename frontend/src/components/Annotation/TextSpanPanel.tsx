import cx from 'classnames';
import chroma from 'chroma-js';
import { motion } from 'framer-motion';
import { FC, useEffect, useState } from 'react';
import { FaCheck } from 'react-icons/fa';
import { AnnotateBlendTag, TextAnnotateBlend } from 'react-text-annotate-blend';
import { DisplayConfig, ElementOutModel } from '../../types';
import { CSSProperties } from 'styled-components';

interface SpanInputProps {
  elementId: string;
  displayConfig: DisplayConfig;
  text: string;
  labels: string[];
  postAnnotation: (label: string, elementId: string, comment?: string) => void;
  lastTag?: string;
  element?: ElementOutModel;
}

export const TextSpanPanel: FC<SpanInputProps> = ({
  elementId,
  displayConfig,
  text,
  postAnnotation,
  labels,
  lastTag,
  element,
}) => {
  // get the context and set the labels

  const [value, setValue] = useState<AnnotateBlendTag[]>([]);
  const [tag, setTag] = useState<string | null>(labels[0] || null);
  const [comment, setComment] = useState<string>(
    element?.history ? element.history[0]?.comment || '' : '',
  );

  useEffect(() => setComment(element?.history ? element.history[0]?.comment || '' : ''), [element]);

  useEffect(() => {
    if (lastTag) {
      setValue(JSON.parse(lastTag));
    } else {
      setValue([]);
    }
  }, [lastTag]);

  const handleChange = (value: AnnotateBlendTag[]) => {
    setValue(value);
  };

  const colormap = chroma.scale('Paired').colors(labels.length);
  const COLORS = Object.fromEntries(labels.map((label, index) => [label, colormap[index]]));
  const options = labels.map((label) => ({
    value: label,
    label: label,
    color: COLORS[label],
  }));

  return (
    <>
      <div className="annotation-frame" style={{ height: `${displayConfig.textFrameHeight}vh` }}>
        <motion.div
          animate={elementId ? { backgroundColor: ['#e8e9ff', '#f9f9f9'] } : {}}
          transition={{ duration: 1 }}
        >
          <TextAnnotateBlend
            style={{
              fontSize: '1.2rem',
            }}
            content={text || ''}
            onChange={handleChange}
            value={value || []}
            getSpan={(span) => {
              if (!tag) return span;
              return {
                ...span,
                tag: tag,
                color: tag && COLORS[tag],
              };
            }}
          />
        </motion.div>
      </div>
      <div>
        <div className="d-flex flex-column gap-2 align-items-center mt-2">
          {options.map((opt) => {
            const isActive = opt.value === tag;

            return (
              <button
                key={opt.value}
                type="button"
                onClick={() => setTag(opt.value)}
                className={cx('span-annotation-label-selector ', isActive ? 'active' : '')}
                style={{ '--label-color': opt.color } as CSSProperties}
              >
                {opt.label}
              </button>
            );
          })}
          <button
            className="span-annotation-label-selector"
            onClick={() => {
              postAnnotation(JSON.stringify(value) || JSON.stringify([]), elementId, comment);
              setValue([]);
            }}
            style={{ '--label-color': 'green' } as CSSProperties}
          >
            <FaCheck size={18} /> Validate the annotation
          </button>
          <textarea
            className="form-control annotation-comment"
            placeholder="Comment"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
          />
        </div>
      </div>
    </>
  );
};
