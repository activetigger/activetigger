import chroma from 'chroma-js';
import cx from 'classnames';
import { motion } from 'framer-motion';
import { CSSProperties, FC, useCallback, useEffect, useMemo, useState } from 'react';
import { FaCheck, FaForward } from 'react-icons/fa';
import { MdClear } from 'react-icons/md';
import { useNavigate, useParams } from 'react-router-dom';
import { AnnotateBlendTag, TextAnnotateBlend } from 'react-text-annotate-blend';

import { useAnnotationSessionHistory } from '../../core/useHistory';
import { reorderLabels } from '../../core/utils';
import { DisplayConfig, ElementOutModel } from '../../types';

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
  const { projectName } = useParams();
  const navigate = useNavigate();
  const { addElementInAnnotationSessionHistory } = useAnnotationSessionHistory();

  const reorderedLabels = useMemo(
    () => reorderLabels(labels || [], displayConfig.labelsOrder || []),
    [displayConfig.labelsOrder, labels],
  );

  const mode = displayConfig.spanAnnotationMode || 'locked';

  const [value, setValue] = useState<AnnotateBlendTag[]>([]);
  const [tag, setTag] = useState<string | null>(reorderedLabels[0] || null);
  const [comment, setComment] = useState<string>(
    element?.history ? element.history[0]?.comment || '' : '',
  );

  const UNTAGGED = '__untagged__';
  const UNTAGGED_COLOR = '#d1d5db';

  useEffect(() => setComment(element?.history ? element.history[0]?.comment || '' : ''), [element]);

  useEffect(() => {
    if (lastTag) {
      setValue(JSON.parse(lastTag));
    } else {
      setValue([]);
    }
  }, [lastTag]);

  useEffect(() => {
    if (mode === 'neutral') {
      setTag(null);
    } else if (!tag) {
      setTag(reorderedLabels[0] || null);
    }
  }, [mode, reorderedLabels, tag]);

  const handleChange = (value: AnnotateBlendTag[]) => {
    setValue(value);
  };

  const skipAnnotation = useCallback(() => {
    if (element)
      addElementInAnnotationSessionHistory(
        element.element_id,
        element.text,
        undefined,
        undefined,
        true,
      );
    navigate(`/projects/${projectName}/tag/`);
  }, [navigate, projectName, addElementInAnnotationSessionHistory, element]);

  const validateAnnotation = useCallback(() => {
    const cleaned = value.filter((s) => s.tag !== UNTAGGED);
    postAnnotation(JSON.stringify(cleaned) || JSON.stringify([]), elementId, comment);
    setValue([]);
  }, [postAnnotation, value, elementId, comment]);

  useEffect(() => {
    const handler = (ev: KeyboardEvent) => {
      const activeElement = document.activeElement;
      const isFormField =
        activeElement?.tagName === 'INPUT' ||
        activeElement?.tagName === 'TEXTAREA' ||
        activeElement?.tagName === 'SELECT';
      if (isFormField) return;
      if (ev.key === 'Enter') {
        ev.preventDefault();
        validateAnnotation();
      }
      if (ev.key === 'ArrowRight') {
        ev.preventDefault();
        skipAnnotation();
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [validateAnnotation, skipAnnotation]);

  const colormap = chroma.scale('Paired').colors(reorderedLabels.length);
  const COLORS = Object.fromEntries(
    reorderedLabels.map((label, index) => [label, colormap[index]]),
  );
  const options = reorderedLabels.map((label) => ({
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
              if (mode === 'neutral') {
                return { ...span, tag: UNTAGGED, color: UNTAGGED_COLOR };
              }
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
      <div className="tag-action-container">
        {options.map((opt) => {
          const isActive = mode === 'locked' && opt.value === tag;
          const handleClick = () => {
            if (mode === 'neutral') {
              setValue((prev) =>
                prev.map((s) =>
                  s.tag === UNTAGGED ? { ...s, tag: opt.value, color: opt.color } : s,
                ),
              );
            } else {
              setTag(opt.value);
            }
          };
          return (
            <button
              key={opt.value}
              type="button"
              onClick={handleClick}
              className={cx('span-annotation-label-selector', isActive && 'active')}
              style={{ '--label-color': opt.color } as CSSProperties}
            >
              {opt.label}
            </button>
          );
        })}
        <button
          type="button"
          className="btn-annotate-general-action tag-action-button span-annotation-validate"
          onClick={validateAnnotation}
        >
          <FaCheck size={16} /> Validate
          <span className="badge hotkey">⏎</span>
        </button>
        <button
          type="button"
          className="btn-annotate-general-action tag-action-button"
          onClick={() => setValue([])}
        >
          <MdClear size={16} /> Clear annotations
        </button>
        <button
          type="button"
          className="btn-annotate-general-action tag-action-button"
          onClick={() => skipAnnotation()}
        >
          <FaForward size={16} /> Skip
          <span className="badge hotkey">→</span>
        </button>
        <textarea
          className="form-control annotation-comment"
          placeholder="Comment"
          value={comment}
          onChange={(e) => setComment(e.target.value)}
        />
      </div>
    </>
  );
};
