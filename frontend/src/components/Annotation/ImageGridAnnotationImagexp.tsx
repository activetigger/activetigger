// Experimental image projects — batch grid annotation panel.
// Displays an N x N grid of image thumbnails: the user selects images
// (click), applies a label to the whole selection, then validates the
// batch. Tagged elements are posted in one call (/annotation/table) and
// untagged elements are skipped (recorded in the session history so they
// are not served again during this session).
// Activated from the annotation Configuration modal (image projects in
// experimental mode only). Designed for computer screens.
import chroma from 'chroma-js';
import classNames from 'classnames';
import { CSSProperties, FC, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { LuRefreshCw } from 'react-icons/lu';
import { useAddTableAnnotations, useGetNextElementsBatch } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { useAppContext } from '../../core/useAppContext';
import { useAnnotationSessionHistory } from '../../core/useHistory';
import { reorderLabels } from '../../core/utils';
import { AnnotationModel, ElementOutModel, SelectionConfig } from '../../types';
import { ImageThumbnailImagexp } from '../ImageThumbnailImagexp';
import { MiddleEllipsis } from './MiddleEllipsis';

interface ImageGridAnnotationImagexpProps {
  projectSlug: string;
  scheme: string;
  selectionConfig: SelectionConfig;
  phase: string;
  availableLabels: string[];
  gridSize: number;
  // called after a batch has been validated (e.g. to refresh statistics)
  onValidated?: () => void;
}

// the grid is designed for computer screens (bootstrap lg breakpoint)
const MIN_SCREEN_WIDTH = 992;

export const ImageGridAnnotationImagexp: FC<ImageGridAnnotationImagexpProps> = ({
  projectSlug,
  scheme,
  selectionConfig,
  phase,
  availableLabels,
  gridSize,
  onValidated,
}) => {
  const { notify } = useNotifications();
  const {
    appContext: { history, currentProjectionName, activeModel, displayConfig },
  } = useAppContext();
  const { addElementInAnnotationSessionHistory } = useAnnotationSessionHistory();

  // grid side is bounded to keep the batch size (n * n) reasonable
  const n = Math.min(Math.max(Math.round(gridSize) || 3, 2), 5);

  const historyIds = useMemo(() => history.map((h) => h.element_id), [history]);
  const { getNextElementsBatch } = useGetNextElementsBatch(
    projectSlug,
    scheme,
    currentProjectionName || null,
    selectionConfig,
    historyIds,
    phase,
    activeModel || null,
  );
  const { addTableAnnotations } = useAddTableAnnotations(projectSlug, scheme, phase);

  // screen-size guard (live: reacts to window resizes)
  const [isWideScreen, setIsWideScreen] = useState<boolean>(
    () => window.matchMedia(`(min-width: ${MIN_SCREEN_WIDTH}px)`).matches,
  );
  useEffect(() => {
    const mq = window.matchMedia(`(min-width: ${MIN_SCREEN_WIDTH}px)`);
    const onChange = (ev: MediaQueryListEvent) => setIsWideScreen(ev.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  const [elements, setElements] = useState<ElementOutModel[]>([]);
  // labels assigned in the current batch, keyed by element id
  const [pending, setPending] = useState<Record<string, string>>({});
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState<boolean>(true);
  // bumped after each validation to trigger the next batch fetch
  const [batchKey, setBatchKey] = useState<number>(0);

  // keep the latest fetcher in a ref so the load effect only refires on
  // explicit triggers (validation, selection change), not on every session
  // history update
  const fetchRef = useRef(getNextElementsBatch);
  fetchRef.current = getNextElementsBatch;
  const requestId = useRef(0);

  useEffect(() => {
    const id = ++requestId.current;
    setLoading(true);
    fetchRef.current(n * n).then((res) => {
      if (id !== requestId.current) return; // drop stale responses
      setElements(res || []);
      setPending({});
      setSelected(new Set());
      setLoading(false);
    });
    // refetch on validation (batchKey) or when the selection setup changes
  }, [batchKey, n, projectSlug, scheme, phase, selectionConfig]);

  const orderedLabels = useMemo(
    () => reorderLabels(availableLabels || [], displayConfig.labelsOrder || []),
    [availableLabels, displayConfig.labelsOrder],
  );

  // one stable color per label (same palette as the span annotation panel),
  // used for both the label buttons and the capsules under the images
  const labelColors = useMemo(() => {
    const colormap = chroma.scale('Paired').colors(orderedLabels.length);
    return Object.fromEntries(orderedLabels.map((label, index) => [label, colormap[index]]));
  }, [orderedLabels]);
  const labelTextColor = (label: string) =>
    labelColors[label] && chroma(labelColors[label]).luminance() > 0.5 ? '#333' : 'white';

  const toggleSelected = useCallback((elementId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(elementId)) next.delete(elementId);
      else next.add(elementId);
      return next;
    });
  }, []);

  const selectUnannotated = useCallback(() => {
    setSelected(
      new Set(
        elements.filter((el) => pending[el.element_id] === undefined).map((el) => el.element_id),
      ),
    );
  }, [elements, pending]);

  const applyLabel = useCallback(
    (label: string) => {
      if (selected.size === 0) {
        notify({ type: 'warning', message: 'Select at least one image first.' });
        return;
      }
      setPending((prev) => {
        const next = { ...prev };
        selected.forEach((elementId) => {
          next[elementId] = label;
        });
        return next;
      });
      setSelected(new Set());
    },
    [selected, notify],
  );

  const validate = useCallback(async () => {
    if (loading || elements.length === 0) return;
    const tagged = elements.filter((el) => pending[el.element_id] !== undefined);
    const skipped = elements.filter((el) => pending[el.element_id] === undefined);
    if (tagged.length > 0) {
      const annotations: AnnotationModel[] = tagged.map((el) => ({
        project_slug: projectSlug,
        dataset: phase,
        scheme: scheme,
        element_id: el.element_id,
        label: pending[el.element_id],
        selection: 'grid',
      }));
      const ok = await addTableAnnotations(annotations);
      if (!ok) return;
      tagged.forEach((el) =>
        addElementInAnnotationSessionHistory(el.element_id, el.text, pending[el.element_id]),
      );
    }
    // untagged elements are skipped: record them in the session history so
    // they are not served again during this session
    skipped.forEach((el) =>
      addElementInAnnotationSessionHistory(el.element_id, el.text, undefined, undefined, true),
    );
    setBatchKey((k) => k + 1);
    // refresh the parent statistics once the annotations are saved
    onValidated?.();
  }, [
    loading,
    elements,
    pending,
    projectSlug,
    phase,
    scheme,
    addTableAnnotations,
    addElementInAnnotationSessionHistory,
    onValidated,
  ]);

  // one document-level keyboard handler for the whole grid (digits apply a
  // label to the selection, U selects unannotated, Enter validates)
  const handleKeyboardEvents = useCallback(
    (ev: KeyboardEvent) => {
      const activeElement = document.activeElement;
      const isFormField =
        activeElement?.tagName === 'INPUT' ||
        activeElement?.tagName === 'TEXTAREA' ||
        activeElement?.tagName === 'SELECT';
      if (isFormField) return;
      if (ev.code === 'Enter' || ev.code === 'NumpadEnter') {
        validate();
        return;
      }
      if (ev.code === 'KeyU') {
        selectUnannotated();
        return;
      }
      if (orderedLabels.length < 10)
        orderedLabels.forEach((label, i) => {
          if (ev.code === `Digit${i + 1}` || ev.code === `Numpad${i + 1}`) applyLabel(label);
        });
    },
    [orderedLabels, applyLabel, selectUnannotated, validate],
  );
  useEffect(() => {
    document.addEventListener('keydown', handleKeyboardEvents);
    return () => document.removeEventListener('keydown', handleKeyboardEvents);
  }, [handleKeyboardEvents]);

  if (!isWideScreen)
    return (
      <div className="alert alert-warning my-3">
        The grid annotation mode is designed for computer screens: the current window is too small.
        Enlarge the window or disable the grid display in the configuration panel.
      </div>
    );

  const taggedCount = elements.filter((el) => pending[el.element_id] !== undefined).length;
  const skippedCount = elements.length - taggedCount;

  return (
    <div className="image-grid-annotation" style={{ '--grid-n': n } as CSSProperties}>
      <div className="image-grid-frame">
        {loading && <div className="image-grid-status">Loading images…</div>}
        {!loading && elements.length === 0 && (
          <div className="image-grid-status">
            No element available with this selection mode.
            <button className="btn-primary-action" onClick={() => setBatchKey((k) => k + 1)}>
              <LuRefreshCw size={20} /> Retry
            </button>
          </div>
        )}
        {!loading &&
          elements.map((el) => {
            const pendingLabel = pending[el.element_id];
            const previousLabel =
              el.history && el.history.length > 0 ? el.history[0].label : undefined;
            const shownLabel = pendingLabel ?? previousLabel ?? undefined;
            return (
              <div
                key={el.element_id}
                className={classNames('image-grid-cell', selected.has(el.element_id) && 'selected')}
                onClick={() => toggleSelected(el.element_id)}
                title={el.element_id}
              >
                <ImageThumbnailImagexp
                  projectSlug={projectSlug}
                  elementId={el.element_id}
                  style={{ maxWidth: '100%', maxHeight: '100%', width: '100%' }}
                />
                <span
                  className={classNames('image-grid-label', shownLabel === undefined && 'untagged')}
                  style={
                    shownLabel !== undefined && labelColors[shownLabel]
                      ? {
                          backgroundColor: labelColors[shownLabel],
                          color: labelTextColor(shownLabel),
                        }
                      : undefined
                  }
                >
                  {shownLabel ?? '—'}
                </span>
              </div>
            );
          })}
      </div>
      <div className="image-grid-side">
        <span className="image-grid-count">
          {selected.size} selected — {taggedCount}/{elements.length} tagged
        </span>
        <div className="tag-action-container">
          {orderedLabels.map((label, i) => (
            <button
              type="button"
              key={label}
              className="tag-action-button btn-annotate-action"
              style={{
                backgroundColor: labelColors[label],
                borderColor: labelColors[label],
                color: labelTextColor(label),
              }}
              onClick={() => applyLabel(label)}
            >
              <MiddleEllipsis label={label} forceComplete={displayConfig.forceCompleteLabel} />
              {orderedLabels.length < 10 && <span className="badge hotkey">{i + 1}</span>}
            </button>
          ))}
        </div>
        <div className="tag-action-container image-grid-batch-actions">
          <button
            type="button"
            className="btn-annotate-general-action tag-action-button"
            onClick={selectUnannotated}
            disabled={loading || elements.length === 0}
            title="Select all images without a label in the current batch"
          >
            Select unannotated <span className="badge hotkey">U</span>
          </button>
          <button
            type="button"
            className="btn-annotate-general-action tag-action-button span-annotation-validate"
            onClick={validate}
            disabled={loading || elements.length === 0}
            title={`Save the ${taggedCount} tagged image(s) and skip the ${skippedCount} other(s)`}
          >
            Validate <span className="badge hotkey">⏎</span>
          </button>
        </div>
      </div>
    </div>
  );
};
