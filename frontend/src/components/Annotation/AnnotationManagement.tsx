import { CSSProperties, FC, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FaBookOpen } from 'react-icons/fa';
import { HiOutlineEyeOff } from 'react-icons/hi';
import { LuRefreshCw } from 'react-icons/lu';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import {
  useAddAnnotation,
  useGetElementById,
  useGetNextElementId,
  useGetSchemeCodebook,
  useStatistics,
} from '../../core/api';
import { useAppContext } from '../../core/useAppContext';
import { ElementOutModel, PromptsProjectStateModel } from '../../types';

import MDEditor from '@uiw/react-md-editor';
import classNames from 'classnames';
import { Modal } from 'react-bootstrap';
import { useNotifications } from '../../core/notifications';
import { useAnnotationSessionHistory } from '../../core/useHistory';
import { isValidRegex } from '../../core/utils';
import { TagDisplayParameters } from '../TagDisplayParameters';
import { DisplayProjection } from '../vizualisation/DisplayProjection';
import { AnnotationHistoryList } from './AnnotationHistoryList';
import { AnnotationModeForm } from './AnnotationMode';
import { ImageClassificationPanelImagexp } from './ImageClassificationPanelImagexp';
import { MulticlassInput } from './MulticlassInput';
import { MultilabelInput } from './MultilabelInput';
import { PromptsPanel } from './PromptsPanel';
import { SelectActiveLearning } from './SelectActiveLearning';
import { TextClassificationPanel } from './TextClassificationPanel';
import { TextSpanPanel } from './TextSpanPanel';

export const AnnotationManagement: FC = () => {
  const { notify } = useNotifications();
  const { projectName, elementId } = useParams();
  const { appContext, setAppContext } = useAppContext();

  const {
    currentScheme,
    currentProject: project,
    selectionConfig,
    displayConfig,
    activeModel,
    history,
    selectionHistory,
    phase,
    currentProjectionName,
  } = appContext;

  const navigate = useNavigate();
  const location = useLocation();

  // Use dataset query param directly to avoid race condition with context update
  const datasetParam = new URLSearchParams(location.search).get('dataset');
  const effectivePhase =
    datasetParam && ['train', 'test', 'valid'].includes(datasetParam) ? datasetParam : phase;
  useEffect(() => {
    if (datasetParam && ['train', 'test', 'valid'].includes(datasetParam)) {
      setAppContext((prev) => ({ ...prev, phase: datasetParam }));
    }
  }, [datasetParam, setAppContext]);

  const [element, setElement] = useState<ElementOutModel | null>(null); //state for the current element
  const [nSample, setNSample] = useState<number | null>(null); // specific info

  // timestamp at which the current element was displayed, used to compute
  // mean annotation duration in the status notch.
  const elementDisplayedAtRef = useRef<{ id: string; at: number } | null>(null);

  // Metadata returned by /elements/next (selection mode, prompt similarity/rank)
  // is keyed by element_id so it survives the navigation + getElementById round-
  // trip and gets merged onto the displayed element. Stored in a ref because
  // updates here should not trigger re-renders.
  const nextSelectionMeta = useRef<
    Record<
      string,
      {
        selection?: string | null;
        similarity?: number | null;
        rank?: number | null;
      }
    >
  >({});

  const [showDisplayConfig, setShowDisplayConfig] = useState<boolean>(false);
  const [showDisplayViz, setShowDisplayViz] = useState<boolean>(false);
  // focus mode: render the annotation block alone in a fullscreen modal.
  // Deliberately not persisted so a reload always comes back to the normal page.
  const [showFocusMode, setShowFocusMode] = useState<boolean>(false);
  const [showCodebook, setShowCodebook] = useState<boolean>(false);
  const [showPromptsModal, setShowPromptsModal] = useState<boolean>(false);
  const { codebook } = useGetSchemeCodebook(projectName || null, currentScheme || null);
  const [selectFirstModelTrained, setSelectFirstModelTrained] = useState<boolean>(false);
  const [authorizeRetraining, setAuthorizeRetraining] = useState<boolean>(false);
  const handleCloseViz = () => setShowDisplayViz(false);
  const handleCloseConfig = () => setShowDisplayConfig(false);

  // Reinitialize scroll in frame
  const frameRef = useRef<HTMLDivElement>(null);
  const resetScroll = () => {
    if (frameRef.current) {
      frameRef.current.scrollTop = 0;
    }
  };

  // hooks to manage element
  const historyIds = useMemo(() => history.map((h) => h.element_id), [history]);
  const { getNextElementId } = useGetNextElementId(
    projectName || null,
    currentScheme || null,
    currentProjectionName || null,
    selectionConfig,
    historyIds,
    effectivePhase,
    activeModel || null,
  );
  const { getElementById } = useGetElementById();

  // hooks to manage annotation
  const { addAnnotation } = useAddAnnotation(
    projectName || null,
    currentScheme || null,
    effectivePhase,
  );

  //hook to manage history
  const { addElementInAnnotationSessionHistory } = useAnnotationSessionHistory();

  // define parameters for configuration panels
  const availableLabels =
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].labels
      : [];
  const [kindScheme] = useState<string>(
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].kind || 'multiclass'
      : 'multiclass',
  );

  // get statistics to display
  const { statistics, reFetchStatistics } = useStatistics(
    projectName || null,
    currentScheme || null,
  );

  // react to URL param change
  useEffect(() => {
    resetScroll();
    if (elementId === 'noelement') {
      return;
    }
    if (elementId === undefined) {
      getNextElementId().then((res) => {
        if (res && res.n_sample) setNSample(res.n_sample);
        if (res && res.element_id) {
          // stash the prompt similarity/rank info so we can merge it onto the
          // element after getElementById resolves (that endpoint doesn't carry
          // the selection-time metadata).
          nextSelectionMeta.current[res.element_id] = {
            selection: res.selection,
            similarity: res.similarity,
            rank: res.rank,
          };
          setAppContext((prev) => ({
            ...prev,
            selectionHistory: {
              ...prev.selectionHistory,
              [res.element_id]: JSON.stringify(selectionConfig),
            },
          }));
          // redirect to the next element page replacing history
          navigate(`/projects/${projectName}/tag/${res.element_id}`, { replace: true });
        } else {
          navigate(`/projects/${projectName}/tag/noelement`);
          setElement(null);
        }
      });
    } else {
      // only if id changed compared to the previous one (otherwise, a change in phase would trigger a reload)
      if (element?.element_id !== elementId) {
        getElementById(elementId, effectivePhase)
          .then((fetched) => {
            if (fetched) {
              const meta = nextSelectionMeta.current[fetched.element_id];
              if (meta) {
                // overlay selection-time metadata coming from /elements/next
                setElement({
                  ...fetched,
                  selection: meta.selection ?? fetched.selection,
                  similarity: meta.similarity ?? fetched.similarity,
                  rank: meta.rank ?? fetched.rank,
                });
              } else {
                setElement(fetched);
              }
            } else {
              navigate(`/projects/${projectName}/tag/noelement`);
              setElement(null);
            }
          })
          .finally(() => {
            //info: get statistics call returns often outdated data
            reFetchStatistics();
          });
      }
    }
  }, [
    elementId,
    getNextElementId,
    getElementById,
    navigate,
    effectivePhase,
    projectName,
    reFetchStatistics,
    selectionConfig,
    setAppContext,
    notify,
    element,
  ]);

  // mark the moment the current element became visible to the user
  useEffect(() => {
    if (element?.element_id && element.element_id !== 'noelement') {
      elementDisplayedAtRef.current = { id: element.element_id, at: Date.now() };
    }
  }, [element?.element_id]);

  const postAnnotation = useCallback(
    async (label: string | null, elementId: string, comment?: string) => {
      if (elementId === 'noelement') return; // forbid annotation on noelement
      if (elementId) {
        const displayedInfo = elementDisplayedAtRef.current;
        const durationMs =
          displayedInfo && displayedInfo.id === elementId
            ? Date.now() - displayedInfo.at
            : undefined;
        await addAnnotation(elementId, label, comment || null, selectionHistory[elementId]);
        const newElement = await getElementById(elementId, effectivePhase);
        if (newElement) {
          addElementInAnnotationSessionHistory(
            elementId,
            newElement.text,
            label,
            comment,
            undefined,
            durationMs,
          );
          setElement(newElement);
          // wait for 500ms before fetch new element to see new button state
          setTimeout(() => {
            navigate(`/projects/${projectName}/tag/`);
          }, 200);
        }
        // does not do nothing as we remount through navigate reFetchStatistics();

        // authorize retraining after first annotation
        setAuthorizeRetraining(true);
      }
    },
    [
      addAnnotation,
      selectionHistory,
      projectName,
      navigate,
      getElementById,
      setElement,
      effectivePhase,
      addElementInAnnotationSessionHistory,
    ],
  );

  const textInFrame = element?.text.slice(0, displayConfig.numberOfTokens * 4) || '';
  const textOutFrame = element?.text.slice(displayConfig.numberOfTokens * 4) || '';

  const lastTag = element?.history && element.history.length > 0 ? element.history[0].label : null;

  const fetchNextElement = useCallback(() => {
    getNextElementId().then((res) => {
      if (res && res.n_sample) setNSample(res.n_sample);
      if (res && res.element_id) {
        if (res.element_id === elementId) {
          notify({
            type: 'warning',
            message:
              'Refetching yielded the same text input. Change selection settings to get a different result.',
          });
        }
        navigate(`/projects/${projectName}/tag/${res.element_id}`);
      } else {
        navigate(`/projects/${projectName}/tag/noelement`);
      }
    });
  }, [getNextElementId, notify, setNSample, navigate, projectName, elementId]);

  const highlightTextRaw = [selectionConfig.filter, ...displayConfig.highlightText.split('\n')];
  const highlightText = highlightTextRaw.filter(
    (text): text is string => typeof text === 'string' && text.trim() !== '',
  );

  // Now filter by valid regex
  const validHighlightText = highlightText.filter(isValidRegex);

  // display active menu
  const [activeMenu, setActiveMenu] = useState<boolean>(false);

  /**
   * Update element if selectionConfig changed :
   * - refetch if active model is activated
   * - getNextElement if in noelement page
   */
  const refetchElement = useCallback(async () => {
    if (elementId) {
      const newElement = await getElementById(elementId, effectivePhase);
      if (newElement) setElement(newElement);
    }
  }, [setElement, getElementById, elementId, effectivePhase]);

  useEffect(() => {
    refetchElement();
    // disabling echaustive deps as we only want to track phase to avoid unnecessary refetch
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeModel]);

  useEffect(() => {
    // fetch next element in the new phase
    // only if there is one current element to avoid triggering fetchnext at page load
    if (element !== null) {
      fetchNextElement();
    }
    // disabling echaustive deps as we only want to track phase to avoid unnecessary fetchNext
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectivePhase]);

  useEffect(() => {
    if (element !== null) {
      fetchNextElement();
    }
    // disabling echaustive deps as we only want to track phase to avoid unnecessary fetchNext
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectionConfig]);

  if (!projectName || !currentScheme) return;

  const numberAnnotated = history.filter(
    (hp) => hp.dataset === 'train' && hp.project_slug === projectName && !hp.skip,
  );

  // The annotation area (element + tag inputs) is built once and rendered
  // either in place or inside the focus-mode modal — never both: the tag
  // inputs attach document-level keyboard shortcuts that must stay unique.
  const annotationArea = (
    <>
      {elementId === 'noelement' && (
        <div className="alert horizontal center">
          No element available
          <button className="btn-primary-action" onClick={fetchNextElement}>
            <LuRefreshCw size={20} /> Get element
          </button>
        </div>
      )}
      {/**
       * ANNOTATION BLOCK
       **/}
      <div
        className={classNames(
          'annotation-block',
          (displayConfig.forceOneColumnLayout || kindScheme == 'multilabel') &&
            'force-one-column-layout',
          project?.params?.kind === 'image' &&
            kindScheme === 'multilabel' &&
            'image-multilabel-wide',
        )} // add class to force bottom if settings OR multiclass label
        style={
          {
            '--text-width': showFocusMode
              ? `${displayConfig.focusTextWidth ?? 70}%`
              : `${displayConfig.textFrameWidth}%`,
          } as CSSProperties
        }
      >
        {elementId !== 'noelement' &&
          (kindScheme !== 'span' ? (
            <>
              {project?.params?.kind === 'image' ? (
                <ImageClassificationPanelImagexp
                  element={element as ElementOutModel}
                  displayConfig={displayConfig}
                  elementId={elementId as string}
                  projectSlug={project.params.project_slug}
                  frameRef={frameRef as unknown as HTMLDivElement}
                />
              ) : (
                <TextClassificationPanel
                  element={element as ElementOutModel}
                  displayConfig={displayConfig}
                  textInFrame={textInFrame}
                  textOutFrame={textOutFrame}
                  validHighlightText={validHighlightText}
                  elementId={elementId as string}
                  lastTag={lastTag as string}
                  phase={effectivePhase}
                  frameRef={frameRef as unknown as HTMLDivElement}
                />
              )}
            </>
          ) : (
            <>
              <TextSpanPanel
                elementId={elementId || 'noelement'}
                displayConfig={displayConfig}
                postAnnotation={postAnnotation}
                labels={availableLabels}
                text={element?.text as string}
                lastTag={lastTag as string}
                element={element as ElementOutModel}
              />
            </>
          ))}

        {elementId !== 'noelement' && (
          <>
            {kindScheme == 'multiclass' && (
              <MulticlassInput
                elementId={elementId || 'noelement'}
                postAnnotation={postAnnotation}
                labels={availableLabels}
                phase={effectivePhase}
                element={element as ElementOutModel}
              />
            )}
            {kindScheme == 'multilabel' && (
              <MultilabelInput
                elementId={elementId || 'noelement'}
                postAnnotation={postAnnotation}
                labels={availableLabels}
                element={element as ElementOutModel}
              />
            )}
          </>
        )}
      </div>
    </>
  );

  return (
    <>
      {/**
       * Annotation mode form
       **/}
      <AnnotationModeForm
        fetchNextElement={fetchNextElement}
        setActiveMenu={setActiveMenu}
        setShowDisplayViz={setShowDisplayViz}
        setShowDisplayConfig={setShowDisplayConfig}
        setShowPromptsModal={setShowPromptsModal}
        setShowFocusMode={setShowFocusMode}
        nSample={nSample}
        statistics={statistics}
      />
      {!showFocusMode && annotationArea}

      <div>
        {displayConfig.displayHistory ? (
          <AnnotationHistoryList />
        ) : (
          <div className="d-flex gap-2 align-items-center">
            <button
              className="btn btn-link p-0"
              onClick={() => {
                setAppContext((prev) => ({
                  ...prev,
                  displayConfig: {
                    ...displayConfig,
                    displayHistory: true,
                  },
                }));
              }}
              title="Show history"
            >
              <HiOutlineEyeOff size={20} />
            </button>
            <button
              className="btn btn-link p-0"
              onClick={() => setShowCodebook(true)}
              title="Show codebook"
            >
              <FaBookOpen size={18} />
            </button>
          </div>
        )}
      </div>
      {/**
       * Manage active learning
       **/}
      <SelectActiveLearning
        display={activeMenu}
        setActiveMenu={setActiveMenu}
        setSelectFirstModelTrained={setSelectFirstModelTrained}
        selectFirstModelTrained={selectFirstModelTrained}
        numberAnnotated={numberAnnotated.length}
        authorize={authorizeRetraining}
      />
      <Modal show={showDisplayViz} onHide={handleCloseViz} size="xl" id="viz-modal">
        <Modal.Header closeButton>
          <Modal.Title>Current projection</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <div className="horizontal center" style={{ overflowY: 'scroll' }}>
            <DisplayProjection
              projectName={projectName}
              currentScheme={currentScheme}
              currentElement={element}
            />
          </div>
        </Modal.Body>
      </Modal>
      <Modal
        show={showFocusMode}
        onHide={() => setShowFocusMode(false)}
        fullscreen={true}
        id="focus-modal"
      >
        <Modal.Header closeButton>
          <Modal.Title>Focus</Modal.Title>
          <div className="focus-modal-controls">
            <label>
              <span className="small-gray">Text size {displayConfig.focusFontSize ?? 100}%</span>
              <input
                type="range"
                min={60}
                max={200}
                step={10}
                value={displayConfig.focusFontSize ?? 100}
                onChange={(e) =>
                  setAppContext((prev) => ({
                    ...prev,
                    displayConfig: {
                      ...prev.displayConfig,
                      focusFontSize: Number(e.target.value),
                    },
                  }))
                }
              />
            </label>
            <label>
              <span className="small-gray">Width {displayConfig.focusTextWidth ?? 70}%</span>
              <input
                type="range"
                min={40}
                max={100}
                step={5}
                value={displayConfig.focusTextWidth ?? 70}
                onChange={(e) =>
                  setAppContext((prev) => ({
                    ...prev,
                    displayConfig: {
                      ...prev.displayConfig,
                      focusTextWidth: Number(e.target.value),
                    },
                  }))
                }
              />
            </label>
          </div>
        </Modal.Header>
        <Modal.Body
          style={{ '--focus-font-size': displayConfig.focusFontSize ?? 100 } as CSSProperties}
        >
          {annotationArea}
        </Modal.Body>
      </Modal>
      <Modal show={showDisplayConfig} onHide={handleCloseConfig} size="sm" id="config-modal">
        <Modal.Header closeButton>
          <Modal.Title>Configuration</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <TagDisplayParameters />
        </Modal.Body>
      </Modal>
      <Modal show={showCodebook} onHide={() => setShowCodebook(false)} size="xl">
        <Modal.Header closeButton>
          <Modal.Title>Codebook</Modal.Title>
        </Modal.Header>
        <Modal.Body data-color-mode="light">
          <MDEditor.Markdown source={codebook || ''} style={{ backgroundColor: 'transparent' }} />
        </Modal.Body>
      </Modal>
      <Modal
        show={showPromptsModal}
        onHide={() => setShowPromptsModal(false)}
        size="lg"
        id="prompts-modal"
      >
        <Modal.Header closeButton>
          <Modal.Title>Prompts for embedding-based selection</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {project?.params?.project_slug && (
            <PromptsPanel
              projectSlug={project.params.project_slug}
              state={(project as unknown as { prompts?: PromptsProjectStateModel | null }).prompts}
              currentText={project.params.kind !== 'image' ? element?.text ?? undefined : undefined}
            />
          )}
        </Modal.Body>
      </Modal>
    </>
  );
};
