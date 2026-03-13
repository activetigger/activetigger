import { CSSProperties, FC, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { HiOutlineEyeOff } from 'react-icons/hi';
import { LuRefreshCw } from 'react-icons/lu';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import {
  useAddAnnotation,
  useGetElementById,
  useGetNextElementId,
  useStatistics,
} from '../../core/api';
import { useAppContext } from '../../core/useAppContext';
import { ElementOutModel } from '../../types';

import classNames from 'classnames';
import { Modal } from 'react-bootstrap';
import { useNotifications } from '../../core/notifications';
import { useAnnotationSessionHistory } from '../../core/useHistory';
import { isValidRegex } from '../../core/utils';
import { TagDisplayParameters } from '../TagDisplayParameters';
import { DisplayProjection } from '../vizualisation/DisplayProjection';
import { AnnotationHistoryList } from './AnnotationHistoryList';
import { AnnotationModeForm } from './AnnotationMode';
import { MulticlassInput } from './MulticlassInput';
import { MultilabelInput } from './MultilabelInput';
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

  const [showDisplayConfig, setShowDisplayConfig] = useState<boolean>(false);
  const [showDisplayViz, setShowDisplayViz] = useState<boolean>(false);
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
          .then((element) => {
            if (element) setElement(element);
            else {
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

  const postAnnotation = useCallback(
    async (label: string | null, elementId: string, comment?: string) => {
      if (elementId === 'noelement') return; // forbid annotation on noelement
      if (elementId) {
        await addAnnotation(elementId, label, comment || null, selectionHistory[elementId]);
        const newElement = await getElementById(elementId, effectivePhase);
        if (newElement) {
          addElementInAnnotationSessionHistory(elementId, newElement.text, label, comment);
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
        nSample={nSample}
        statistics={statistics}
      />

      {/**
       * ANNOTATION BLOCK
       **/}
      <div
        className={classNames(
          'annotation-block',
          (displayConfig.forceOneColumnLayout || kindScheme == 'multilabel') &&
            'force-one-column-layout',
        )} // add class to force bottom if settings OR multiclass label
        style={
          {
            '--text-width': `${displayConfig.textFrameWidth}%`,
          } as CSSProperties
        }
      >
        {elementId === 'noelement' ? (
          <div className="alert horizontal center">
            <div>
              No element available
              <button className="btn-primary-action" onClick={fetchNextElement}>
                <LuRefreshCw size={20} /> Get element
              </button>
            </div>
          </div>
        ) : kindScheme !== 'span' ? (
          <>
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
        )}

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

      <div>
        {displayConfig.displayHistory ? (
          <AnnotationHistoryList />
        ) : (
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
      <Modal show={showDisplayConfig} onHide={handleCloseConfig} size="sm" id="config-modal">
        <Modal.Header closeButton>
          <Modal.Title>Configuration</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <TagDisplayParameters />
        </Modal.Body>
      </Modal>
    </>
  );
};
