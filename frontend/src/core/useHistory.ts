import { useCallback } from 'react';
import { ElementHistoryPoint } from '../types';
import { useAppContext } from './useAppContext';

export function useAnnotationSessionHistory() {
  const {
    appContext: { currentProject, phase, currentScheme },
    setAppContext,
  } = useAppContext();

  const addElementInAnnotationSessionHistory = useCallback(
    (
      elementId: string,
      elementText: string | undefined,
      label: string | undefined | null,
      comment?: string,
      skip?: boolean,
    ) => {
      if (currentProject?.params.project_slug && currentScheme) {
        const historyPoint: ElementHistoryPoint = {
          project_slug: currentProject?.params.project_slug,
          dataset: phase,
          scheme: currentScheme,
          element_id: elementId,
          element_text: elementText || '',
          label: label,
          comment: comment,
          time: new Date().toISOString(),
          skip: skip,
        };
        setAppContext((prev) => ({
          ...prev,
          history: [historyPoint, ...prev.history],
        }));
      }
    },
    [currentProject?.params.project_slug, phase, currentScheme, setAppContext],
  );

  const clearAnnotationSessionHistory = useCallback(() => {
    setAppContext((prev) => ({
      ...prev,
      history: [],
    }));
  }, [setAppContext]);

  return { addElementInAnnotationSessionHistory, clearAnnotationSessionHistory };
}
