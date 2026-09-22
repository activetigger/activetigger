import { useCallback, useState } from 'react';
import { AnnotationModel } from '../types';
import { useAddTableAnnotations } from './api';

/**
 * Pending label changes made in a table, sent in batch on validation
 */
export function useTableAnnotationEdits(
  projectSlug: string | null,
  scheme: string | null,
  dataset: string | null,
) {
  const [modifiedRows, setModifiedRows] = useState<Record<string, AnnotationModel>>({});
  const { addTableAnnotations } = useAddTableAnnotations(projectSlug, scheme, dataset);

  const registerChange = useCallback(
    (elementId: string, label: string) => {
      if (!projectSlug || !scheme) return;
      setModifiedRows((prev) => ({
        ...prev,
        [elementId]: {
          element_id: elementId,
          label,
          scheme,
          project_slug: projectSlug,
          dataset: dataset || 'train',
        },
      }));
    },
    [projectSlug, scheme, dataset],
  );

  const validateChanges = useCallback(async () => {
    await addTableAnnotations(Object.values(modifiedRows));
    setModifiedRows({});
  }, [addTableAnnotations, modifiedRows]);

  const resetChanges = useCallback(() => setModifiedRows({}), []);

  return { modifiedRows, registerChange, validateChanges, resetChanges };
}
