import { FC, useEffect, useMemo, useState } from 'react';
import { Modal } from 'react-bootstrap';
import DataGrid, { Column, RenderEditCellProps } from 'react-data-grid';
import { Link } from 'react-router-dom';
import { useAppContext } from '../core/useAppContext';
import { useTableAnnotationEdits } from '../core/useTableAnnotationEdits';
import { MLStatisticsModel } from '../types';
import { DisplaySpanFalsePredictions, SpanFalsePredictionDoc } from './DisplaySpanFalsePredictions';
import { ImageThumbnailImagexp } from './ImageThumbnailImagexp';
import { LabelDropdownEditor, LabelMultiSelectEditor } from './LabelCellEditors';

interface WrongPredictionsModalProps {
  show: boolean;
  onHide: () => void;
  scores: MLStatisticsModel;
  modelName?: string;
  projectSlug?: string | null;
  dataset: string;
}

interface Row {
  id: string;
  'GS-label': string;
  prediction: string;
  text: string;
}

/**
 * Modal listing the wrong predictions of a model, with the possibility to reannotate them
 **/
export const WrongPredictionsModal: FC<WrongPredictionsModalProps> = ({
  show,
  onHide,
  scores,
  modelName,
  projectSlug,
  dataset,
}) => {
  const {
    appContext: { currentProject, currentScheme },
  } = useAppContext();
  const isImageKind = currentProject?.params.kind === 'image';
  const isNer = scores.training_kind === 'ner';
  const isMultilabel = scores.training_kind === 'multilabel';
  const availableLabels =
    (currentScheme && currentProject?.schemes.available[currentScheme]?.labels) || [];
  const datasetClean = dataset.includes('test')
    ? 'test'
    : dataset.includes('valid')
      ? 'valid'
      : 'train';
  const canEdit = !!projectSlug && !!currentScheme && !isNer;

  const { modifiedRows, registerChange, validateChanges, resetChanges } = useTableAnnotationEdits(
    projectSlug || null,
    currentScheme || null,
    datasetClean,
  );

  // corrected labels by element id, kept to display them over the computed scores
  const [corrected, setCorrected] = useState<Record<string, string>>({});
  const falsePredictions = scores['false_predictions'];
  const rows = useMemo(
    () =>
      !isNer && Array.isArray(falsePredictions)
        ? (falsePredictions as Row[])
            .filter((r) => typeof r === 'object')
            .map((r) => (r.id in corrected ? { ...r, 'GS-label': corrected[r.id] } : r))
        : [],
    [falsePredictions, corrected, isNer],
  );

  // pending changes are tied to a scheme and a dataset
  useEffect(() => {
    setCorrected({});
    resetChanges();
  }, [currentScheme, datasetClean, resetChanges]);

  const onRowsChange = (newRows: Row[], { indexes }: { indexes: number[] }) => {
    indexes.forEach((i) => {
      const { id, 'GS-label': label } = newRows[i];
      if (label === rows[i]['GS-label']) return;
      setCorrected((prev) => ({ ...prev, [id]: label }));
      registerChange(id, label);
    });
  };

  function renderLabelEditor({ row, onRowChange }: RenderEditCellProps<Row>) {
    const Editor = isMultilabel ? LabelMultiSelectEditor : LabelDropdownEditor;
    return (
      <Editor
        value={row['GS-label']}
        availableLabels={availableLabels}
        onChange={(label, commit) => onRowChange({ ...row, 'GS-label': label }, commit)}
        onClose={() => onRowChange(row, true)}
      />
    );
  }

  const columns: readonly Column<Row>[] = [
    {
      key: 'id',
      name: 'Id',
      resizable: true,
      width: 180,
      renderCell: (props) => (
        <div className={props.row.id in modifiedRows ? 'modified-cell' : ''}>
          {projectSlug ? (
            <Link to={`/projects/${projectSlug}/tag/${props.row.id}?dataset=${datasetClean}`}>
              {props.row.id}
            </Link>
          ) : (
            props.row.id
          )}
        </div>
      ),
    },
    {
      name: canEdit ? 'Label ✎' : 'Label',
      key: 'GS-label',
      resizable: true,
      width: isMultilabel ? 220 : 120,
      renderEditCell: canEdit ? renderLabelEditor : undefined,
      editorOptions: isMultilabel ? { commitOnOutsideClick: false } : undefined,
    },
    {
      name: 'Prediction',
      key: 'prediction',
      resizable: true,
      width: 120,
    },
    {
      name: isImageKind ? 'Image' : 'Text',
      key: 'text',
      resizable: true,
      renderCell: (props) =>
        isImageKind && projectSlug ? (
          <div
            style={{
              width: '100%',
              height: '100%',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
            }}
          >
            <ImageThumbnailImagexp
              projectSlug={projectSlug}
              elementId={props.row.id}
              maxWidth={160}
              maxHeight={110}
            />
          </div>
        ) : (
          <div
            style={{
              maxHeight: '100%',
              width: '100%',
              whiteSpace: 'wrap',
              overflowY: 'auto',
              userSelect: 'none',
            }}
          >
            {props.row.text}
          </div>
        ),
    },
  ];

  return (
    <Modal show={show} id="quickmodel-modal" onHide={onHide} centered size="xl">
      <Modal.Header closeButton>
        <Modal.Title>Wrong predictions of the model {modelName}</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        {isNer ? (
          <DisplaySpanFalsePredictions
            falsePredictions={(falsePredictions as unknown as SpanFalsePredictionDoc[]) || []}
            projectSlug={projectSlug}
            dataset={dataset}
          />
        ) : (
          <>
            {canEdit && (
              <div className="explanations">
                Double-click on a label to correct the annotation. Scores are only updated once the
                model is trained or evaluated again.
              </div>
            )}
            <div className="horizontal center">
              {Object.keys(modifiedRows).length > 0 && (
                <button className="btn-primary-action" onClick={validateChanges}>
                  Validate changes
                </button>
              )}
            </div>
            <DataGrid<Row>
              className="fill-grid rdg-light"
              columns={columns}
              rows={rows}
              rowHeight={isImageKind ? 120 : 80}
              onRowsChange={onRowsChange}
            />
          </>
        )}
      </Modal.Body>
    </Modal>
  );
};
