import { FC, useEffect, useState } from 'react';
import { SubmitHandler, useForm, useWatch } from 'react-hook-form';

import { useImportFeature } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { loadFile } from '../../core/utils';
import { UploadProgressBar } from '../UploadProgressBar';

interface ImportFeatureFormProps {
  projectSlug: string;
  callback?: (state: boolean) => void;
}

type FormValues = {
  files: FileList;
  name: string;
  id_column: string;
  mode: 'all' | 'select';
};

const MAX_SIZE_MB = 200;

export const ImportFeature: FC<ImportFeatureFormProps> = ({ projectSlug, callback }) => {
  const { notify } = useNotifications();
  const { importFeature, progression } = useImportFeature();

  const { register, control, handleSubmit, reset } = useForm<FormValues>({
    defaultValues: { mode: 'all', name: '', id_column: '' },
  });
  const files = useWatch({ control, name: 'files' });
  const mode = useWatch({ control, name: 'mode' });
  const idColumn = useWatch({ control, name: 'id_column' });

  const [headers, setHeaders] = useState<string[]>([]);
  const [previewFile, setPreviewFile] = useState<File | null>(null);
  const [selectedColumns, setSelectedColumns] = useState<Set<string>>(new Set());
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!files || files.length === 0) {
      setHeaders([]);
      setPreviewFile(null);
      setSelectedColumns(new Set());
      return;
    }
    const file = files[0];
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      notify({ type: 'error', message: `File is too large (max ${MAX_SIZE_MB} MB)` });
      return;
    }
    loadFile(file).then((data) => {
      if (data === null) {
        notify({ type: 'error', message: 'Error reading the file.' });
        return;
      }
      setHeaders(data.headers.filter(Boolean));
      setPreviewFile(file);
      setSelectedColumns(new Set());
    });
  }, [files, notify]);

  const availableColumns = headers.filter((h) => h !== idColumn);

  const toggleColumn = (col: string) => {
    setSelectedColumns((prev) => {
      const next = new Set(prev);
      if (next.has(col)) next.delete(col);
      else next.add(col);
      return next;
    });
  };

  const onSubmit: SubmitHandler<FormValues> = async (formData) => {
    if (!previewFile) {
      notify({ type: 'error', message: 'Please select a file first.' });
      return;
    }
    if (!formData.name.trim()) {
      notify({ type: 'error', message: 'Please enter a feature name.' });
      return;
    }
    if (!formData.id_column) {
      notify({ type: 'error', message: 'Please choose an ID column.' });
      return;
    }
    let columns: string[] | null = null;
    if (formData.mode === 'select') {
      columns = Array.from(selectedColumns);
      if (columns.length === 0) {
        notify({ type: 'error', message: 'Please select at least one column.' });
        return;
      }
    }
    setSubmitting(true);
    const ok = await importFeature(projectSlug, {
      file: previewFile,
      name: formData.name.trim(),
      idColumn: formData.id_column,
      columns,
    });
    setSubmitting(false);
    if (ok) {
      reset();
      setHeaders([]);
      setPreviewFile(null);
      setSelectedColumns(new Set());
      if (callback) callback(false);
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div className="alert alert-warning">
        Import a pre-computed feature. The file must contain one row per project element, matched on
        the project's ID. Only numeric columns can be imported.
      </div>

      <label htmlFor="importFile">File (csv, parquet, xlsx)</label>
      <input
        id="importFile"
        type="file"
        accept=".csv,.parquet,.xlsx"
        className="form-control"
        {...register('files')}
      />

      {headers.length > 0 && (
        <>
          <label htmlFor="id_column" className="mt-2">
            ID column (matched on the project's original IDs)
          </label>
          <select id="id_column" className="form-select" {...register('id_column')}>
            <option value="">-- select --</option>
            {headers.map((h) => (
              <option key={h} value={h}>
                {h}
              </option>
            ))}
          </select>

          <div className="mt-2">
            <label>Columns to import</label>
            <div>
              <label className="me-3">
                <input type="radio" value="all" {...register('mode')} /> All remaining columns
                (embedding)
              </label>
              <label>
                <input type="radio" value="select" {...register('mode')} /> Select columns
              </label>
            </div>
          </div>

          {mode === 'select' && (
            <div className="mt-2 border p-2" style={{ maxHeight: '200px', overflowY: 'auto' }}>
              {availableColumns.length === 0 && (
                <small>Pick an ID column to see available columns.</small>
              )}
              {availableColumns.map((c) => (
                <div key={c}>
                  <label>
                    <input
                      type="checkbox"
                      checked={selectedColumns.has(c)}
                      onChange={() => toggleColumn(c)}
                    />{' '}
                    {c}
                  </label>
                </div>
              ))}
            </div>
          )}

          <label htmlFor="feature_name" className="mt-2">
            Feature name
          </label>
          <input
            id="feature_name"
            type="text"
            className="form-control"
            placeholder="e.g. embeddings_v1"
            {...register('name', { required: true })}
          />
          <small>
            Stored as <code>imported-&lt;name&gt;</code>
          </small>

          <button type="submit" className="btn-submit mt-3" disabled={submitting}>
            Import feature
          </button>
        </>
      )}

      {submitting && <UploadProgressBar progression={progression} />}
    </form>
  );
};
