import axios from 'axios';
import { FC, useEffect, useRef, useState } from 'react';
import { Button, Modal } from 'react-bootstrap';
import DataTable from 'react-data-table-component';
import { Controller, SubmitHandler, useForm, useWatch } from 'react-hook-form';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import Select from 'react-select';
import PulseLoader from 'react-spinners/PulseLoader';
import { Tooltip } from 'react-tooltip';

import {
  getPrepareStatus,
  useGetPreparedFile,
  usePrepareSplit,
  useStopPrepareTask,
  useUploadPrepareFile,
} from '../../core/api';
import { formatUploadError } from '../../core/HTTPError';
import { useNotifications } from '../../core/notifications';
import { isValidRegex } from '../../core/utils';
import { PrepareSessionModel, PrepareStatusModel } from '../../types';
import { UploadProgressBar } from '../UploadProgressBar';

type Option = { value: string; label: string };

interface PrepareFormValues {
  files: FileList;
  cols_text: string[];
  col_id: string;
  cols_keep: string[];
  method: 'chunk' | 'regex' | 'wtpsplit';
  chunk_size: number;
  regex_pattern: string;
  granularity: 'sentence' | 'paragraph';
  language: string;
  min_chars: number;
}

const langages = [
  { value: 'en', label: 'English' },
  { value: 'fr', label: 'French' },
  { value: 'es', label: 'Spanish' },
  { value: 'de', label: 'German' },
  { value: 'cn', label: 'Chinese' },
  { value: 'ja', label: 'Japanese' },
  { value: 'nb', label: 'Norwegian' },
];

// preview of the first rows of a dataframe sent by the API
const PreviewTable: FC<{ rows: Record<string, string>[] }> = ({ rows }) => {
  if (rows.length === 0) return null;
  const headers = Object.keys(rows[0]);
  return (
    <DataTable<Record<string, string>>
      columns={headers.map((h) => ({
        name: h,
        selector: (row) => row[h],
        width: '200px',
      }))}
      data={rows}
    />
  );
};

/**
 * Standalone tool to prepare a dataset : upload a file, select columns,
 * split the texts into chunks and export the result
 */
export const DatasetPreparationForm: FC = () => {
  const maxSizeMB = 400;
  const maxSize = maxSizeMB * 1024 * 1024;

  const { notify } = useNotifications();
  const { uploadPrepareFile, progression, cancel } = useUploadPrepareFile();
  const { prepareSplit } = usePrepareSplit();
  const { stopPrepareTask } = useStopPrepareTask();
  const { getPreparedFile } = useGetPreparedFile();

  const { register, control, handleSubmit } = useForm<PrepareFormValues>({
    defaultValues: {
      cols_text: [],
      col_id: 'row_number',
      cols_keep: [],
      method: 'chunk',
      chunk_size: 500,
      regex_pattern: '\\n',
      granularity: 'sentence',
      language: 'en',
      min_chars: 10,
    },
  });
  const files = useWatch({ control, name: 'files' });
  const method = useWatch({ control, name: 'method' });

  const [showImportModal, setShowImportModal] = useState<boolean>(false);
  const [uploading, setUploading] = useState<boolean>(false);
  const [session, setSession] = useState<PrepareSessionModel | null>(null);
  const [splitting, setSplitting] = useState<boolean>(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [result, setResult] = useState<PrepareStatusModel | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [exportFormat, setExportFormat] = useState<string>('csv');
  const [downloading, setDownloading] = useState<boolean>(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // clear the polling on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const stopPolling = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setSplitting(false);
    setProgress(null);
    setTaskId(null);
  };

  // user asked to stop the running split task
  const handleStopSplit = async () => {
    if (taskId) await stopPrepareTask(taskId);
    stopPolling();
    notify({ type: 'warning', message: 'Split task cancelled.' });
  };

  const availableFields: Option[] = (session?.columns || [])
    .filter((c) => !!c)
    .map((c) => ({ value: c, label: c }));

  const handleUpload = async () => {
    if (!files || files.length === 0) {
      notify({ type: 'error', message: 'Select a file first.' });
      return;
    }
    const file = files[0];
    if (file.size > maxSize) {
      notify({ type: 'error', message: `File is too large (maximum size: ${maxSizeMB} MB)` });
      return;
    }
    try {
      setUploading(true);
      setSession(null);
      setResult(null);
      const newSession = await uploadPrepareFile(file);
      setSession(newSession);
      setShowImportModal(false);
    } catch (error) {
      if (axios.isAxiosError(error)) {
        notify({ type: 'error', message: formatUploadError(error, file.size) });
      } else {
        notify({ type: 'error', message: error + '' });
      }
    } finally {
      setUploading(false);
    }
  };

  const onSubmit: SubmitHandler<PrepareFormValues> = async (formData) => {
    if (!session) return;
    if (!formData.cols_text || formData.cols_text.length === 0) {
      notify({ type: 'error', message: 'Select at least one text column.' });
      return;
    }
    if (formData.method === 'chunk' && (!formData.chunk_size || Number(formData.chunk_size) <= 0)) {
      notify({ type: 'error', message: 'Enter a positive number of characters.' });
      return;
    }
    if (
      formData.method === 'regex' &&
      (!formData.regex_pattern || !isValidRegex(formData.regex_pattern))
    ) {
      notify({ type: 'error', message: 'Enter a valid regex pattern.' });
      return;
    }

    setResult(null);
    const newTaskId = await prepareSplit({
      session_id: session.session_id,
      cols_text: formData.cols_text,
      col_id: formData.col_id,
      cols_keep: formData.cols_keep,
      method: formData.method,
      chunk_size: formData.method === 'chunk' ? Number(formData.chunk_size) : null,
      regex_pattern: formData.method === 'regex' ? formData.regex_pattern : null,
      granularity: formData.method === 'wtpsplit' ? formData.granularity : null,
      language: formData.method === 'wtpsplit' ? formData.language : null,
      min_chars: Math.max(Number(formData.min_chars) || 0, 0),
    });
    if (!newTaskId) return;

    setSplitting(true);
    setTaskId(newTaskId);
    const maxDuration = 30 * 60 * 1000;
    const startTime = Date.now();
    intervalRef.current = setInterval(async () => {
      try {
        const status = await getPrepareStatus(session.session_id, newTaskId);
        // the polling may have been stopped while the request was in flight
        if (!intervalRef.current) return;
        if (status?.status === 'done') {
          stopPolling();
          setResult(status);
          notify({ type: 'success', message: `Dataset prepared: ${status.n_rows} rows.` });
        } else if (status?.status === 'failed' || status?.status === 'not found') {
          stopPolling();
          notify({ type: 'error', message: status.error || 'The split task failed.' });
        } else if (status) {
          setProgress(status.progress ?? null);
        }
        if (Date.now() - startTime >= maxDuration) {
          stopPolling();
          notify({ type: 'error', message: 'Timeout during the split. Try again later.' });
        }
      } catch (error) {
        stopPolling();
        notify({ type: 'error', message: error + '' });
      }
    }, 1000);
  };

  const handleDownload = async () => {
    if (!session) return;
    setDownloading(true);
    await getPreparedFile(session.session_id, exportFormat);
    setDownloading(false);
  };

  return (
    <div>
      <div className="explanations">
        Prepare a dataset : upload a file, split the texts into smaller units, and export the
        result. You can then use it to create a project.
      </div>

      <button
        type="button"
        className="btn btn-secondary-action my-2"
        onClick={() => setShowImportModal(true)}
        disabled={splitting}
      >
        {session ? 'Load another file' : 'Load a file'}
      </button>

      <Modal
        show={showImportModal}
        onHide={uploading ? undefined : () => setShowImportModal(false)}
      >
        <Modal.Header closeButton={!uploading}>
          <Modal.Title>Load a file</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <label htmlFor="prepareFile" className="form-label">
            File to process
          </label>
          <input className="form-control" id="prepareFile" type="file" {...register('files')} />
          <div className="explanations" style={{ fontSize: 'smaller', fontWeight: 'normal' }}>
            Tabular file (csv, xlsx or parquet) or zip archive of documents (txt or docx, one row
            per file) &lt; {maxSizeMB} MB
          </div>
          {uploading && <UploadProgressBar progression={progression} cancel={cancel} />}
        </Modal.Body>
        <Modal.Footer>
          <Button
            variant="primary"
            onClick={handleUpload}
            disabled={uploading || !files || files.length === 0}
          >
            {uploading ? 'Loading…' : 'Load the file'}
          </Button>
          <Button
            variant="secondary"
            onClick={() => setShowImportModal(false)}
            disabled={uploading}
          >
            Cancel
          </Button>
        </Modal.Footer>
      </Modal>

      {session && (
        <form onSubmit={handleSubmit(onSubmit)}>
          <div>
            Loaded <b>{session.filename}</b> : <b>{session.n_rows}</b> rows
          </div>
          <PreviewTable rows={session.preview as Record<string, string>[]} />

          <label htmlFor="col_id">Id column (must contain unique values)</label>
          <select id="col_id" disabled={splitting} {...register('col_id')}>
            <option key="row_number" value="row_number">
              Row number
            </option>
            {session.columns
              .filter((h) => !!h)
              .map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              ))}
          </select>

          <label htmlFor="cols_text">Text columns (selected fields will be concatenated)</label>
          <Controller
            name="cols_text"
            control={control}
            render={({ field: { value, onChange } }) => (
              <Select
                inputId="cols_text"
                options={availableFields}
                isMulti
                isDisabled={splitting}
                value={
                  value
                    ? value
                        .map((v: string) => availableFields.find((opt) => opt.value === v))
                        .filter(Boolean)
                    : []
                }
                onChange={(selectedOptions) => {
                  onChange(selectedOptions ? selectedOptions.map((option) => option?.value) : []);
                }}
              />
            )}
          />

          <label htmlFor="cols_keep">
            Columns to keep (duplicated on each row of the new dataset)
          </label>
          <Controller
            name="cols_keep"
            control={control}
            render={({ field: { value, onChange } }) => (
              <Select
                inputId="cols_keep"
                options={availableFields}
                isMulti
                isDisabled={splitting}
                value={
                  value
                    ? value
                        .map((v: string) => availableFields.find((opt) => opt.value === v))
                        .filter(Boolean)
                    : []
                }
                onChange={(selectedOptions) => {
                  onChange(selectedOptions ? selectedOptions.map((option) => option?.value) : []);
                }}
              />
            )}
          />

          <label htmlFor="method">
            Rule to split the texts{' '}
            <a className="split-method-info">
              <HiOutlineQuestionMarkCircle />
            </a>
            <Tooltip anchorSelect=".split-method-info" place="top" clickable>
              Chunks and regex are simple text operations.
              <br />
              wtpsplit (Segment Any Text) is a machine learning model that detects sentence and
              paragraph boundaries in 85 languages, even without punctuation.
              <br />
              See the{' '}
              <a
                href="https://github.com/segment-any-text/wtpsplit"
                target="_blank"
                rel="noopener noreferrer"
              >
                wtpsplit documentation
              </a>
              .
            </Tooltip>
          </label>
          <select id="method" disabled={splitting} {...register('method')}>
            <option value="chunk">Chunks of about N characters (words are not cut)</option>
            <option value="regex">Split on a regex pattern</option>
            <option value="wtpsplit">Sentences or paragraphs (wtpsplit segmentation model)</option>
          </select>

          {method === 'chunk' && (
            <>
              <label htmlFor="chunk_size">Number of characters per chunk</label>
              <input
                id="chunk_size"
                type="number"
                min={1}
                disabled={splitting}
                {...register('chunk_size')}
              />
            </>
          )}

          {method === 'regex' && (
            <>
              <label htmlFor="regex_pattern">Regex pattern to split on</label>
              <input
                id="regex_pattern"
                type="text"
                placeholder="For instance \n\n for empty lines"
                disabled={splitting}
                {...register('regex_pattern')}
              />
            </>
          )}

          {method === 'wtpsplit' && (
            <>
              <label htmlFor="granularity">Segmentation unit</label>
              <select id="granularity" disabled={splitting} {...register('granularity')}>
                <option value="sentence">Sentence</option>
                <option value="paragraph">Paragraph</option>
              </select>
              <label htmlFor="language">Language of the texts</label>
              <select id="language" disabled={splitting} {...register('language')}>
                {langages.map((lang) => (
                  <option key={lang.value} value={lang.value}>
                    {lang.label}
                  </option>
                ))}
              </select>
              <div className="explanations" style={{ fontSize: 'smaller', fontWeight: 'normal' }}>
                The segmentation runs on the server (GPU if available) and can take a while
              </div>
            </>
          )}

          <label htmlFor="min_chars">
            Minimum length of a text unit (shorter units are dropped)
          </label>
          <input
            id="min_chars"
            type="number"
            min={0}
            disabled={splitting}
            {...register('min_chars')}
          />

          <button type="submit" className="btn-submit" disabled={splitting}>
            Split the dataset
          </button>
          {splitting && (
            <UploadProgressBar
              progression={{
                loaded: progress !== null ? progress : undefined,
                total: 100,
              }}
              onCancel={handleStopSplit}
              statusMessage="Splitting the dataset"
            />
          )}
        </form>
      )}

      {result && result.status === 'done' && (
        <div className="mt-3">
          <div>
            New dataset : <b>{result.n_rows}</b> rows
          </div>
          <PreviewTable rows={(result.preview || []) as Record<string, string>[]} />
          <div className="d-flex align-items-center gap-2 my-2">
            <select
              value={exportFormat}
              onChange={(e) => setExportFormat(e.target.value)}
              style={{ width: 'auto' }}
            >
              <option value="csv">csv</option>
              <option value="xlsx">xlsx</option>
              <option value="parquet">parquet</option>
            </select>
            <button
              type="button"
              className="btn btn-secondary-action"
              onClick={handleDownload}
              disabled={downloading}
            >
              Export the prepared dataset
            </button>
            {downloading && <PulseLoader size={8} />}
          </div>
        </div>
      )}
    </div>
  );
};
