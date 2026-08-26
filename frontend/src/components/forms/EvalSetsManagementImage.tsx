import { FC, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SubmitHandler, useForm, useWatch } from 'react-hook-form';

import { Modal } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';

import { useCreateValidSetImage, useDropEvalSet, useStopProcesses } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { loadFile } from '../../core/utils';
import { useAppContext } from '../../core/useAppContext';

import { UploadProgressBar } from '../UploadProgressBar';

export interface EvalSetsManagementImageModel {
  projectSlug: string;
  currentScheme: string;
  dataset: 'test' | 'valid';
  exist: boolean;
}

interface FormShape {
  zipFiles: FileList;
  labelFiles?: FileList;
  col_id?: string;
  col_label?: string;
  n_eval?: number;
}

interface LabelsPreview {
  headers: string[];
  rowCount: number;
  filename: string;
}

/**
 * Image-project counterpart to EvalSetsManagement. Uploads an image zip and
 * an optional labels CSV/Parquet, then triggers AddEvalSetImage on the backend.
 */
export const EvalSetsManagementImage: FC<EvalSetsManagementImageModel> = ({
  projectSlug,
  currentScheme,
  dataset,
  exist,
}) => {
  const datasetCleanForPrinting = dataset === 'test' ? 'Test' : 'Validation';
  const { register, control, handleSubmit } = useForm<FormShape>();
  const {
    appContext: { currentProject },
  } = useAppContext();
  const proj_errors = useMemo(() => currentProject?.errors || [], [currentProject?.errors]);

  const { progression, createValidSet, cancel } = useCreateValidSetImage();
  const { notify } = useNotifications();
  const dropEvalSet = useDropEvalSet(projectSlug);
  const { stopProcesses } = useStopProcesses(projectSlug);
  const navigate = useNavigate();

  const [alertDrop, setAlertDrop] = useState<boolean>(false);
  const [labelsPreview, setLabelsPreview] = useState<LabelsPreview | null>(null);

  const labelFiles = useWatch({ control, name: 'labelFiles' });
  const add_eval_storageKey = `add-evalset-${dataset}-${projectSlug}`;

  const [uploading, setUploading] = useState<boolean>(
    () => sessionStorage.getItem(add_eval_storageKey) === 'true',
  );
  const uploadingRef = useRef(uploading);
  const cancelRef = useRef(cancel);
  const [displayCancel, setDisplayCancel] = useState<AbortController | undefined>(undefined);
  const errorCountAtSubmit = useRef(0);
  const maxDuration = 5 * 60 * 1000;

  const isUploading = useCallback(
    (val: boolean) => {
      if (!val) cancelRef.current = undefined;
      val
        ? sessionStorage.setItem(add_eval_storageKey, 'true')
        : sessionStorage.removeItem(add_eval_storageKey);
      setUploading(val);
    },
    [add_eval_storageKey],
  );

  useEffect(() => {
    uploadingRef.current = uploading;
  }, [uploading]);

  useEffect(() => {
    if (exist && uploadingRef.current) isUploading(false);
  }, [exist, isUploading]);

  // surface backend errors raised during the task
  useEffect(() => {
    if (!uploading) return;
    const len_new_errors = proj_errors.length - errorCountAtSubmit.current;
    if (len_new_errors > 0) {
      const newErrors = proj_errors.slice(-len_new_errors);
      const evalsetError = [...newErrors]
        .reverse()
        .find((e) => Array.isArray(e) && (e[0] as string).includes(`add_evalset_${dataset}`));
      if (evalsetError) {
        notify({ type: 'error', message: (evalsetError as string[]).join('-') });
        isUploading(false);
      }
    }
  }, [proj_errors, uploading, dataset, notify, isUploading]);

  useEffect(() => {
    if (!uploading) return;
    const timer = setTimeout(() => {
      console.warn('Upload timeout reached');
      isUploading(false);
    }, maxDuration);
    return () => clearTimeout(timer);
  }, [uploading, isUploading, maxDuration]);

  useEffect(() => {
    cancelRef.current = cancel;
    setDisplayCancel(cancel);
  }, [cancel]);

  useEffect(() => {
    if (!uploading) return;
    const stop = () => {
      stopProcesses(`add_evalset_${dataset}`);
      isUploading(false);
      setTimeout(() => navigate(0), 750);
    };
    if (cancel?.signal) {
      const signal = cancel.signal;
      signal.addEventListener('abort', stop);
      return () => signal.removeEventListener('abort', stop);
    }
    const n_cancel = new AbortController();
    cancelRef.current = n_cancel;
    setDisplayCancel(n_cancel);
    n_cancel.signal.addEventListener('abort', stop);
    return () => n_cancel.signal.removeEventListener('abort', stop);
  }, [cancel, uploading, dataset, stopProcesses, navigate, isUploading]);

  // preview labels file locally to populate col_id / col_label dropdowns
  useEffect(() => {
    if (labelFiles && labelFiles.length > 0) {
      const file = labelFiles[0];
      loadFile(file).then((data) => {
        if (data === null) {
          notify({ type: 'error', message: 'Error reading the labels file' });
          setLabelsPreview(null);
          return;
        }
        setLabelsPreview({
          headers: data.headers,
          rowCount: Math.max(0, data.data.length - 1),
          filename: data.filename,
        });
      });
    } else {
      setLabelsPreview(null);
    }
  }, [labelFiles, notify]);

  const labelColumnOptions = labelsPreview?.headers.map((h) => (
    <option key={h} value={h}>
      {h}
    </option>
  ));

  const onSubmit: SubmitHandler<FormShape> = async (formData) => {
    if (!formData.zipFiles || formData.zipFiles.length === 0) {
      notify({ type: 'error', message: 'Please select a .zip archive of images' });
      return;
    }
    const zipFile = formData.zipFiles[0];
    if (!zipFile.name.toLowerCase().endsWith('.zip')) {
      notify({ type: 'error', message: 'Only .zip archives are accepted' });
      return;
    }
    const labelsFile =
      formData.labelFiles && formData.labelFiles.length > 0 ? formData.labelFiles[0] : undefined;
    if (labelsFile && !formData.col_id) {
      notify({ type: 'error', message: 'Please choose the ID column in the labels file' });
      return;
    }

    errorCountAtSubmit.current = proj_errors.length;
    isUploading(true);
    try {
      const ok = await createValidSet(projectSlug, dataset, {
        zipFile,
        labelsFile,
        scheme: currentScheme || null,
        col_id: formData.col_id || null,
        col_label: formData.col_label || null,
        n_eval: formData.n_eval ? Number(formData.n_eval) : null,
      });
      if (!ok) isUploading(false);
    } catch {
      isUploading(false);
      notify({ type: 'error', message: 'Failed to start the process' });
    }
  };

  const capFirstLetter = (word: string) => word.charAt(0).toUpperCase() + word.slice(1);

  return (
    <div>
      <h4 className="subsection">{capFirstLetter(dataset)} set (images)</h4>
      {exist && (
        <button className="btn-drop-dataset" onClick={() => setAlertDrop(true)}>
          Drop {datasetCleanForPrinting} set
        </button>
      )}

      {!exist && (
        <form onSubmit={handleSubmit(onSubmit)}>
          <div className="col-lg-6">
            <div className="explanations">
              No {datasetCleanForPrinting} data set has been created. Upload a .zip of images;
              optionally add a labels file (CSV/Parquet) whose ID column matches the image filename
              stems. Corrupt or unreadable images are dropped at import. IDs are prefixed with
              "imported-" to keep them distinct from the train pool.
            </div>
            <label htmlFor="imageZip">Images (.zip)</label>
            <input
              id="imageZip"
              className="form-control"
              type="file"
              accept=".zip"
              {...register('zipFiles', { required: true })}
            />

            <label htmlFor="labelsFile" className="mt-2">
              Labels file (optional, .csv / .parquet / .xlsx)
            </label>
            <input
              id="labelsFile"
              className="form-control"
              type="file"
              accept=".csv,.parquet,.xlsx"
              {...register('labelFiles')}
            />

            {labelsPreview && (
              <div className="mt-2">
                <div>
                  Labels file: <b>{labelsPreview.filename}</b> ({labelsPreview.rowCount} rows)
                </div>
                <label htmlFor="col_id" className="mt-2">
                  ID column (must match image filenames without extension)
                </label>
                <select id="col_id" {...register('col_id')}>
                  <option value="">— select —</option>
                  {labelColumnOptions}
                </select>
                <label htmlFor="col_label" className="mt-2">
                  Label column (optional, values must already exist in the current scheme)
                </label>
                <select id="col_label" {...register('col_label')}>
                  <option value="">No label</option>
                  {labelColumnOptions}
                </select>
              </div>
            )}

            <label htmlFor="n_eval" className="mt-2">
              Max images to import (optional)
            </label>
            <input id="n_eval" type="number" {...register('n_eval')} />

            <button type="submit" className="btn-submit mt-2" disabled={uploading}>
              {uploading ? 'Uploading…' : 'Create'}
            </button>
          </div>
        </form>
      )}

      {uploading && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            zIndex: 10,
          }}
        >
          <UploadProgressBar progression={progression} cancel={cancel || displayCancel} />
        </div>
      )}

      <Modal show={alertDrop} onHide={() => setAlertDrop(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Drop the {datasetCleanForPrinting} set</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          Do you really want to drop the {datasetCleanForPrinting} set? Eval images and thumbnails
          will be removed, and features / quick models will be reset.
          <div className="horizontal">
            <button onClick={() => setAlertDrop(false)} style={{ flex: '1 1 auto' }}>
              Cancel
            </button>
            <button
              className="btn-danger"
              onClick={() => {
                dropEvalSet(dataset).then(() => navigate(0));
              }}
              style={{ flex: '1 1 auto' }}
            >
              Confirm
            </button>
          </div>
        </Modal.Body>
      </Modal>
    </div>
  );
};
