// Experimental image projects — see docs/image-projects-strategy.md
import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import {
  getProjectStatus,
  useAddFeature,
  useAddProjectFile,
  useCreateProject,
} from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { useAppContext } from '../../core/useAppContext';
import { getRandomName } from '../../core/utils';
import { UploadProgressBar } from '../UploadProgressBar';

interface ImageFormValues {
  project_name: string;
  n_train: number;
  n_test: number;
  n_valid: number;
  files: FileList;
}

export const ProjectCreationFormImagexp: FC = () => {
  const { register, handleSubmit } = useForm<ImageFormValues>({
    defaultValues: { project_name: getRandomName('Project'), n_train: 100, n_test: 0, n_valid: 0 },
  });
  const navigate = useNavigate();
  const { notify } = useNotifications();
  const { resetContext } = useAppContext();
  const createProject = useCreateProject();
  const { addProjectFile, progression, cancel } = useAddProjectFile();
  const addFeature = useAddFeature();
  const [submitting, setSubmitting] = useState(false);
  const [creationProgress, setCreationProgress] = useState<number | null>(null);
  const [computeEmbeddings, setComputeEmbeddings] = useState(true);
  const [batchSize, setBatchSize] = useState(16);

  const onSubmit: SubmitHandler<ImageFormValues> = async (data) => {
    if (!data.files || data.files.length === 0) {
      notify({ type: 'error', message: 'Please select a .zip archive of images' });
      return;
    }
    const file = data.files[0];
    if (!file.name.toLowerCase().endsWith('.zip')) {
      notify({ type: 'error', message: 'Only .zip archives are accepted for image projects' });
      return;
    }
    setSubmitting(true);
    try {
      // upload the zip (kind=image switches the backend validation)
      await addProjectFile(data.project_name, file, 'image');
      const slug = await createProject({
        // the kind field is persisted in parameters JSON
        // (see ProjectBaseModel in api/activetigger/datamodels.py)
        kind: 'image',
        project_name: data.project_name,
        filename: file.name,
        n_train: Number(data.n_train),
        n_test: Number(data.n_test),
        n_valid: Number(data.n_valid),
        cols_text: [],
        col_id: '',
        language: 'en',
      } as unknown as Parameters<typeof createProject>[0]);

      // wait for the project to actually be created before navigating
      // (mirrors the text-project flow in ProjectCreationForm.tsx)
      const maxDuration = 5 * 60 * 1000;
      const startTime = Date.now();
      const intervalId = setInterval(async () => {
        try {
          const status = await getProjectStatus(slug);
          if (status?.startsWith('error') || status === 'not existing') {
            clearInterval(intervalId);
            const errorDetail = status?.startsWith('error:')
              ? status.slice('error:'.length).trim()
              : null;
            notify({
              type: 'error',
              message: errorDetail
                ? `Project creation failed: ${errorDetail}`
                : 'Project creation failed.',
            });
            setSubmitting(false);
            navigate('/projects');
            return;
          }
          // Parse creation progress (e.g. "creating:45.2")
          if (status?.startsWith('creating:')) {
            const pct = parseFloat(status.split(':')[1]);
            if (!isNaN(pct)) setCreationProgress(pct);
            return;
          }
          if (status === 'creating') {
            return;
          }
          if (status === 'existing') {
            clearInterval(intervalId);
            // Fire image embedding computation from frontend, same pattern
            // as text projects do with sentence-embeddings.
            if (computeEmbeddings) {
              addFeature(slug, 'image-embeddings', 'default', true, {
                model: 'generic',
                batch_size: batchSize,
              });
            }
            resetContext();
            setSubmitting(false);
            navigate(`/projects/${slug}?fromCreatePage=true`);
            return;
          }
          if (Date.now() - startTime >= maxDuration) {
            clearInterval(intervalId);
            notify({
              type: 'error',
              message: 'Timeout during the creation of the project. Try again later.',
            });
            setSubmitting(false);
            navigate('/projects');
          }
        } catch (err) {
          console.error('Error polling project status:', err);
          clearInterval(intervalId);
          setSubmitting(false);
        }
      }, 1000);
    } catch (e) {
      notify({ type: 'error', message: String(e) });
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div className="alert alert-warning small">
        Experimental: image projects support only a minimal workflow (no BERT, no regex). Max 1 GB
        zip, 100 MB per image.
      </div>
      <label className="form-label">Project name</label>
      <input className="form-control mb-2" {...register('project_name', { required: true })} />
      <label className="form-label">Images (.zip)</label>
      <input className="form-control mb-2" type="file" accept=".zip" {...register('files')} />
      <div className="row">
        <div className="col">
          <label className="form-label">Train size</label>
          <input className="form-control" type="number" {...register('n_train')} />
        </div>
        <div className="col">
          <label className="form-label">Test size</label>
          <input className="form-control" type="number" {...register('n_test')} />
        </div>
        <div className="col">
          <label className="form-label">Valid size</label>
          <input className="form-control" type="number" {...register('n_valid')} />
        </div>
      </div>
      <details className="mt-3">
        <summary>Advanced parameters</summary>
        <div className="d-flex align-items-center gap-2 mt-2">
          <label htmlFor="compute_embeddings" className="m-0">
            <input
              id="compute_embeddings"
              type="checkbox"
              disabled={submitting}
              checked={computeEmbeddings}
              onChange={() => setComputeEmbeddings(!computeEmbeddings)}
            />{' '}
            Compute image embeddings
          </label>
          {computeEmbeddings && (
            <label className="batch-size-label">
              batch
              <input
                type="number"
                min={1}
                max={512}
                value={batchSize}
                onChange={(e) => setBatchSize(Math.max(1, parseInt(e.target.value) || 1))}
                title="Batch size for embedding computation"
                disabled={submitting}
                style={{ width: 60, marginLeft: 4 }}
              />
            </label>
          )}
        </div>
      </details>
      <button type="submit" className="btn btn-primary mt-3" disabled={submitting}>
        {submitting ? 'Creating…' : 'Create image project'}
      </button>
      {submitting && <UploadProgressBar progression={progression} cancel={cancel} />}
      {submitting && creationProgress !== null && (
        <div className="mt-3 text-muted">
          <div className="d-flex align-items-center gap-2 mb-2">
            <div className="spinner-border spinner-border-sm" role="status" />
            <span>
              Extracting images and generating thumbnails ({Math.round(creationProgress)}%)...
            </span>
          </div>
          <div className="progress" style={{ height: 8 }}>
            <div
              className="progress-bar"
              role="progressbar"
              style={{ width: `${creationProgress}%` }}
            />
          </div>
        </div>
      )}
    </form>
  );
};
