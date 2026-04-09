// Experimental image projects — see docs/image-projects-strategy.md
import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import { getProjectStatus, useAddProjectFile, useCreateProject } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { useAppContext } from '../../core/useAppContext';
import { getRandomName } from '../../core/utils';

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
  const { addProjectFile } = useAddProjectFile();
  const [submitting, setSubmitting] = useState(false);

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
          if (status === 'existing') {
            clearInterval(intervalId);
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
        Experimental: image projects support only a minimal workflow (no BERT, no regex). Max 5000
        images, 2 GB zip, 10 MB per image.
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
      <button type="submit" className="btn btn-primary mt-3" disabled={submitting}>
        {submitting ? 'Creating…' : 'Create image project'}
      </button>
    </form>
  );
};
