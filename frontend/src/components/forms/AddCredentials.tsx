import { FC, useEffect, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useAddUserCredentials, useGetGenModels } from '../../core/api';
import { GenerationModelApi } from '../../types';

type CredentialsForm = {
  name: string;
  api: string;
  endpoint: string;
  credentials: string;
};

export const AddCredentials: FC<{ onSuccess?: () => void }> = ({ onSuccess }) => {
  const { addUserCredentials } = useAddUserCredentials();
  const { models } = useGetGenModels();
  const [availableAPIs, setAvailableAPIs] = useState<GenerationModelApi[]>([]);
  const { handleSubmit, register, reset } = useForm<CredentialsForm>({
    defaultValues: { name: '', api: '', endpoint: '', credentials: '' },
  });

  useEffect(() => {
    const fetchModels = async () => {
      setAvailableAPIs(await models());
    };
    fetchModels();
  }, [models]);

  const onSubmit: SubmitHandler<CredentialsForm> = async (data) => {
    const ok = await addUserCredentials({
      name: data.name,
      api: data.api,
      endpoint: data.endpoint || null,
      credentials: data.credentials,
    });
    if (ok) {
      reset();
      onSuccess?.();
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div className="mb-2">
        <label className="form-label">Name</label>
        <input
          type="text"
          className="form-control"
          placeholder="e.g. my OpenRouter key"
          required
          {...register('name', { required: true })}
        />
      </div>
      <div className="mb-2">
        <label className="form-label">API</label>
        <select className="form-select" required {...register('api', { required: true })}>
          <option value="">Select an API</option>
          {availableAPIs.map((api) => (
            <option key={api.name} value={api.name}>
              {api.name}
            </option>
          ))}
        </select>
      </div>
      <div className="mb-2">
        <label className="form-label">Endpoint (optional)</label>
        <input
          type="text"
          className="form-control"
          placeholder="e.g. https://api.example.com/v1"
          {...register('endpoint')}
        />
      </div>
      <div className="mb-2">
        <label className="form-label">API key</label>
        <input
          type="password"
          className="form-control"
          placeholder="Stored encrypted, never displayed again"
          autoComplete="off"
          required
          {...register('credentials', { required: true })}
        />
      </div>
      <button type="submit" className="btn-submit">
        Save credentials
      </button>
    </form>
  );
};
