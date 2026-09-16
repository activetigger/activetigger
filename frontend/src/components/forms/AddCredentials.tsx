import { FC } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useAddGenCredentials } from '../../core/api';
import { GenCredentialsInput } from '../../types';

/**
 * Save an OpenAI-compatible endpoint/key pair for the current user.
 * The entry is tested against the endpoint when saved.
 */
export const AddCredentials: FC<{ onSuccess?: () => void }> = ({ onSuccess }) => {
  const { addGenCredentials } = useAddGenCredentials();
  const { handleSubmit, register, reset } = useForm<GenCredentialsInput>({
    defaultValues: { name: '', endpoint: '', api_key: '' },
  });

  const onSubmit: SubmitHandler<GenCredentialsInput> = async (data) => {
    const saved = await addGenCredentials(data);
    if (saved) {
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
        <label className="form-label">Endpoint (OpenAI-compatible base URL)</label>
        <input
          type="text"
          className="form-control"
          placeholder="e.g. https://openrouter.ai/api/v1"
          required
          {...register('endpoint', { required: true })}
        />
      </div>
      <div className="mb-2">
        <label className="form-label">API key (optional for local servers)</label>
        <input
          type="password"
          className="form-control"
          placeholder="Stored encrypted, never displayed again"
          autoComplete="off"
          {...register('api_key')}
        />
      </div>
      <button type="submit" className="btn-submit">
        Save & test credentials
      </button>
    </form>
  );
};
