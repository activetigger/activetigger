import { FC } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useChangeEmail } from '../../core/api';

type EmailForm = {
  email: string;
  password: string;
};

export const ChangeEmail: FC<{ currentEmail?: string | null; onSuccess?: () => void }> = ({
  currentEmail,
  onSuccess,
}) => {
  const { changeEmail } = useChangeEmail();
  const { handleSubmit, register, reset } = useForm<EmailForm>({
    defaultValues: { email: currentEmail ?? '', password: '' },
  });

  const onSubmit: SubmitHandler<EmailForm> = async (data) => {
    const ok = await changeEmail(data.email, data.password);
    if (ok) {
      reset({ email: data.email, password: '' });
      onSuccess?.();
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div className="mb-2">
        <label className="form-label">New email address</label>
        <input
          type="email"
          className="form-control"
          placeholder="name@example.com"
          required
          {...register('email', { required: true })}
        />
      </div>
      <div className="mb-2">
        <label className="form-label">Current password</label>
        <input
          type="password"
          className="form-control"
          placeholder="Confirm with your password"
          required
          {...register('password', { required: true })}
        />
      </div>
      <button type="submit" className="btn-submit">
        Update email
      </button>
    </form>
  );
};
