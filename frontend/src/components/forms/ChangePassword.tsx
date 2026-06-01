import { FC } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useChangePassword } from '../../core/api';

type PasswordForm = {
  pwdOld: string;
  pwd1: string;
  pwd2: string;
};

export const ChangePassword: FC<{ onSuccess?: () => void }> = ({ onSuccess }) => {
  const { changePassword } = useChangePassword();

  const { handleSubmit, register, reset } = useForm<PasswordForm>({});
  const onSubmit: SubmitHandler<PasswordForm> = async (data) => {
    const ok = await changePassword(data.pwdOld, data.pwd1, data.pwd2);
    reset();
    if (ok) onSuccess?.();
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div className="mb-2">
        <label className="form-label">Old password</label>
        <input
          type="password"
          className="form-control"
          placeholder="Old password"
          required
          {...register('pwdOld', { required: true })}
        />
      </div>
      <div className="mb-2">
        <label className="form-label">New password</label>
        <input
          type="password"
          className="form-control"
          placeholder="New password"
          required
          {...register('pwd1', { required: true })}
        />
      </div>
      <div className="mb-2">
        <label className="form-label">Confirm new password</label>
        <input
          type="password"
          className="form-control"
          placeholder="Confirm new password"
          required
          {...register('pwd2', { required: true })}
        />
      </div>
      <button type="submit" className="btn-submit">
        Update password
      </button>
    </form>
  );
};
