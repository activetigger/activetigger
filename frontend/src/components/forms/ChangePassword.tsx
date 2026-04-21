import { FC } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useChangePassword } from '../../core/api';

type PasswordForm = {
  pwdOld: string;
  pwd1: string;
  pwd2: string;
};

export const ChangePassword: FC = () => {
  const { changePassword } = useChangePassword();

  const { handleSubmit, register, reset } = useForm<PasswordForm>({});
  const onSubmit: SubmitHandler<PasswordForm> = async (data) => {
    changePassword(data.pwdOld, data.pwd1, data.pwd2);
    reset();
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div className="subsection">Change password for the current user</div>
      <input type="password" placeholder="Old password" {...register('pwdOld')} />
      <input type="password" placeholder="New password" {...register('pwd1')} />
      <input type="password" placeholder="Confirm new password" {...register('pwd2')} />
      <button className="btn-submit">Valid</button>
    </form>
  );
};
