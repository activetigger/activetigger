import { FC } from 'react';
import { useForm } from 'react-hook-form';
import { useSendMessage } from '../../core/api';

type NewMessage = {
  content: string;
  kind: string;
  to_user?: string;
  to_project?: string;
};

export const SendMessage: FC = () => {
  const { sendMessage } = useSendMessage();

  const { handleSubmit, register, reset } = useForm<NewMessage>({
    defaultValues: {
      kind: 'system',
      content: '',
    },
  });

  const postForm = async function (data: NewMessage) {
    sendMessage(data.content, data.kind);
    reset();
  };

  return (
    <form onSubmit={handleSubmit(postForm)}>
      <label>
        Kind
        <select {...register('kind')}>
          <option value="system">System</option>
        </select>
      </label>
      <label>Message</label>
      <textarea placeholder="Message" {...register('content')} />
      <button type="submit" className="btn-submit">
        Send
      </button>
    </form>
  );
};
