import MDEditor from '@uiw/react-md-editor';
import { Dispatch, FC, SetStateAction, useEffect } from 'react';
import rehypeSanitize from 'rehype-sanitize';

interface CodebookManagementProps {
  codebook: string | null;
  time: string | null;
  modifiedCodebook: string | undefined;
  setModifiedCodebook: Dispatch<SetStateAction<string | undefined>>;
  saveCodebook: () => Promise<void>;
  callbackOnClose?: (arg: boolean) => void;
}

/**
 * TODO : there is a need to manage the multiuser
 */

export const CodebookManagement: FC<CodebookManagementProps> = ({
  codebook,
  time,
  modifiedCodebook,
  setModifiedCodebook,
  saveCodebook,
  callbackOnClose,
}) => {
  // update the text zone once (if undefined)
  useEffect(() => {
    if (codebook && modifiedCodebook === undefined) {
      setModifiedCodebook(codebook);
    }
  }, [codebook, modifiedCodebook, time, setModifiedCodebook]);

  const handleClose = () => {
    saveCodebook();
    if (callbackOnClose) {
      callbackOnClose(false);
    }
  };

  return (
    <div>
      <MDEditor
        value={modifiedCodebook || ''}
        onChange={setModifiedCodebook}
        previewOptions={{
          rehypePlugins: [[rehypeSanitize]],
        }}
      />
      <button className="btn-submit" onClick={handleClose}>
        Save
      </button>
    </div>
  );
};
