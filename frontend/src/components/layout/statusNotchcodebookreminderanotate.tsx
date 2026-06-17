/*
import { FC, useState } from 'react';
import MDEditor from '@uiw/react-md-editor';
import { useGetSchemeCodebook } from '../../core/api';

interface GuidelinesNotchProps {
  projectSlug: string | null;
  currentScheme: string | null;
  canEdit?: boolean;
}

export const GuidelinesNotch: FC<GuidelinesNotchProps> = ({ projectSlug, currentScheme }) => {
  const { codebook } = useGetSchemeCodebook(projectSlug, currentScheme);
  const [size, setSize] = useState<number>(300);

  return (
    <div id="status-notch-codebookreminder" style={{ height: `${size}px`, width: `${size + 100}px` }}>
      <div data-color-mode="light">
        <MDEditor.Markdown
          source={codebook || ''}
          style={{
            backgroundColor: 'transparent',
            fontSize: '0.85rem',
            lineHeight: '1.6',
            maxWidth: '100%',
          }}
        />
      </div>
    </div>
  );
}; */


import { FC, useState } from 'react';
import MDEditor from '@uiw/react-md-editor';
import { useGetSchemeCodebook } from '../../core/api';

interface GuidelinesNotchProps {
  projectSlug: string | null;
  currentScheme: string | null;
  canEdit?: boolean;
}

export const GuidelinesNotch: FC<GuidelinesNotchProps> = ({ projectSlug, currentScheme }) => {
  const { codebook } = useGetSchemeCodebook(projectSlug, currentScheme);
  const [isOpen, setIsOpen] = useState<boolean>(true);

  return (
    <>
      <button
        id="status-notch-codebookreminder-toggle"
        onClick={() => setIsOpen((o) => !o)}
      >
        {isOpen ? '📖 Guidelines ▼' : '📖 Guidelines ▲'}
      </button>

      {isOpen && (
        <div id="status-notch-codebookreminder">
          <div data-color-mode="light">
            <MDEditor.Markdown
              source={codebook || ''}
              style={{
                backgroundColor: 'transparent',
                fontSize: '0.85rem',
                lineHeight: '1.6',
                maxWidth: '100%',
              }}
            />
          </div>
        </div>
      )}
    </>
  );
};
