import { FC, useEffect, useMemo, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { FaPlusCircle, FaRegTrashAlt } from 'react-icons/fa';

import { Modal } from 'react-bootstrap';
import { IoDuplicate } from 'react-icons/io5';
import { MdDriveFileRenameOutline } from 'react-icons/md';
import Select from 'react-select';
import { useAddScheme, useDeleteScheme, useDuplicateScheme, useRenameScheme } from '../core/api';
import { useNotifications } from '../core/notifications';
import { useAppContext } from '../core/useAppContext';
import { getRandomName } from '../core/utils';
import { SchemeModel } from '../types';
/*
 * Select current scheme
 */

interface SchemeManagementProps {
  projectSlug: string;
  canEdit?: boolean;
  username?: string | null;
}

export const SelectCurrentScheme: FC<{ username?: string | null }> = ({ username }) => {
  const { notify } = useNotifications();

  // get element from the context
  const {
    appContext: { currentProject, currentScheme, selectionConfig },
    setAppContext,
  } = useAppContext();

  const availableSchemes = useMemo(() => {
    return currentProject ? Object.keys(currentProject.schemes.available) : [];
  }, [currentProject]);

  // manage scheme selection
  useEffect(() => {
    // case of there is no selected scheme and schemes are available
    const nonDefaultSchemes = availableSchemes.filter((element) => element !== 'default');
    if (!currentScheme && nonDefaultSchemes.length > 0) {
      // if there is a previous scheme in the user history select it
      if (
        username &&
        currentProject?.users.last_schemes[username] &&
        nonDefaultSchemes.includes(currentProject.users.last_schemes[username])
      )
        setAppContext((state) => ({
          ...state,
          currentScheme: currentProject.users.last_schemes[username],
          activeModel: null,
        }));
      else
        setAppContext((state) => ({
          ...state,
          currentScheme: nonDefaultSchemes[0],
          activeModel: null,
        }));
      notify({
        type: 'success',
        message: `Scheme selected`,
      });
    }
    // case of the scheme have been deleted
    // if (availableSchemes[0] && currentScheme && !availableSchemes.includes(currentScheme)) {
    //   setAppContext((state) => ({
    //     ...state,
    //     currentScheme: availableSchemes[0],
    //   }));
    //   console.log('U');
    // }
  }, [currentScheme, availableSchemes, setAppContext, notify, currentProject, username]);

  // put the current scheme in the context on change
  const handleSelectScheme = (selectedOption: { value: string; label: string } | null) => {
    setAppContext((state) => ({
      ...state,
      currentScheme: selectedOption ? selectedOption.value : availableSchemes[0],
      activeModel: null,
      selectionConfig: {
        ...selectionConfig,
        mode: 'fixed',
      },
      history: [],
    }));
    notify({
      type: 'success',
      message: `Scheme ${selectedOption ? selectedOption.value : availableSchemes[0]} selected`,
    });
  };

  return (
    <div className="horizontal center" style={{ width: 'fit-content' }}>
      <span id="current-scheme-badge">Current Scheme</span>
      <div style={{ width: '200px' }}>
        <Select
          id="scheme-selected"
          className="flex-grow-1"
          options={availableSchemes.map((element) => ({
            value: element,
            label: element,
          }))}
          value={{
            value: currentScheme || availableSchemes[0],
            label: currentScheme || availableSchemes[0],
          }}
          onChange={handleSelectScheme}
        />
      </div>
    </div>
  );
};

/* Manage schemes
 * Select ; Delete ; Add
 */

export const SchemesManagement: FC<SchemeManagementProps> = ({
  projectSlug,
  canEdit,
  username,
}) => {
  // get element from the context
  const {
    appContext: { currentProject, currentScheme, reFetchCurrentProject },
    setAppContext,
  } = useAppContext();

  const availableSchemes = currentProject ? Object.keys(currentProject.schemes.available) : [];

  // hooks to use the objets
  const { register, handleSubmit } = useForm<SchemeModel>({
    defaultValues: {
      name: getRandomName('Scheme'),
    },
  });

  const { notify } = useNotifications();

  // hook to get the api call
  const addScheme = useAddScheme(projectSlug);
  const deleteScheme = useDeleteScheme(projectSlug, currentScheme || null);
  const renameScheme = useRenameScheme(projectSlug, currentScheme || '');
  const duplicateScheme = useDuplicateScheme(projectSlug, currentScheme || '');

  // state for displaying the new scheme menu
  const [showCreateNewScheme, setShowCreateNewScheme] = useState(false);
  const [showRename, setShowRename] = useState(false);
  const [showDelete, setShowDelete] = useState(false);
  const [newSchemeName, setNewSchemeName] = useState('New name');
  const [initialLabelsText, setInitialLabelsText] = useState('');

  // action to create the new scheme
  const createNewScheme: SubmitHandler<SchemeModel> = async (formData) => {
    try {
      const labels = initialLabelsText
        .split('\n')
        .map((l) => l.trim())
        .filter((l) => l.length > 0);
      await addScheme(formData.name, formData.kind || 'multiclass', labels);
      setInitialLabelsText('');
      if (reFetchCurrentProject) reFetchCurrentProject();
      notify({ type: 'success', message: `Scheme ${formData.name} created` });
      // set the new scheme as current
      setAppContext((state) => ({
        ...state,
        currentScheme: formData.name,
        activeModel: null,
      }));
    } catch (error) {
      notify({ type: 'error', message: error + '' });
    }
    setShowCreateNewScheme(false);
  };

  // action to delete scheme
  const deleteSelectedScheme = async () => {
    //TODO: try catch and throw
    await deleteScheme();

    // set another scheme
    const otherSchemes = availableSchemes.filter((element) => element !== currentScheme);
    setAppContext((state) => ({
      ...state,
      currentScheme: otherSchemes[0],
      activeModel: null,
    }));
    notify({ type: 'success', message: 'Scheme changed to ' + otherSchemes[0] });

    if (reFetchCurrentProject) reFetchCurrentProject();
  };

  return (
    <>
      <div
        className="horizontal center wrap"
        style={{ width: '80%', minWidth: '420', margin: '20px 0px' }}
      >
        {/* BUG: the box slides out of the screen for small screens */}
        <SelectCurrentScheme username={username} />
        {canEdit && (
          <div className="horizontal">
            <button
              onClick={() => setShowCreateNewScheme(!showCreateNewScheme)}
              className="btn btn-link ms-2 p-0"
              title="New scheme"
            >
              <FaPlusCircle size={20} />
            </button>
            <button
              onClick={() => setShowRename(!showRename)}
              className="btn btn-link p-0"
              title="Rename current scheme"
            >
              <MdDriveFileRenameOutline size={20} />
            </button>
            <button
              onClick={() => duplicateScheme()}
              className="btn btn-link p-0"
              title="Duplicate current scheme"
            >
              <IoDuplicate size={20} />
            </button>
            <button
              onClick={() => setShowDelete(!showDelete)}
              className="btn btn-link p-0"
              title="Delete current scheme"
            >
              <FaRegTrashAlt size={20} />
            </button>
          </div>
        )}
      </div>
      <Modal
        show={showCreateNewScheme}
        id="addschememodal"
        onHide={() => setShowCreateNewScheme(false)}
      >
        <Modal.Header closeButton>
          <Modal.Title>Add a new scheme</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <form onSubmit={handleSubmit(createNewScheme)}>
            <input
              id="scheme_name"
              type="text"
              {...register('name')}
              placeholder="Enter new scheme name"
            />
            <label htmlFor="scheme-selected">Type</label>
            <select id="scheme_kind" {...register('kind')}>
              <option value="multiclass">Multiclass</option>
              <option value="multilabel">
                Multilabel (experimental - only some features implemented)
              </option>
              <option value="span">Span (experimental - only annotation)</option>
            </select>
            <details style={{ marginTop: '10px' }}>
              <summary style={{ cursor: 'pointer' }}>Initial labels (optional)</summary>
              <textarea
                rows={5}
                style={{ width: '100%', marginTop: '5px' }}
                placeholder="Enter one label per line"
                value={initialLabelsText}
                onChange={(e) => setInitialLabelsText(e.target.value)}
              />
            </details>
            <button className="btn-submit">Create</button>
          </form>
        </Modal.Body>
      </Modal>
      <Modal show={showRename} id="renamescheme" onHide={() => setShowRename(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Rename the current scheme</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <input
            id="scheme_rename"
            type="text"
            placeholder="Enter new scheme name"
            value={newSchemeName}
            onChange={(e) => setNewSchemeName(e.target.value)}
            onFocus={(e) => e.target.select()}
          />
          <button
            className="btn-submit"
            onClick={async () => {
              await renameScheme(newSchemeName);
              setAppContext((state) => ({
                ...state,
                currentScheme: newSchemeName,
              }));
              if (reFetchCurrentProject) reFetchCurrentProject();
              setShowRename(false);
            }}
          >
            Rename
          </button>
        </Modal.Body>
      </Modal>
      <Modal show={showDelete} id="deletescheme" onHide={() => setShowDelete(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Delete the current scheme</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          Are you sure you want to delete the scheme <b>{currentScheme}</b>?
          <button
            className="btn-submit-danger"
            onClick={() => {
              deleteSelectedScheme();
              setShowDelete(false);
            }}
          >
            Delete
          </button>
        </Modal.Body>
      </Modal>
    </>
  );
};
