import { FC } from 'react';
import { Controller, SubmitHandler, useForm } from 'react-hook-form';
import Select from 'react-select';

import { useParams } from 'react-router-dom';
import { useUpdateProject } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { useAppContext } from '../../core/useAppContext';
import { ProjectUpdateModel } from '../../types';

export const ProjectUpdateForm: FC<{ closeModal: () => void }> = ({ closeModal }) => {
  const { projectName } = useParams();

  const {
    appContext: { currentProject: project },
  } = useAppContext();

  const columnsSelect =
    project && project.params.all_columns
      ? project.params.all_columns.map((e) => ({ value: e, label: e })) || []
      : [];
  const langages = [
    { value: 'en', label: 'English' },
    { value: 'fr', label: 'French' },
    { value: 'es', label: 'Spanish' },
    { value: 'de', label: 'German' },
    { value: 'cn', label: 'Chinese' },
    { value: 'ja', label: 'Japanese' },
  ];
  const {
    register,
    control,
    handleSubmit,
    formState: { isDirty },
  } = useForm<ProjectUpdateModel>({
    defaultValues: {
      project_name: project ? project.params.project_name : '',
      language: project ? project.params.language : '',
      cols_context: project ? project.params.cols_context : [],
      cols_text: project ? project.params.cols_text : [],
      add_n_train: 0,
    },
  });
  const { notify } = useNotifications();
  const updateProject = useUpdateProject(projectName || null);

  // action when form validated
  const onSubmit: SubmitHandler<ProjectUpdateModel> = async (formData) => {
    if (!formData.cols_text) {
      notify({ type: 'error', message: 'Please select a text column.' });
      return;
    }
    if (!isDirty) {
      notify({ type: 'info', message: 'No changes detected.' });
      return;
    }
    updateProject(formData);
    closeModal();
  };

  const maxElementsToAdd =
    project && project?.params?.n_total
      ? project?.params?.n_total -
        project.params.n_train -
        project.params.n_valid -
        project.params.n_test
      : 0;

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="form-frame">
      <label htmlFor="project_name">Change name</label>
      <input id="project_name" type="text" {...register('project_name')} />

      <label htmlFor="cols_text">Change text columns</label>
      <Controller
        name="cols_text"
        control={control}
        render={({ field: { onChange, value } }) => (
          <Select
            options={columnsSelect}
            isMulti
            value={
              value
                ? value
                    .map((v: string) => columnsSelect.find((opt) => opt.value === v))
                    .filter(Boolean)
                : []
            }
            onChange={(selectedOptions) => {
              onChange(selectedOptions ? selectedOptions.map((option) => option?.value) : []);
            }}
          />
        )}
      />

      <label htmlFor="language">
        Change language of the corpus (for tokenization and word segmentation)
      </label>
      <select id="language" {...register('language')}>
        {langages.map((lang) => (
          <option key={lang.value} value={lang.value}>
            {lang.label}
          </option>
        ))}
      </select>

      <label htmlFor="cols_context">Change contextual information columns</label>

      <Controller
        name="cols_context"
        control={control}
        render={({ field: { onChange, value } }) => (
          <Select
            options={columnsSelect}
            isMulti
            value={
              value
                ? value
                    .map((v: string) => columnsSelect.find((opt) => opt.value === v))
                    .filter(Boolean)
                : []
            }
            onChange={(selectedOptions) => {
              onChange(selectedOptions ? selectedOptions.map((option) => option?.value) : []);
            }}
          />
        )}
      />

      <label htmlFor="add_n_train">
        Add N elements in the train set (without stratification ; total : {maxElementsToAdd})
      </label>
      <input
        id="add_n_train"
        type="number"
        {...register('add_n_train')}
        max={maxElementsToAdd}
        min={0}
      />

      <button type="submit" className="btn-submit">
        Validate modifications
      </button>
    </form>
  );
};
