import { FC, useState } from 'react';
import Modal from 'react-bootstrap/Modal';
import { useNavigate } from 'react-router-dom';
import { useDeleteProject } from '../core/api';
import { ProjectStateModel } from '../types';
import { ProjectUpdateForm } from './forms/ProjectUpdateForm';
import { ModelParametersTab } from './ModelParametersTab';

export interface ProjectParametersModel {
  project: ProjectStateModel;
  projectSlug: string;
}

export const ProjectParameters: FC<ProjectParametersModel> = ({ project, projectSlug }) => {
  const navigate = useNavigate();

  // modals to delete
  const [show, setShow] = useState(false);
  const handleClose = () => setShow(false);
  const handleShow = () => setShow(true);

  // show modify
  const [showModify, setShowModify] = useState(false);

  // function to delete project
  const deleteProject = useDeleteProject();
  const actionDelete = async () => {
    if (projectSlug) {
      await deleteProject(projectSlug);
      navigate(`/projects/`);
    }
  };

  return (
    <>
      <div className="explanations">Parameters of this project</div>

      <ModelParametersTab
        params={
        {
          'Project Name': project.params.project_name,
          Filename: project.params.filename ?? 'N/A',
          Language: project.params.language,
          'Text Columns': Array.isArray(project.params.cols_text)
            ? project.params.cols_text.join(', ')
            : project.params.cols_text,
          'Column ID': project.params.col_id,
          'Context Columns':
            project.params.cols_context.length > 0
              ? project.params.cols_context.join(', ')
              : 'None',
          'Label Columns':
            project.params.cols_label.length > 0
              ? project.params.cols_label.join(', ')
              : 'None',
          'Total Rows': project.params.n_total ?? 'N/A',
          'Rows in train set': project.params.n_train,
          'Rows in test set': project.params.test ? project.params.n_test : 'Empty',
          'Rows in valid set': project.params.valid ? project.params.n_valid : 'Empty',
          'Skipped Rows': project.params.n_skip,
          'Train Selection': project.params.train_selection,
          'Holdout Selection': project.params.holdout_selection ?? 'None',
          'Stratification Columns':
            project.params.cols_stratify.length > 0
              ? project.params.cols_stratify.join(', ')
              : 'None',
          'From Project': project.params.from_project ?? 'None',
          'From Toy Dataset': project.params.from_toy_dataset ? 'Yes' : 'No',
          'Embeddings':
            project.params.embeddings.length > 0
              ? project.params.embeddings.join(', ')
              : 'None',
          Seed: project.params.seed,
        } as Record<string, unknown>
      }
            />
      <Modal show={showModify} onHide={() => setShowModify(false)} id="addfeature-modal">
        <Modal.Header closeButton>
          <Modal.Title>Change project parameters</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <div className="alert alert-warning">
            <strong>Note:</strong> Changing project parameters may affect your project : adding
            elements will delete features and quick models
          </div>
          <ProjectUpdateForm closeModal={() => setShowModify(false)} />
        </Modal.Body>
      </Modal>

      <div className="horizontal wrap">
        <button className="btn-primary-action" onClick={() => setShowModify(!showModify)}>
          Change parameters
        </button>
        <button onClick={handleShow} className="btn-danger">
          Delete project
        </button>
      </div>

      <div>
        <Modal show={show} onHide={handleClose}>
          <Modal.Header>
            <Modal.Title>Delete the project</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            Do you really want to delete this project
            <div className="horizontal">
              <button onClick={handleClose} style={{ flex: '1 1 auto' }}>
                Cancel
              </button>
              <button className="btn-danger" onClick={actionDelete} style={{ flex: '1 1 auto' }}>
                Confirm
              </button>
            </div>
          </Modal.Body>
        </Modal>
      </div>
    </>
  );
};
