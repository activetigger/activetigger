import { FC, useState } from 'react';
import { Button } from 'react-bootstrap';
import Modal from 'react-bootstrap/Modal';
import { FaRegTrashAlt } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';
import { useDeleteProject } from '../core/api';
import { AvailableProjectsModel } from '../types';

interface ProjectCardProps {
  project: AvailableProjectsModel;
  resetContext: (() => void) | undefined;
}

export const ProjectCard: FC<ProjectCardProps> = ({ project, resetContext }) => {
  const [showDelete, setShowDelete] = useState(false);
  const handleClose = () => setShowDelete(false);
  const handleShow = () => setShowDelete(true);
  const navigate = useNavigate();

  // function to delete project
  const deleteProject = useDeleteProject();
  const actionDelete = async () => {
    await deleteProject(project.parameters.project_slug);
    handleClose();
    navigate(0);
  };

  // function to navigate to the project page
  const navigateToProject = () => {
    if (resetContext) resetContext();
    navigate(`/projects/${project.parameters.project_slug}?fromProjectPage=true`);
  };

  return (
    <div key={project.parameters.project_slug} id="project-card">
      <div onClick={navigateToProject} className="clickable-zone">
        <h3 className="projecttitle">{project.parameters.project_name}</h3>
        <p className="projectdetails">
          <span id="key">Creator: </span>
          <span id="value">{project.created_by}</span>
        </p>
        <p className="projectdetails">
          <span id="key">Created at: </span>
          <span id="value">{project.created_at}</span>
        </p>

        <span className="badge info">memory {project.size} Mo</span>
      </div>
      <div onClick={handleShow} className="trash-wrapper">
        <FaRegTrashAlt size={20} />
      </div>
      <Modal show={showDelete} onHide={handleClose}>
        <Modal.Header>
          <Modal.Title>Delete the project</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          Do you really want to delete the project {project.parameters.project_name}?
        </Modal.Body>
        <Modal.Footer>
          <Button variant="danger" onClick={actionDelete} key="delete">
            Delete
          </Button>
          <Button variant="secondary" onClick={handleClose} key="cancel">
            Exit
          </Button>
        </Modal.Footer>
      </Modal>
    </div>
  );
};
