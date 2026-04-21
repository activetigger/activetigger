import { FC, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ProjectCard } from '../components/ProjectCard';

import { PageLayout } from '../components/layout/PageLayout';
import { useUserProjects } from '../core/api';

import { FaPlusCircle } from 'react-icons/fa';
import { useAppContext } from '../core/useAppContext';
import { AvailableProjectsModel } from '../types';

export const ProjectsPage: FC = () => {
  // hooks
  const {
    appContext: { currentProject, displayConfig },
    resetContext,
  } = useAppContext();
  const currentProjectSlug = currentProject?.params.project_slug;
  const canEdit = displayConfig.interfaceType !== 'annotator';
  // api call
  const { projects, storageUsed, storageLimit } = useUserProjects();

  // rows to display
  const [rows, setRows] = useState<AvailableProjectsModel[]>([]);
  useEffect(() => {
    setRows(projects || []);
  }, [projects]);

  // handle search input
  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    const searchValue = e.target.value.toLowerCase();

    setRows(
      (projects || []).filter((project) => {
        const projectName = project.parameters.project_name.toLowerCase();
        const projectSlug = project.parameters.project_slug.toLowerCase();
        const createdBy = project.created_by.toLowerCase();

        return (
          projectName.includes(searchValue) ||
          projectSlug.includes(searchValue) ||
          createdBy.includes(searchValue)
        );
      }),
    );
  };

  return (
    <PageLayout currentPage="projects">
      <div className="container-fluid">
        {
          <div className="row">
            <div className="col-0 col-lg-3" />
            <div className="col-12 col-lg-6">
              {canEdit && (
                <div className="w-100 d-flex align-items-center justify-content-center">
                  <Link
                    to="/projects/new"
                    className="btn w-75 mt-3 d-flex align-items-center justify-content-center fw-bold"
                    style={{
                      backgroundColor: '#ff9a3c',
                      border: 'none',
                      fontSize: '1.1rem',
                      letterSpacing: '0.5px',
                      color: 'white',
                    }}
                    onMouseOver={(e) => {
                      e.currentTarget.style.boxShadow = '0 6px 16px rgba(0,0,0,0.25)';
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.style.boxShadow = 'none';
                    }}
                  >
                    <FaPlusCircle className="me-2" size={22} />
                    <span
                      style={{
                        background: 'linear-gradient(90deg, #fff 0%, #ffe8cc 100%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                      }}
                    >
                      New Project
                    </span>
                  </Link>
                </div>
              )}

              {storageUsed !== null && storageLimit !== null && (
                <div className="mt-3 p-3 border rounded bg-light">
                  <div className="d-flex justify-content-between align-items-center mb-1">
                    <small className="fw-bold">Storage usage (projects you created)</small>
                    <small>
                      {storageUsed.toFixed(2)} GB / {storageLimit.toFixed(0)} GB
                    </small>
                  </div>
                  <div className="progress" style={{ height: '8px' }}>
                    <div
                      className={`progress-bar ${
                        storageUsed / storageLimit > 0.9
                          ? 'bg-danger'
                          : storageUsed / storageLimit > 0.7
                            ? 'bg-warning'
                            : 'bg-success'
                      }`}
                      role="progressbar"
                      style={{ width: `${Math.min((storageUsed / storageLimit) * 100, 100)}%` }}
                    />
                  </div>
                  {storageUsed / storageLimit > 0.9 && (
                    <small className="text-danger mt-1 d-block">
                      You are approaching your storage limit.
                    </small>
                  )}
                </div>
              )}

              <div className="project-list">
                <input
                  type="text"
                  className="form-control mt-3"
                  placeholder="Search for a project or a user"
                  onChange={handleSearch}
                />
                {rows.map((project) => (
                  <ProjectCard
                    project={project}
                    key={project.parameters.project_slug}
                    resetContext={
                      currentProjectSlug === project.parameters.project_slug
                        ? undefined
                        : resetContext
                    }
                  />
                ))}
              </div>
            </div>
          </div>
        }
      </div>
    </PageLayout>
  );
};
