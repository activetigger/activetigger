import { FC } from 'react';

import { ProjectCreationForm } from '../components/forms/ProjectCreationForm';
import { PageLayout } from '../components/layout/PageLayout';

export const ProjectNewPage: FC = () => {
  return (
    <PageLayout>
      <div className="container">
        <div className="row justify-content-center">
          <div className="col-8">
            <ProjectCreationForm />
          </div>
        </div>
      </div>
    </PageLayout>
  );
};
