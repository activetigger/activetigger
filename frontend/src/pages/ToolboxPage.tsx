import { FC } from 'react';

import { DatasetPreparationForm } from '../components/forms/DatasetPreparationForm';
import { PageLayout } from '../components/layout/PageLayout';

export const ToolboxPage: FC = () => {
  return (
    <PageLayout>
      <div className="container">
        <div className="row justify-content-center">
          <div className="col-8">
            <DatasetPreparationForm />
          </div>
        </div>
      </div>
    </PageLayout>
  );
};
