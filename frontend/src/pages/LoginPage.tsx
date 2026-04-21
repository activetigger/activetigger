import { FC } from 'react';
import { useLocation } from 'react-router-dom';

import { LoginForm } from '../components/forms/LoginForm';
import { PageLayout } from '../components/layout/PageLayout';
import { useAuth } from '../core/useAuth';

export const LoginPage: FC = () => {
  const { authenticatedUser } = useAuth();
  const { state } = useLocation();

  return (
    <PageLayout currentPage="login">
      <div className="container-fluid">
        {authenticatedUser?.username && (
          <div className="user-info text-center mb-3">
            You're logged in as <span>{authenticatedUser.username}</span> (status:{' '}
            {authenticatedUser.status})
          </div>
        )}

        <div className="row">
          <div className="col-11 col-md-8 col-lg-5 mx-auto">
            <div className="subsection text-center mb-3">Change account</div>
            <LoginForm redirectTo={state?.path || '/projects'} />
          </div>
        </div>
      </div>
    </PageLayout>
  );
};
