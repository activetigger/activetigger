import { FC } from 'react';
import { ChangePassword } from '../components/forms/ChangePassword';
import { PageLayout } from '../components/layout/PageLayout';
import { useAuth } from '../core/useAuth';

export const AccountPage: FC = () => {
  const { authenticatedUser } = useAuth();

  return (
    <PageLayout currentPage="account">
      <div className="container">
        {authenticatedUser?.username && (
          <div className="row">
            <div className="col-0 col-sm-2 col-md-3" />
            <div className="col-12 col-sm-8 col-md-6 ">
              <div className="user-info">
                You're logged in as <span>{authenticatedUser.username}</span> ( status :{' '}
                {authenticatedUser.status})
              </div>
              <ChangePassword />
            </div>
            <div className="col-0 col-sm-2 col-md-3" />
          </div>
        )}
      </div>
    </PageLayout>
  );
};
