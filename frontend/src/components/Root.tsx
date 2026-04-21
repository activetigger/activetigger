import { FC, useMemo } from 'react';
import { RouterProvider } from 'react-router-dom';

import { AuthProvider } from '../core/auth';
import { AppContextProvider } from '../core/context';
import { getRouter } from '../core/router';

const Root: FC = () => {
  const router = useMemo(() => getRouter(), []);

  return (
    // first app context as AuthProvider uses AppContext for notifications
    <AppContextProvider>
      <AuthProvider>
        <RouterProvider router={router} />
      </AuthProvider>
    </AppContextProvider>
  );
};

export default Root;
