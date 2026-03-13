import { Middleware } from 'openapi-fetch';
import { FC, PropsWithChildren, useCallback, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import { api } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { useAuth } from '../../core/useAuth';

/**
 * AuthRequired
 * a component which protect private routes from access by unauthenticated user
 * @returns children or a redirection to the login form
 */
export const APIMiddlewares: FC<PropsWithChildren> = ({ children }) => {
  // first we get authenticated user state
  const { authenticatedUser } = useAuth();
  // location is provided by react-router library and contains the current page (i.e. in which URL path this component has been mounted)
  const location = useLocation();
  const navigate = useNavigate();
  const { notify } = useNotifications();

  const redirectToLogin = useCallback(
    (message: string) => {
      notify({
        type: 'error',
        message,
      });
      setTimeout(
        () => navigate('/login', { state: { path: location.pathname }, replace: true }),
        500,
      );
    },
    [navigate, notify, location],
  );

  useEffect(() => {
    const apiErrorMiddleware: Middleware = {
      // on response check if session is correct
      onResponse: async ({ response }) => {
        const clonedResponse = response.clone();

        // if session is expired or invalid we catch the 401 and redirect to login page
        if ([401, 403].includes(response.status)) {
          redirectToLogin('Invalid user session: redirecting you to login page...');
        } else {
          if (response.status !== 200) {
            //TODO : check error body is correct
            const { body, ...resOptions } = response;
            const message = await clonedResponse.json();
            notify({
              type: 'error',
              message:
                'detail' in message ? JSON.stringify(message.detail) : JSON.stringify(message),
            });

            // STILL AN ERROR TO FIX TODO
            return new Response(body, { ...resOptions, status: 500 });
          }
        }
        return clonedResponse;
      },
    };
    if (!authenticatedUser) {
      redirectToLogin('you must authenticate to view this page.');
    }
    api.use(apiErrorMiddleware);
    return () => {
      api.eject(apiErrorMiddleware);
    };
  }, [authenticatedUser, notify, redirectToLogin]);

  if (authenticatedUser)
    // if the user is currently authenticated just let him go through its destination by mounting children (i.e. the requested page component)
    return children;
  else {
    // else ask him to first login and then redirects him back to where we wanted to go
    return null;
  }
};
