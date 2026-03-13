import { FC, PropsWithChildren, useCallback, useState } from 'react';

import { LoginParams, UserModel } from '../types';
import { HttpError } from './HTTPError';
import { login, logout, me } from './api';
import { useNotifications } from './notifications';
import { DEFAULT_CONTEXT, useAppContext } from './useAppContext';
import { authContext } from './useAuth';

// Information about the current authenticated user
export type AuthenticatedUser = UserModel & {
  access_token?: string;
};
// the global auth context which provide the authenticated user and the login internal logic
export type AuthContext = {
  authenticatedUser?: AuthenticatedUser;
  login: (params: LoginParams) => Promise<void>;
  logout: () => Promise<boolean>;
};

// internal hook which must not be used outside the context
const _useAuth = (): AuthContext => {
  // we use localstorage to persist session
  const storedAuth = localStorage.getItem('activeTigger.auth');
  // TODO check for session deprecation

  const { setAppContext } = useAppContext();

  // internal state to store the current authenticated user
  const [authenticatedUser, setAuthenticatedUser] = useState<AuthenticatedUser | undefined>(
    // by default we load the local storage version
    storedAuth ? JSON.parse(storedAuth) : undefined,
  );

  // notifications
  const { notify } = useNotifications();

  /**
   * This method wraps the login API call into our authenticated user state management
   * It does the call and make sure to update our internal state accordingly
   * by providing this method we encapsulate the setter and make sure to control all updates
   */
  const _login = useCallback(
    async (params: LoginParams) => {
      try {
        const response = await login(params);
        if (response.access_token) {
          const user = await me(response.access_token);

          if (user !== undefined) {
            const authUser = { ...user, access_token: response.access_token };
            localStorage.setItem('activeTigger.auth', JSON.stringify(authUser));
            setAuthenticatedUser((previousAuthUser) => {
              // before setting new user, if app had previously another user reset the app context
              if (previousAuthUser && previousAuthUser.username !== user.username) {
                //reset context
                setAppContext(DEFAULT_CONTEXT);
              }
              // force the type of interface
              setAppContext((appContext) => {
                const interfaceType = user.status === 'annotator' ? 'annotator' : 'default';
                return {
                  ...appContext,
                  displayConfig: { ...appContext.displayConfig, interfaceType },
                };
              });

              return authUser;
            });
            notify({ type: 'success', message: `Logged in as ${user.username}` });
          } else {
            throw new Error('Good token but no user?');
          }
        }
      } catch (error) {
        notify({ type: 'warning', message: String(error) });
        localStorage.removeItem('activeTigger.auth');
        setAuthenticatedUser(undefined);
      }
    },
    // the method code will change if setAuthenticatedUser changes which happens only at init
    [setAuthenticatedUser, notify, setAppContext],
  );

  const _logout = useCallback(
    async () => {
      if (authenticatedUser && authenticatedUser.access_token) {
        try {
          const success = await logout(authenticatedUser.access_token);
          if (success) {
            localStorage.removeItem('activeTigger.auth');
            //Reset context
            setAppContext(DEFAULT_CONTEXT);
            setAuthenticatedUser(undefined);
            return success;
          }
          return false;
        } catch (error) {
          console.log(error);
          if (error instanceof HttpError) {
            // TODO: create a nice message depending on error.status
          }
          notify({ type: 'error', message: error + '' });
          return false;
        }
      } else {
        notify({ type: 'warning', message: 'You must be logged to log out!' });
        return false;
      }
    },
    // the method code will change if setAuthenticatedUser changes which happens only at init
    [setAuthenticatedUser, notify, authenticatedUser, setAppContext],
  );

  // returns the AuthContext new value each time the authenticatedUser changes
  return {
    authenticatedUser,
    login: _login,
    logout: _logout,
  };
};

/**
 * AuthProvider a react context component which will propagate the auth context value to all its children
 */
export const AuthProvider: FC<PropsWithChildren> = ({ children }) => {
  const auth = _useAuth();
  return <authContext.Provider value={auth}>{children}</authContext.Provider>;
};
