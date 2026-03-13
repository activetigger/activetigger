import { createContext, useContext } from 'react';
import { LoginParams } from '../types';
import { AuthContext, AuthenticatedUser } from './auth';

/**
 * useAuth
 * the main auth hook which provides the auth context current value
 * @returns
 */
export function useAuth() {
  // we could use useContext(authContext) in our components but we reright a more elegant useAuth for clarity
  return useContext(authContext);
}

/**
 * getAuthHeaders
 * utility functions which provides the HTTP headers to be included in API calls
 * we write this code here to centralize auth logics
 * @param user a Authenticated user
 * @returns Authorization and username HTTP headers to be included in API calls
 */

export function getAuthHeaders(user?: AuthenticatedUser) {
  return user
    ? {
        headers: {
          Authorization: `Bearer ${user.access_token}`,
          username: user.username,
        },
      }
    : undefined;
} // create a react context to centralize and share auth state and mechanism with the whole application

export const authContext = createContext<AuthContext>({
  authenticatedUser: undefined,
  login: async (_: LoginParams) => {},
  logout: async () => false,
});
