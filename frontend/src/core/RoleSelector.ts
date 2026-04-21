import { FC } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useAuth } from './useAuth';

export interface RoleSelectorProps {
  allowedRoles: string[];
}

export const RoleSelector: FC<RoleSelectorProps> = ({ allowedRoles }) => {
  const { authenticatedUser } = useAuth(); // Assuming `user` has a `role` property
  const { projectName } = useParams();
  const navigate = useNavigate();

  if (!authenticatedUser || !allowedRoles.includes(authenticatedUser.status as string)) {
    navigate(`/projects/${projectName}`, { replace: true });
  }

  return null;
};
