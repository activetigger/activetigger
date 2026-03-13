import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { MdOutlineDeleteOutline } from 'react-icons/md';

import { Modal } from 'react-bootstrap';
import { FaPlusCircle } from 'react-icons/fa';
import Select from 'react-select';
import { PageLayout } from '../components/layout/PageLayout';
import {
  useAddUserAuthProject,
  useCreateUser,
  useDeleteUser,
  useDeleteUserAuthProject,
  useUserProjects,
  useUsers,
  useUsersAuth,
} from '../core/api';
import { useNotifications } from '../core/notifications';
import { useAuth } from '../core/useAuth';

interface newUser {
  username: string;
  password: string;
  status: string;
  mail: string;
}

export const UsersPage: FC = () => {
  const { projects } = useUserProjects();
  const [currentProjectSlug, setCurrentProjectSlug] = useState<string | null>(null);
  const { notify } = useNotifications();

  const { authenticatedUser } = useAuth();

  const [currentUser, setCurrentUser] = useState<string | null>(null);
  const [currentAuth, setCurrentAuth] = useState<string>('manager');

  // display boxes
  const [showCreateUser, setShowCreateUser] = useState<boolean>(false);

  const { authUsers, reFetchUsersAuth } = useUsersAuth(currentProjectSlug);
  const { users, reFetchUsers } = useUsers();

  const { deleteUser } = useDeleteUser(reFetchUsers);
  const { createUser } = useCreateUser(reFetchUsers);

  const { deleteUserAuth } = useDeleteUserAuthProject(currentProjectSlug, reFetchUsersAuth);
  const { addUserAuth } = useAddUserAuthProject(currentProjectSlug, reFetchUsersAuth);

  const { handleSubmit, register, reset } = useForm<newUser>();
  const onSubmit: SubmitHandler<newUser> = async (data) => {
    await createUser(data.username, data.password, data.status, data.mail);
    reset();
    setShowCreateUser(false);
  };

  const userOptions = users
    ? Object.keys(users).map((userKey) => ({
        value: userKey,
        label: userKey,
      }))
    : [];

  const projectOptions = (projects || []).map((project) => ({
    value: project.parameters.project_slug,
    label: project.parameters.project_slug,
  }));

  const accessToList = authenticatedUser?.username === 'root';

  return (
    <PageLayout currentPage="users">
      <div className="container">
        <div className="row">
          <div className="col-0 col-sm-1 col-md-2" />
          <div className="col-12 col-sm-10 col-md-8">
            <div className="explanations">Manage users and rights</div>

            <span className="explanations">User</span>
            <div className="d-flex align-items-center">
              {accessToList ? (
                <>
                  <Select
                    id="select-user"
                    className="flex-grow-1"
                    options={userOptions}
                    onChange={(selectedOption) => {
                      setCurrentUser(selectedOption ? selectedOption.value : null);
                    }}
                    isClearable
                    placeholder="Select a user"
                  />
                  <button
                    className="btn btn p-0 m-2"
                    onClick={() => {
                      setShowCreateUser(!showCreateUser);
                      reFetchUsers();
                    }}
                  >
                    <FaPlusCircle size={25} />
                  </button>
                  <button
                    className="btn btn p-0"
                    onClick={() => {
                      deleteUser(currentUser);
                      reFetchUsers();
                    }}
                  >
                    <MdOutlineDeleteOutline size={30} />
                  </button>
                </>
              ) : (
                <input
                  className="form-control"
                  type="text"
                  onChange={(c) => {
                    setCurrentUser(c.target.value);
                  }}
                  placeholder="Write a user handle"
                />
              )}
            </div>
            <Modal
              show={showCreateUser}
              onHide={() => setShowCreateUser(false)}
              size="xl"
              id="users-modal"
            >
              <Modal.Header closeButton>
                <Modal.Title>Create a new user</Modal.Title>
              </Modal.Header>
              <Modal.Body>
                <form onSubmit={handleSubmit(onSubmit)}>
                  <input
                    className="form-control me-2 mt-2"
                    type="text"
                    {...register('username')}
                    placeholder="New user name"
                  />
                  <input
                    className="form-control me-2 mt-2"
                    type="text"
                    {...register('password')}
                    placeholder="Password"
                  />
                  <input
                    className="form-control me-2 mt-2"
                    type="email"
                    {...register('mail')}
                    placeholder="Mail"
                  />
                  <select {...register('status')} className="me-2 mt-2">
                    <option>manager</option>
                    <option>annotator</option>
                  </select>
                  <button className="btn btn-primary me-2 mt-2">Add user</button>
                </form>
              </Modal.Body>
            </Modal>
            <div className="mt-3">
              <span className="explanations">Project</span>
              <Select
                id="select-project"
                options={projectOptions}
                onChange={(selectedOption) => {
                  setCurrentProjectSlug(selectedOption ? selectedOption.value : null);
                }}
                isClearable
                placeholder="Select a project"
              />
              {/* <input
                  className="form-control"
                  type="text"
                  onChange={(c) => {
                    setCurrentProjectSlug(c.target.value);
                  }}
                  placeholder="Write a project handle"
                /> */}
              <div>
                {authUsers ? (
                  <table className="table-auth">
                    <tbody>
                      <tr>
                        <th>User</th>
                        <th>Role</th>
                        <th>Delete</th>
                      </tr>
                      {Object.entries(authUsers).map(([user, auth]) => (
                        <tr key={user}>
                          <td>{user}</td>
                          <td>{auth}</td>
                          <td>
                            <button
                              className="btn btn p-0"
                              onClick={() => {
                                deleteUserAuth(user);
                              }}
                            >
                              <MdOutlineDeleteOutline />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <span></span>
                )}
              </div>
            </div>

            <div className="alert alert-light d-flex justify-content-between align-items-center mt-3 gap-2">
              <span>
                Add user <b>{currentUser}</b> to project as:
              </span>
              <div className="position-relative flex-grow-1">
                <select
                  id="select-auth"
                  className="form-select"
                  onChange={(e) => {
                    setCurrentAuth(e.target.value);
                  }}
                  defaultValue={'manager'}
                >
                  <option key={'manager'}>manager</option>
                  <option key={'contributor'}>contributor</option>
                  <option key={'annotator'}>annotator</option>
                </select>
              </div>
              <button
                onClick={() => {
                  if (currentUser && currentAuth) {
                    addUserAuth(currentUser, currentAuth);
                  } else
                    notify({
                      type: 'error',
                      message: 'Please select a user, a project and a role.',
                    });
                  reFetchUsers();
                }}
                className="btn"
              >
                <FaPlusCircle size={25} />
              </button>
            </div>
          </div>
          <div className="col-0 col-sm-1 col-md-2" />
        </div>
      </div>
    </PageLayout>
  );
};

// list of users associated + possibility to destroy them
// button to add user > menu with name/password/status
// refetch la page
