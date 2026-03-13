import { FC } from 'react';
import { FaGithub } from 'react-icons/fa';
import { IoMdLogOut } from 'react-icons/io';
import { Link, useNavigate } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import logo from '../assets/at.png';
import { LoginForm } from '../components/forms/LoginForm';
import Notifications from '../components/layout/Notifications';
import { useGetActiveUsers, useGetServer } from '../core/api';
import { useAuth } from '../core/useAuth';
import { LoginParams } from '../types';

export const HomePage: FC = () => {
  const { authenticatedUser, logout } = useAuth();
  const { n_users } = useGetActiveUsers();

  // add a development mode switch
  // const {
  //   appContext: { developmentMode },
  //   setAppContext,
  // } = useAppContext();

  // function to change the status of the interface
  // const actionDevelopmentMode = () => {
  //   setAppContext((prev) => ({ ...prev, developmentMode: !prev.developmentMode }));
  // };

  // possibility to log directly from the URL
  const navigate = useNavigate();
  const params = new URLSearchParams(window.location.search);
  const { login } = useAuth();
  const { version, messages } = useGetServer(null);
  if (params.get('username') && params.get('password')) {
    login({
      username: params.get('username'),
      password: params.get('password'),
    } as LoginParams).then(() => {
      navigate('/projects');
    });
  }

  return (
    <>
      <main className="container-fluid">
        <div className="row">
          <center>
            <div className="alert alert-warning mt-3">
              ⚠️ This interface is experimental. Please save your data regularly.{' '}
              <a
                href="https://github.com/activetigger/activetigger/issues"
                target="_blank"
                rel="noopener noreferrer"
              >
                If you encounter a problem, reach out or open an issue.
              </a>
            </div>
          </center>
          <div className="col-0 col-lg-3" />
          <div className="col-12 col-lg-6">
            <center>
              <div className="text-center">
                <h1
                  className="mb-1 fs-2 activetigger"
                  style={{
                    color: '#ff9a3c',
                  }}
                >
                  Active Tigger
                </h1>
                <h3 className="m-0 fs-5 text-muted fw-normal">Explore & Annotate Text</h3>
                <img
                  src={logo}
                  alt="ActiveTigger"
                  className="me-2"
                  style={{ width: '200px', height: '200px' }}
                />
              </div>
              {n_users == null && (
                <div className="alert alert-danger mt-3">
                  ⚠️ Can't reach the API (either temporary lag or server down). You can reach out on
                  the discord for more information.
                </div>
              )}

              {!authenticatedUser && n_users != null && <LoginForm />}

              {authenticatedUser && n_users != null && (
                <div>
                  <div className="text-center">
                    <div>
                      Welcome{' '}
                      <span className="fw-bold">
                        {authenticatedUser.username}
                        <IoMdLogOut
                          title="Logout"
                          onClick={async () => {
                            const success = await logout();
                            if (success) navigate('/');
                          }}
                          className="logout mx-2"
                          style={{ cursor: 'pointer' }}
                        />
                        <Tooltip anchorSelect=".logout" place="top">
                          Log out
                        </Tooltip>
                      </span>
                    </div>
                    <div className="justify-content-center">
                      <Link
                        to="/projects"
                        className="btn btn-lg text-white fw-bold shadow-sm px-4 py-2"
                        style={{
                          background: 'linear-gradient(90deg, #ff9a3c, #ff6f3c, #ffb347)',
                          border: 'none',
                          borderRadius: '2rem',
                          transition: 'transform 0.2s ease, box-shadow 0.2s ease',
                        }}
                        onMouseOver={(e) => {
                          e.currentTarget.style.transform = 'scale(1.05)';
                          e.currentTarget.style.boxShadow = '0 8px 16px rgba(0,0,0,0.2)';
                        }}
                        onMouseOut={(e) => {
                          e.currentTarget.style.transform = 'scale(1)';
                          e.currentTarget.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
                        }}
                      >
                        🐯 Go to your projects
                      </Link>
                    </div>
                    <div style={{ maxWidth: '600px', margin: '1rem auto' }}>
                      {(messages || []).map((msg) => (
                        <div
                          key={msg.id}
                          style={{
                            fontSize: '0.9rem',
                            color: '#333',
                          }}
                        >
                          <div style={{ color: '#777' }}>
                            {msg.content} •{' '}
                            <span style={{ fontSize: '0.5rem' }}>
                              {new Date(msg.time || Date.now()).toLocaleString()}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </center>
            <div style={{ height: '50px' }}></div>
          </div>
        </div>
        <footer className="footer mt-auto py-1 bg-primary text-white fixed-bottom">
          <div className="container text-center">
            <div
              className="ml-2 d-flex justify-content-center align-items-center"
              style={{ fontSize: '0.8rem' }}
            >
              {n_users} active users • API {version} • Client {__BUILD_DATE__}
              <a
                href="https://www.css.cnrs.fr/active-tigger/"
                target="_blank"
                rel="noopener noreferrer"
                style={{ all: 'unset', cursor: 'pointer' }}
              >
                {'\u00A0•\u00A0'}
                CREST / CSS @ IPP © 2025
              </a>
              <a
                href="https://github.com/activetigger/activetigger"
                target="_blank"
                rel="noopener noreferrer"
                className="d-flex align-items-center"
                style={{ all: 'unset', cursor: 'pointer' }}
              >
                <FaGithub className="mx-2" />
              </a>
            </div>
          </div>
        </footer>
      </main>
      <Notifications />
    </>
  );
};
