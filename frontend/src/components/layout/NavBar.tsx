import cx from 'classnames';
import { FC, useState } from 'react';
import { HiInbox, HiOutlineInbox } from 'react-icons/hi';
import { IoMdLogIn, IoMdLogOut } from 'react-icons/io';
import { Link, useNavigate } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import logo from '../../assets/at.png';
import { useGetInbox } from '../../core/api';
import { useAppContext } from '../../core/useAppContext';
import { useAuth } from '../../core/useAuth';

const DOCUMENTATION_LINK = 'https://activetigger.github.io/documentation';

interface NavBarPropsType {
  currentPage?: string;
  projectName?: string | null;
}

const NavBar: FC<NavBarPropsType> = ({ currentPage }) => {
  const { authenticatedUser, logout } = useAuth();
  const navigate = useNavigate();
  const { inbox } = useGetInbox();
  const inboxCount = (inbox || []).length;
  const hasMessages = inboxCount > 0;

  const [expanded, setExpanded] = useState<boolean>(false);

  // function to clear history
  const {
    appContext: { displayConfig },
  } = useAppContext();

  const PAGES: { id: string; label: string; href: string }[] =
    displayConfig.interfaceType === 'default'
      ? [
          { id: 'projects', label: 'Projects', href: '/projects' },
          { id: 'account', label: 'Account', href: '/account' },
          { id: 'users', label: 'Users', href: '/users' },
        ]
      : [
          { id: 'projects', label: 'Projects', href: '/projects' },
          { id: 'account', label: 'Account', href: '/account' },
        ];

  return (
    // NOTE: Axel: I didn't do no css refactor here cause it's highly specific
    <nav className="navbar navbar-dark bg-primary navbar-expand-md" id="nav-bar-header">
      <div className="container-fluid">
        <div id="logo-container" className="navbar-brand">
          <Link className="navbar-brand" to="/">
            <img
              src={logo}
              alt="ActiveTigger"
              className="d-inline-bock me-2"
              style={{ width: '50px', height: '50px' }}
            />
            Active Tigger
          </Link>
        </div>
        <button
          className="navbar-toggler"
          type="button"
          aria-controls="navbarSupportedContent"
          aria-expanded={expanded}
          aria-label="Toggle navigation"
          onClick={() => setExpanded((e) => !e)}
        >
          <span className="navbar-toggler-icon"></span>
        </button>
        <div
          className={cx('navbar-collapse navbar navbar-dark', expanded ? 'd-flex' : 'd-none')}
          id="navbarSupportedContent"
        >
          <ul className="navbar-nav">
            {PAGES.map(({ id, label, href }) => (
              <li key={id} className="nav-item">
                <Link
                  className={cx('nav-link', currentPage === id && 'active')}
                  aria-current={currentPage === id ? 'page' : undefined}
                  to={href}
                >
                  {label}
                </Link>
              </li>
            ))}
            {authenticatedUser && authenticatedUser.username === 'root' && (
              <li key="monitor" className="nav-item">
                <Link
                  className={cx('nav-link', currentPage === 'monitor' && 'active')}
                  aria-current={'page'}
                  to="/monitor"
                >
                  Monitor
                </Link>
              </li>
            )}
            <li className="nav-item" key="docs">
              <a
                className={cx('nav-link', currentPage === 'docs' && 'active')}
                href={DOCUMENTATION_LINK}
                target="_blank"
                rel="noreferrer"
                aria-current={currentPage === 'docs' ? 'page' : undefined}
              >
                Documentation
              </a>
            </li>
          </ul>

          {authenticatedUser ? (
            <div className="d-flex navbar-nav  navbar-text navbar-text-margins align-items-center gap-2">
              <Link
                to="/messages"
                className="nav-messages text-white d-flex align-items-center position-relative"
                aria-label="Messages"
                style={{ textDecoration: 'none' }}
              >
                {hasMessages ? (
                  <HiInbox
                    size={28}
                    style={{
                      color: '#ff9a3c',
                      filter: 'drop-shadow(0 0 4px rgba(255, 154, 60, 0.6))',
                    }}
                  />
                ) : (
                  <HiOutlineInbox size={22} />
                )}
                {hasMessages && (
                  <span
                    className="position-absolute badge rounded-pill bg-danger"
                    style={{
                      top: '-4px',
                      right: '-8px',
                      fontSize: '0.65rem',
                      padding: '2px 5px',
                      lineHeight: 1,
                      border: '1px solid white',
                    }}
                  >
                    {inboxCount > 9 ? '9+' : inboxCount}
                  </span>
                )}
              </Link>
              <Tooltip anchorSelect=".nav-messages" place="bottom">
                {hasMessages
                  ? `You have ${inboxCount} message${inboxCount > 1 ? 's' : ''}`
                  : 'Messages'}
              </Tooltip>
              <button
                className="btn btn-primary logout text-white"
                onClick={async () => {
                  const success = await logout();
                  if (success) navigate('/');
                }}
              >
                Logged as {authenticatedUser.username} <IoMdLogOut title="Logout" />
              </button>
              <Tooltip anchorSelect=".logout" place="top">
                Log out
              </Tooltip>
            </div>
          ) : (
            <Link to="/login">
              <IoMdLogIn title="login" />
            </Link>
          )}
        </div>
      </div>
    </nav>
  );
};

export default NavBar;
