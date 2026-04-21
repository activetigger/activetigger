import { FC } from 'react';

import { IoWarning } from 'react-icons/io5';
import { PageLayout } from '../components/layout/PageLayout';

export const DocPage: FC = () => {
  return (
    <PageLayout currentPage="help">
      <div className="container-fluid">
        <div className="row">
          <div className="col-1"></div>
          <div className="col-8">
            <h2 className="subsection">Documentation</h2>
            <div className="alert alert-warning" role="alert">
              <IoWarning /> Documentation to write
            </div>
            <h4 className="subsection">Create a project</h4>
            To write
            <h4 className="subsection">Prepare labels and features</h4>
            To write
            <h4 className="subsection">Explore the data</h4>
            To write
            <li>
              <ul>Search field regex are case sensitive</ul>
            </li>
            <h4 className="subsection">Annotation phase</h4>
            To write
            <li>
              <ul>Explain the X/Y/Z count</ul>
            </li>
            <h4 className="subsection">Fine-tune a BERT model</h4>
            To write
            <h4 className="subsection">Test model</h4>
            The test set:
            <ul>
              <li>Created on the beginning of the project</li>
              <li>Uploaded latter</li>
            </ul>
            Once activated, the test mode :
            <ul>
              <li>Deactivate for the user the choice of scheme, label management</li>
              <li>Allow only annotation for the test set</li>
              <li>Allow to explore the test set</li>
            </ul>
            <h4 className="subsection">Export data / models</h4>
            To write
            <h4 className="subsection">General comments</h4>
            <ul>
              <li>Only one process allowed in the same time by user</li>
            </ul>
          </div>
        </div>
      </div>{' '}
    </PageLayout>
  );
};
