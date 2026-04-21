import { FC } from 'react';

import { useStatistics } from '../core/api';

interface TableProps {
  dataDict: Record<string, number>;
}

export const DataTable: FC<TableProps> = ({ dataDict }) => {
  if (!dataDict) {
    return null;
  }

  return (
    <div>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <tbody>
          {Object.entries(dataDict).map(([key, value]) => (
            <tr key={key}>
              <td>{key}</td>
              <td>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

interface StatisticsProps {
  projectSlug: string;
  scheme: string;
}

/**
 * Component to display statistics
 */
// NOTE: Axel: Not used
export const ProjectStatistics: FC<StatisticsProps> = ({ projectSlug, scheme }) => {
  // get the statistics of the project from the API
  const { statistics } = useStatistics(projectSlug, scheme);

  // graph for the distribution
  //const dataDict = statistics ? statistics['train_annotated_distribution'] : {};

  return (
    <div className="container-fluid">
      <div className="row">
        <div className="col-md-12">
          {statistics && (
            <div>
              <table className="table-statistics">
                <tbody>
                  <tr className="table-delimiter">
                    <td>Trainset</td>
                    <td></td>
                  </tr>
                  <tr>
                    <td>Users involved</td>
                    <td>
                      <ul>
                        {statistics['users'].map((e) => (
                          <li key={e}>{e}</li>
                        ))}
                      </ul>
                    </td>
                  </tr>
                  <tr>
                    <td>Total</td>
                    <td>{statistics['train_set_n']}</td>
                  </tr>
                  <tr>
                    <td>Annotated</td>
                    <td>{statistics['train_annotated_n']}</td>
                  </tr>

                  <tr>
                    <td>Distribution</td>
                    <td>
                      <DataTable
                        dataDict={
                          statistics['train_annotated_distribution'] as Record<string, number>
                        }
                      />
                    </td>
                  </tr>
                  <tr className="table-delimiter">
                    <td>Testset</td>
                    <td></td>
                  </tr>
                  <tr>
                    <td>Total</td>
                    <td>{statistics['test_set_n']}</td>
                  </tr>
                  <tr>
                    <td>Annotated</td>
                    <td>{statistics['test_annotated_n']}</td>
                  </tr>
                  <tr>
                    <td>Distribution</td>
                    <td>
                      {' '}
                      {statistics['test_annotated_distribution'] && (
                        <DataTable
                          dataDict={
                            statistics['test_annotated_distribution'] as Record<string, number>
                          }
                        />
                      )}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
