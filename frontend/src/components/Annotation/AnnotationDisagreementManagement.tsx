import { FC, useEffect, useState } from 'react';
import Select from 'react-select';

import { useReconciliate, useTableDisagreement } from '../../core/api';
import { useAppContext } from '../../core/useAppContext';

/*
 * Manage disagreement in annotations
 */

interface AnnotationDisagreementManagementProps {
  projectSlug: string;
  dataset: string;
}

export const AnnotationDisagreementManagement: FC<AnnotationDisagreementManagementProps> = ({
  projectSlug,
  dataset,
}) => {
  const {
    appContext: { currentScheme, currentProject: project },
  } = useAppContext();

  // type of scheme from context
  const kindScheme = currentScheme
    ? project?.schemes?.available?.[currentScheme]?.kind
    : 'multiclass';

  // available labels from context
  const availableLabels = currentScheme ? project?.schemes?.available?.[currentScheme]?.labels : [];

  // get disagreement table
  const { tableDisagreement, users, reFetchTable } = useTableDisagreement(
    projectSlug,
    currentScheme,
    dataset,
  );

  useEffect(() => {
    reFetchTable();
  }, [reFetchTable, dataset]);

  const { postReconciliate } = useReconciliate(projectSlug, currentScheme || null, dataset);

  // state elements to validate
  const [changes, setChanges] = useState<{ [key: string]: string }>({});

  // function to validate changes
  const validateChanges = () => {
    Object.entries(changes).map(([id, label]) => {
      postReconciliate(id, label, users || []);
      setChanges({});
    });
    reFetchTable();
  };

  return (
    <>
      <div className="explanations">
        Disagreements between users on annotations. Abitrate for the correct label.
      </div>
      <div>{users?.length} user(s) involved in annotation</div>
      <div>
        <b>{tableDisagreement?.length} disagreements</b>
      </div>
      {Object.entries(changes).length > 0 && (
        <button className="btn btn-warning my-3" onClick={validateChanges}>
          Validate changes
        </button>
      )}

      {tableDisagreement?.map((element, index) => (
        <div className="alert alert-info" role="alert" key={index}>
          <details>
            <summary>
              <span className="badge">
                {element.id as string} - {element.current_label as string}
              </span>
            </summary>
            <span>{element.text as string}</span>
          </details>

          {element.annotations && (
            <div className="horizontal wrap">
              {Object.entries(element.annotations).map(([key, value], _) => (
                <div key={key}>
                  <span className="badge info">
                    {key}
                    <span className="badge hotkey">{value}</span>
                  </span>
                </div>
              ))}

              {kindScheme === 'multiclass' && (
                <select
                  style={{ flex: '1 0 200px' }}
                  onChange={(event) =>
                    setChanges({ ...changes, [element.id as string]: event.target.value })
                  }
                >
                  <option>Arbitation</option>
                  {(availableLabels || []).map((e) => (
                    <option key={e}>{e}</option>
                  ))}
                </select>
              )}
              {kindScheme === 'multilabel' && (
                <Select
                  isMulti
                  options={(availableLabels || []).map((e) => ({ value: e, label: e }))}
                  onChange={(e) => {
                    setChanges({
                      ...changes,
                      [element.id as string]: e.map((e) => e.value).join('|'),
                    });
                  }}
                />
              )}
            </div>
          )}
        </div>
      ))}
    </>
  );
};
