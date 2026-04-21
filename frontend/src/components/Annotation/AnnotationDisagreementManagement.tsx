import { FC, useEffect, useMemo, useState } from 'react';
import Select from 'react-select';

import { useReconciliate, useTableDisagreement } from '../../core/api';
import { useAppContext } from '../../core/useAppContext';
import { ModelParametersTab } from '../ModelParametersTab';

/*
 * Manage disagreement in annotations
 */

interface AnnotationDisagreementManagementProps {
  projectSlug: string;
  dataset: string;
  onDirtyChange?: (dirty: boolean) => void;
}

export const AnnotationDisagreementManagement: FC<AnnotationDisagreementManagementProps> = ({
  projectSlug,
  dataset,
  onDirtyChange,
}) => {
  const {
    appContext: { currentScheme, currentProject: project },
  } = useAppContext();

  // type of scheme from context
  const kindScheme = currentScheme
    ? project?.schemes?.available?.[currentScheme]?.kind
    : 'multiclass';

  // available labels from context
  const availableLabels = useMemo(
    () => (currentScheme ? project?.schemes?.available?.[currentScheme]?.labels : []),
    [currentScheme, project],
  );

  // get disagreement table
  const { tableDisagreement, users, agreementStats, reFetchTable } = useTableDisagreement(
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

  // selected user for filtering
  const [selectedUser, setSelectedUser] = useState<string | null>(null);

  // notify parent when dirty state changes
  useEffect(() => {
    onDirtyChange?.(Object.keys(changes).length > 0);
  }, [changes, onDirtyChange]);

  // function to validate changes
  const validateChanges = () => {
    Object.entries(changes).map(([id, label]) => {
      postReconciliate(id, label, users || []);
      setChanges({});
    });
    reFetchTable();
  };

  // filter and group disagreements by selected user's label
  const groupedDisagreements = useMemo(() => {
    if (!tableDisagreement) return null;
    if (!selectedUser) return { '': tableDisagreement };

    const NOT_TAGGED = 'Not tagged by this user';

    // initialize groups in scheme label order for consistent sorting across users
    const groups: Record<string, typeof tableDisagreement> = {};
    for (const label of availableLabels || []) {
      groups[label] = [];
    }
    groups[NOT_TAGGED] = [];

    for (const element of tableDisagreement) {
      const annotations = element.annotations as Record<string, string | null> | undefined;
      if (!annotations) continue;
      const userLabel = annotations[selectedUser];
      if (userLabel && userLabel !== '-----') {
        if (!groups[userLabel]) groups[userLabel] = [];
        groups[userLabel].push(element);
      } else {
        // keep elements untagged by selected user but with disagreement between others
        groups[NOT_TAGGED].push(element);
      }
    }

    // drop empty groups
    return Object.fromEntries(Object.entries(groups).filter(([, els]) => els.length > 0));
  }, [tableDisagreement, selectedUser, availableLabels]);

  // render a single disagreement element
  const renderElement = (element: NonNullable<typeof tableDisagreement>[0], index: number) => (
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
          {/* show selected user's annotation first */}
          {selectedUser && (element.annotations as Record<string, string>)[selectedUser] && (
            <div>
              <span className="badge info">
                {selectedUser}
                <span className="badge hotkey">
                  {(element.annotations as Record<string, string>)[selectedUser]}
                </span>
              </span>
            </div>
          )}
          {Object.entries(element.annotations as Record<string, string>)
            .filter(([key]) => key !== selectedUser)
            .map(([key, value]) => (
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
              <option>Resolve</option>
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
  );

  return (
    <>
      {agreementStats && agreementStats.n_total > 0 && (
        <div className="horizontal center">
          <ModelParametersTab
            sortKeys={false}
            params={
              {
                'Annotated by 2+ users': agreementStats.n_total,
                Agreements: agreementStats.n_agreements,
                Disagreements: agreementStats.n_disagreements,
                'Agreement (%)': agreementStats.agreement_percentage
                  ? (agreementStats.agreement_percentage * 100).toFixed(2)
                  : null,
                "Cohen's Kappa": agreementStats.cohen_kappa
                  ? agreementStats.cohen_kappa.toFixed(2)
                  : null,
              } as unknown as Record<string, unknown>
            }
          />
        </div>
      )}
      <div className="explanations">
        Disagreements between users on annotations. Abitrate for the correct label.
      </div>
      <div>{users?.length} user(s) involved in annotation</div>
      <div>
        <b>{tableDisagreement?.length} disagreements</b>
      </div>

      <div style={{ maxWidth: '300px', marginTop: '10px', marginBottom: '10px' }}>
        <label>Filter by user (ordered by label)</label>
        <Select
          options={(users || []).map((u) => ({ value: u, label: u }))}
          value={selectedUser ? { value: selectedUser, label: selectedUser } : null}
          onChange={(option) => setSelectedUser(option ? option.value : null)}
          isClearable
          placeholder="All users"
        />
      </div>

      {Object.entries(changes).length > 0 && (
        <button className="btn btn-warning my-3" onClick={validateChanges}>
          Validate changes
        </button>
      )}

      {groupedDisagreements &&
        Object.entries(groupedDisagreements).map(([label, elements]) => (
          <div key={label}>
            {label && (
              <h6 style={{ marginTop: '15px' }}>
                <span className="badge">{label}</span> ({elements.length})
              </h6>
            )}
            {elements.map((element, index) => renderElement(element, index))}
          </div>
        ))}
    </>
  );
};
