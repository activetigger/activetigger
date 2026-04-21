import { FC, useEffect, useMemo, useState } from 'react';
import Select from 'react-select';

import { useGetCompareSchemes } from '../core/api';
import { useAppContext } from '../core/useAppContext';
import { ModelParametersTab } from './ModelParametersTab';

/*
 * Manage disagreement in annotations
 */

interface AnnotationDisagreementManagementProps {
  projectSlug: string;
  dataset: string;
}

export const SchemesComparisonManagement: FC<AnnotationDisagreementManagementProps> = ({
  projectSlug,
  dataset,
}) => {
  const {
    appContext: { currentScheme, currentProject: project },
  } = useAppContext();

  const [schemeA, setSchemeA] = useState<string | null>(currentScheme ? currentScheme : null);
  const [schemeB, setSchemeB] = useState<string | null>(null);
  const { compare, reFetchCompare } = useGetCompareSchemes(
    projectSlug,
    schemeA || '',
    schemeB || '',
    dataset,
  );

  const availableSchemes = useMemo(() => {
    return project ? Object.keys(project.schemes.available) : [];
  }, [project]);

  useEffect(() => {
    reFetchCompare();
  }, [reFetchCompare, dataset]);

  return (
    <>
      <div className="explanations">Compare the annotations between 2 schemes</div>
      <div className="horizontal wrap">
        <div style={{ flex: '1 1 auto', minWidth: '100px', marginRight: '15px' }}>
          <label>Scheme A</label>
          <Select
            id="select-scheme-a"
            className="flex-grow-1"
            options={availableSchemes.map((e) => ({ value: e, label: e }))}
            value={{ value: schemeA, label: schemeA }}
            onChange={(selectedOption) => {
              setSchemeA(selectedOption ? selectedOption.value : null);
            }}
            isClearable
            placeholder="Select scheme A"
          />
        </div>
        <div style={{ flex: '1 1 auto', minWidth: '100px' }}>
          <label>Scheme B</label>
          <Select
            id="select-scheme-b"
            className="flex-grow-1"
            options={availableSchemes.map((e) => ({ value: e, label: e }))}
            value={{ value: schemeB, label: schemeB }}
            onChange={(selectedOption) => {
              setSchemeB(selectedOption ? selectedOption.value : null);
            }}
            isClearable
            placeholder="Select scheme B"
          />
        </div>
      </div>
      {/* TODO: Axel: Refactor */}
      {compare ? (
        <div className="horizontal center">
          <ModelParametersTab
            params={
              {
                'Overlapping labels (%)': compare.labels_overlapping.toFixed(2),
                'Annotated elements in common': compare.n_annotated,
                'Agreement Cohen-Kappa': compare.cohen_kappa && compare.cohen_kappa.toFixed(2),
                'Agreement Percentage': compare.percentage && compare.percentage.toFixed(2),
              } as unknown as Record<string, unknown>
            }
          />
        </div>
      ) : (
        <span>No data</span>
      )}
    </>
  );
};
