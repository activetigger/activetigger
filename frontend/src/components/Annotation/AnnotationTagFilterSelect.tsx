import { FC, useCallback, useMemo } from 'react';
import Select, { MultiValueGenericProps, OptionProps, components } from 'react-select';

import { omit, sortBy, uniq } from 'lodash';
import { useAppContext } from '../../core/useAppContext';
import { truncateInMiddle } from '../../core/utils';
import { SelectionConfig } from '../../types';
import { AnnotationIcon, NoAnnotationIcon, UserIcon } from '../Icons';

export interface TagFilterOption {
  sample: string;
  user?: string | undefined;
  label?: string | undefined;
  value: string;
}

interface AnnotationTagFilterSelectProps {
  availableLabels: string[];
}

function filterOptionValue(option: Omit<TagFilterOption, 'value'>) {
  const value = [option.sample, option.label || '', option.user || ''].join('|');
  return value;
}

/**
 * React select customizations
 **/

const OptionIcon: FC<{ option: TagFilterOption }> = ({ option }) => (
  <>
    {option.sample === 'untagged' && <NoAnnotationIcon className="me-1" />}
    {option.sample === 'tagged' && option.user === undefined && <AnnotationIcon className="me-1" />}
    {option.sample === 'tagged' && option.user !== undefined && <UserIcon className="me-1" />}
  </>
);

const MultiValueLabel = ({ children, ...props }: MultiValueGenericProps<TagFilterOption>) => {
  return (
    <components.MultiValueLabel {...props}>
      <OptionIcon option={props.data} />
      {children}
    </components.MultiValueLabel>
  );
};
const Option = ({ children, ...props }: OptionProps<TagFilterOption>) => {
  return (
    <components.Option {...props}>
      <OptionIcon option={props.data} />
      {children}
    </components.Option>
  );
};

/**
 * Main component
 */
export const AnnotationTagFilterSelect: FC<AnnotationTagFilterSelectProps> = ({
  availableLabels,
}) => {
  // context data
  const {
    appContext: { selectionConfig, currentProject: project },
    setAppContext,
  } = useAppContext();

  // OPTIONS
  const tagFilterOptions: TagFilterOption[] = useMemo(() => {
    const samples =
      // all = no option selected so we remove it from options
      project?.next.sample
        .filter((s) => s !== 'all')
        .map((s) => ({ sample: s, label: undefined, user: undefined })) || [];
    const labels = availableLabels.map((l) => ({
      sample: 'tagged',
      label: l,
      user: undefined,
    }));
    const users =
      project?.users?.users.map((u) => ({
        sample: 'tagged',
        label: undefined,
        user: u,
      })) || [];
    return [...samples, ...labels, ...users].map((o) => ({ ...o, value: filterOptionValue(o) }));
  }, [project?.next.sample, availableLabels, project?.users?.users]);
  // handle option coherence by disabling options depending on current selection Config
  const isTagFilterOptionDisabled = useCallback(
    (option: TagFilterOption) => {
      return selectionConfig.sample !== 'all' && selectionConfig.sample !== option.sample;
    },
    [selectionConfig.sample],
  );

  const tagFilterSelectValue = useMemo(() => {
    const selectedValues =
      selectionConfig.labels?.length || selectionConfig.users?.length
        ? [
            ...(selectionConfig.labels?.map((l) =>
              filterOptionValue({
                sample: selectionConfig.sample,
                label: l,
                user: undefined,
              }),
            ) || []),
            ...(selectionConfig.users?.map((u) =>
              filterOptionValue({
                sample: selectionConfig.sample,
                label: undefined,
                user: u,
              }),
            ) || []),
          ]
        : [
            filterOptionValue({
              sample: selectionConfig.sample,
              label: undefined,
              user: undefined,
            }),
          ];
    const values = tagFilterOptions.filter((o) => selectedValues.includes(o.value));
    return values;
  }, [tagFilterOptions, selectionConfig.sample, selectionConfig.labels, selectionConfig.users]);

  return (
    <Select
      className="react-select"
      isMulti
      isClearable
      options={tagFilterOptions}
      isOptionDisabled={isTagFilterOptionDisabled}
      value={tagFilterSelectValue}
      getOptionLabel={(o) =>
        truncateInMiddle(
          o.sample === 'tagged' && o.label !== undefined
            ? o.label
            : o.sample === 'tagged' && o.user !== undefined
              ? `by ${o.user}`
              : o.sample || '',
          20,
        )
      }
      components={{
        MultiValueLabel,
        Option,
      }}
      onChange={(options) => {
        if (options !== null && options.length > 0) {
          // we transform list of options into one selectionConfig object
          const compactOptions: Pick<SelectionConfig, 'sample' | 'labels' | 'users'> = {
            sample: options[0].sample,
            labels:
              sortBy(
                uniq(options.map((o) => o.label).filter((l): l is string => l !== undefined)),
              ) || undefined,
            users:
              sortBy(
                uniq(options.map((o) => o.user).filter((u): u is string => u !== undefined)),
              ) || undefined,
          };
          setAppContext((prev) => {
            return {
              ...prev,
              selectionConfig: {
                ...omit(prev.selectionConfig, ['sample', 'labels', 'users']),
                ...compactOptions,
              },
            };
          });
        }
        // if not option selected, selection is all
        else
          setAppContext((prev) => ({
            ...prev,
            selectionConfig: {
              ...omit(prev.selectionConfig, ['sample', 'labels', 'users']),
              sample: 'all',
            },
          }));
      }}
    />
  );
};
