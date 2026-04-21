import { FC, useCallback, useMemo } from 'react';
import { CiClock1 } from 'react-icons/ci';
import Select, { MultiValueGenericProps, OptionProps, components } from 'react-select';

import { sortBy, uniq } from 'lodash';
import { truncateInMiddle } from '../core/utils';
import { AnnotationIcon, UserIcon } from './Icons';

export interface TableFilterState {
  recent: boolean;
  labels: string[];
  users: string[];
}

interface TableFilterOption {
  sample: 'recent' | 'label' | 'user';
  label?: string;
  user?: string;
  value: string;
}

interface TableTagFilterSelectProps {
  availableLabels: string[];
  availableUsers: string[];
  onChange: (filter: TableFilterState) => void;
}

function optionValue(option: Omit<TableFilterOption, 'value'>) {
  return [option.sample, option.label || '', option.user || ''].join('|');
}

/**
 * Custom react-select rendering
 */
const OptionIcon: FC<{ option: TableFilterOption }> = ({ option }) => (
  <>
    {option.sample === 'recent' && <CiClock1 className="me-1" />}
    {option.sample === 'label' && <AnnotationIcon className="me-1" />}
    {option.sample === 'user' && <UserIcon className="me-1" />}
  </>
);

const MultiValueLabel = ({ children, ...props }: MultiValueGenericProps<TableFilterOption>) => {
  return (
    <components.MultiValueLabel {...props}>
      <OptionIcon option={props.data} />
      {children}
    </components.MultiValueLabel>
  );
};

const Option = ({ children, ...props }: OptionProps<TableFilterOption>) => {
  return (
    <components.Option {...props}>
      <OptionIcon option={props.data} />
      {children}
    </components.Option>
  );
};

/**
 * Multi-select filter for the tabular data view.
 * Allows filtering by labels, users, or showing recent annotations.
 */
export const TableTagFilterSelect: FC<TableTagFilterSelectProps> = ({
  availableLabels,
  availableUsers,
  onChange,
}) => {
  const options: TableFilterOption[] = useMemo(() => {
    const recentOption: TableFilterOption = {
      sample: 'recent',
      value: optionValue({ sample: 'recent' }),
    };
    const labelOptions: TableFilterOption[] = availableLabels.map((l) => ({
      sample: 'label' as const,
      label: l,
      value: optionValue({ sample: 'label', label: l }),
    }));
    const userOptions: TableFilterOption[] = availableUsers.map((u) => ({
      sample: 'user' as const,
      user: u,
      value: optionValue({ sample: 'user', user: u }),
    }));
    return [recentOption, ...labelOptions, ...userOptions];
  }, [availableLabels, availableUsers]);

  // recent is incompatible with label/user filters
  const isOptionDisabled = useCallback(
    (option: TableFilterOption, selectValue: readonly TableFilterOption[]) => {
      if (selectValue.length === 0) return false;
      const hasRecent = selectValue.some((o) => o.sample === 'recent');
      if (hasRecent) return option.sample !== 'recent';
      return option.sample === 'recent';
    },
    [],
  );

  return (
    <Select<TableFilterOption, true>
      className="react-select"
      isMulti
      isClearable
      placeholder="All elements"
      options={options}
      isOptionDisabled={(option, selectValue) => isOptionDisabled(option, selectValue)}
      getOptionLabel={(o) =>
        truncateInMiddle(
          o.sample === 'label' && o.label !== undefined
            ? o.label
            : o.sample === 'user' && o.user !== undefined
              ? `by ${o.user}`
              : o.sample === 'recent'
                ? 'recent'
                : '',
          20,
        )
      }
      components={{ MultiValueLabel, Option }}
      onChange={(selected) => {
        if (!selected || selected.length === 0) {
          onChange({ recent: false, labels: [], users: [] });
          return;
        }
        const arr = [...selected];
        if (arr.some((o) => o.sample === 'recent')) {
          onChange({ recent: true, labels: [], users: [] });
        } else {
          onChange({
            recent: false,
            labels: sortBy(
              uniq(arr.filter((o) => o.label !== undefined).map((o) => o.label as string)),
            ),
            users: sortBy(
              uniq(arr.filter((o) => o.user !== undefined).map((o) => o.user as string)),
            ),
          });
        }
      }}
    />
  );
};
