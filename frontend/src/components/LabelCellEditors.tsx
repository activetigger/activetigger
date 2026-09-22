import { FC } from 'react';
import Select from 'react-select';

interface LabelCellEditorProps {
  value: string;
  availableLabels: string[];
  // commit=true closes the grid editor
  onChange: (label: string, commit: boolean) => void;
  onClose?: () => void;
}

// select editor for multiclass schemes
export const LabelDropdownEditor: FC<LabelCellEditorProps> = ({
  value,
  availableLabels,
  onChange,
}) => (
  <select value={value} onChange={(event) => onChange(event.target.value, true)} autoFocus>
    <option></option>
    {availableLabels.map((l) => (
      <option key={l} value={l}>
        {l}
      </option>
    ))}
  </select>
);

// multi-select editor for multilabel schemes: chips with remove + dropdown to add
export const LabelMultiSelectEditor: FC<LabelCellEditorProps> = ({
  value,
  availableLabels,
  onChange,
  onClose,
}) => {
  const current = value ? value.split('|').filter(Boolean) : [];
  return (
    <Select
      isMulti
      autoFocus
      defaultMenuIsOpen
      closeMenuOnSelect={false}
      blurInputOnSelect={false}
      menuPortalTarget={document.body}
      menuPosition="fixed"
      options={availableLabels.map((l) => ({ value: l, label: l }))}
      value={current.map((l) => ({ value: l, label: l }))}
      onChange={(selected) => onChange((selected || []).map((o) => o.value).join('|'), false)}
      onBlur={onClose}
      styles={{
        container: (base) => ({ ...base, width: '100%' }),
        control: (base) => ({ ...base, minHeight: 30, fontSize: 12 }),
        menu: (base) => ({ ...base, fontSize: 12 }),
        menuPortal: (base) => ({ ...base, zIndex: 9999 }),
      }}
    />
  );
};
