import { FC, useCallback, useEffect, useMemo, useState } from 'react';
import { FaCheck } from 'react-icons/fa6';
import { useNavigate, useParams } from 'react-router-dom';
import Select from 'react-select';
import { ElementOutModel } from '../../types';
import { useAppContext } from '../../core/useAppContext';
import { useAnnotationSessionHistory } from '../../core/useHistory';
import { reorderLabels } from '../../core/utils';

interface MultilabelInputProps {
  elementId: string;
  labels: string[];
  postAnnotation: (label: string, elementId: string, comment?: string) => void;
  element?: ElementOutModel;
}

interface LabelType {
  id: number;
  label: string;
}

// Base hues for level-1 groups, well-spaced on the color wheel
const GROUP_HUES = [210, 340, 150, 30, 270, 180, 60, 310, 120, 240];

// Parsed label item: full original label + sub-category color index + display text
interface LabelItem {
  label: string; // full original label, e.g. "cat1>sub1>element"
  subIndex: number; // index of sub-category within the line (for color)
  displayName: string; // just the element name shown on button
}

interface LabelGroup {
  name: string; // level 1: line name
  subCategories: string[]; // unique level-2 names for legend
  items: LabelItem[];
}

/**
 * Parses labels with up to 3 levels separated by ">":
 *   level1 > level2 > level3
 *   line   > color  > button text
 *
 * Labels without ">" are placed in a single flat group.
 */
function parseLabels(labels: string[]): { groups: LabelGroup[]; isHierarchical: boolean } {
  const hasHierarchy = labels.some((l) => l.includes('>'));

  if (!hasHierarchy) {
    return {
      groups: [
        {
          name: '',
          subCategories: [],
          items: labels.map((l) => ({ label: l, subIndex: 0, displayName: l })),
        },
      ],
      isHierarchical: false,
    };
  }

  const groupMap = new Map<string, { subOrder: string[]; items: LabelItem[] }>();
  const groupOrder: string[] = [];

  for (const label of labels) {
    const parts = label.split('>').map((s) => s.trim());
    // level1 = line, level2 = color sub-category, level3+ = display name
    const lineName = parts[0];
    const subName = parts.length > 2 ? parts[1] : '';
    const displayName =
      parts.length > 2 ? parts.slice(2).join(' > ') : parts.length > 1 ? parts[1] : parts[0];

    if (!groupMap.has(lineName)) {
      groupMap.set(lineName, { subOrder: [], items: [] });
      groupOrder.push(lineName);
    }

    const group = groupMap.get(lineName)!;
    let subIndex = 0;
    if (subName) {
      if (!group.subOrder.includes(subName)) {
        group.subOrder.push(subName);
      }
      subIndex = group.subOrder.indexOf(subName);
    }

    group.items.push({ label, subIndex, displayName });
  }

  return {
    groups: groupOrder.map((name) => {
      const g = groupMap.get(name)!;
      return { name, subCategories: g.subOrder, items: g.items };
    }),
    isHierarchical: true,
  };
}

/**
 * Colors: each line (level 1) gets a base hue.
 * Within a line, each sub-category (level 2) shifts the saturation/lightness.
 */
function getButtonColors(
  groupIndex: number,
  subIndex: number,
  subCount: number,
  selected: boolean,
) {
  const hue = GROUP_HUES[groupIndex % GROUP_HUES.length];
  // Shift saturation per sub-category so they look distinct but related
  const satShift = subCount > 1 ? subIndex * Math.min(12, 40 / subCount) : 0;

  if (!selected) {
    return {
      backgroundColor: `hsl(${hue}, ${25 + satShift}%, ${92 - satShift * 0.5}%)`,
      color: `hsl(${hue}, 40%, 30%)`,
      borderColor: `hsl(${hue}, ${25 + satShift}%, ${77 - satShift * 0.5}%)`,
    };
  }

  return {
    backgroundColor: `hsl(${hue}, ${55 + satShift}%, ${42 - satShift * 0.4}%)`,
    color: '#fff',
    borderColor: `hsl(${hue}, ${55 + satShift}%, ${32 - satShift * 0.4}%)`,
  };
}

export const MultilabelInput: FC<MultilabelInputProps> = ({
  elementId,
  postAnnotation,
  labels,
  element,
}) => {
  const {
    appContext: { displayConfig },
  } = useAppContext();
  const { projectName } = useParams();
  const navigate = useNavigate();
  const { addElementInAnnotationSessionHistory } = useAnnotationSessionHistory();

  const [selectedLabels, setSelectedLabels] = useState<string[]>([]);
  const [comment, setComment] = useState<string>(
    element?.history ? element.history[0]?.comment || '' : '',
  );

  const skipAnnotation = useCallback(() => {
    if (element)
      addElementInAnnotationSessionHistory(
        element.element_id,
        element.text,
        undefined,
        undefined,
        true,
      );
    navigate(`/projects/${projectName}/tag/`);
  }, [navigate, projectName, addElementInAnnotationSessionHistory, element]);

  useEffect(() => setComment(element?.history ? element.history[0]?.comment || '' : ''), [element]);
  const availableLabels = useMemo<LabelType[]>(
    () =>
      reorderLabels(labels || [], displayConfig.labelsOrder || []).map((label, index) => ({
        id: index,
        label: label,
      })),
    [displayConfig.labelsOrder, labels],
  );

  const { groups, isHierarchical } = useMemo(
    () => parseLabels(availableLabels.map((o) => o.label)),
    [availableLabels],
  );

  const toggleLabel = useCallback((label: string) => {
    setSelectedLabels((prev) =>
      prev.includes(label) ? prev.filter((l) => l !== label) : [...prev, label],
    );
  }, []);

  const handleKeyboardEvents = useCallback(
    (ev: KeyboardEvent) => {
      const activeElement = document.activeElement;
      const isFormField =
        activeElement?.tagName === 'INPUT' ||
        activeElement?.tagName === 'TEXTAREA' ||
        activeElement?.tagName === 'SELECT';
      if (isFormField) return;

      if (ev.key === 'Enter' && ev.ctrlKey) {
        postAnnotation(selectedLabels.join('|'), elementId, comment);
        setSelectedLabels([]);
      }
      if (ev.code === 'KeyS') {
        skipAnnotation();
      }
    },
    [postAnnotation, selectedLabels, elementId, comment, skipAnnotation],
  );

  useEffect(() => {
    document.addEventListener('keydown', handleKeyboardEvents);
    return () => {
      document.removeEventListener('keydown', handleKeyboardEvents);
    };
  }, [handleKeyboardEvents]);

  return (
    <div className="d-flex flex-column gap-2 w-100">
      {/* Label buttons */}
      <div className="d-flex flex-column gap-2">
        {groups.map((group, groupIndex) => (
          <div key={group.name || groupIndex}>
            {isHierarchical && group.name && (
              <div
                className="fw-semibold mb-1"
                style={{
                  fontSize: '0.8rem',
                  color: `hsl(${GROUP_HUES[groupIndex % GROUP_HUES.length]}, 40%, 30%)`,
                }}
              >
                {group.name}
              </div>
            )}
            <div className="d-flex flex-wrap gap-1 align-items-center">
              {group.subCategories.length > 0
                ? group.subCategories.map((sub, subIdx) => {
                    const subItems = group.items.filter((i) => i.subIndex === subIdx);
                    const subColors = getButtonColors(
                      groupIndex,
                      subIdx,
                      group.subCategories.length,
                      false,
                    );
                    return (
                      <span key={sub} className="d-inline-flex align-items-center gap-1">
                        {subIdx > 0 && <span style={{ width: '0.25rem' }} />}
                        <span
                          style={{
                            fontSize: '0.7rem',
                            fontStyle: 'italic',
                            color: subColors.borderColor,
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {sub}
                        </span>
                        {subItems.map((item) => {
                          const selected = selectedLabels.includes(item.label);
                          const colors = getButtonColors(
                            groupIndex,
                            subIdx,
                            group.subCategories.length,
                            selected,
                          );
                          return (
                            <button
                              key={item.label}
                              type="button"
                              className="btn btn-sm"
                              style={{
                                backgroundColor: colors.backgroundColor,
                                color: colors.color,
                                border: `1px solid ${colors.borderColor}`,
                                fontWeight: selected ? 600 : 400,
                                transition: 'all 0.15s ease',
                              }}
                              onClick={() => toggleLabel(item.label)}
                              title={item.label}
                            >
                              {item.displayName}
                            </button>
                          );
                        })}
                      </span>
                    );
                  })
                : group.items.map((item) => {
                    const selected = selectedLabels.includes(item.label);
                    const colors = getButtonColors(groupIndex, 0, 0, selected);
                    return (
                      <button
                        key={item.label}
                        type="button"
                        className="btn btn-sm"
                        style={{
                          backgroundColor: colors.backgroundColor,
                          color: colors.color,
                          border: `1px solid ${colors.borderColor}`,
                          fontWeight: selected ? 600 : 400,
                          transition: 'all 0.15s ease',
                        }}
                        onClick={() => toggleLabel(item.label)}
                        title={item.label}
                      >
                        {item.displayName}
                      </button>
                    );
                  })}
            </div>
          </div>
        ))}
      </div>

      {/* Select input + validate */}
      <div className="d-flex gap-2 align-items-center w-100">
        <Select
          isMulti
          options={labels.map((e) => ({ value: e, label: e }))}
          onChange={(e) => {
            setSelectedLabels(e.map((e) => e.value));
          }}
          value={selectedLabels.map((e) => ({ value: e, label: e }))}
          className="w-100"
        />
        <button
          className="btn btn-success d-flex align-items-center justify-content-center gap-2 px-3 py-2 validate-btn"
          onClick={() => {
            postAnnotation(selectedLabels.join('|'), elementId, comment);
            setSelectedLabels([]);
          }}
        >
          <FaCheck size={18} />
          <span className="fw-semibold">Validate</span>
        </button>
        <button
          type="button"
          className="btn btn-secondary d-flex align-items-center justify-content-center gap-2 px-3 py-2"
          onClick={() => {
            skipAnnotation();
          }}
        >
          <span className="fw-semibold">Skip</span>
          <span className="badge hotkey">S</span>
        </button>
      </div>
      <textarea
        className="form-control annotation-comment"
        placeholder="Comment"
        value={comment}
        onChange={(e) => setComment(e.target.value)}
      />
    </div>
  );
};
