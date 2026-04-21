import classNames from 'classnames';
import { ChangeEvent, Dispatch, FC, SetStateAction, useEffect, useMemo, useState } from 'react';
import { FaMapMarkedAlt } from 'react-icons/fa';
import { GiTigerHead } from 'react-icons/gi';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { LuRefreshCw } from 'react-icons/lu';
import { MdDisplaySettings } from 'react-icons/md';
import Select from 'react-select';
import { Tooltip } from 'react-tooltip';

import { keys, sortBy } from 'lodash';
import { useDebounceValue } from 'usehooks-ts';
import { useGetQuickModel, useStatistics } from '../../core/api';
import { useAppContext } from '../../core/useAppContext';
import { isValidRegex } from '../../core/utils';
import { AnnotationTagFilterSelect } from './AnnotationTagFilterSelect';

interface AnnotationModeFormProps {
  fetchNextElement: () => void;
  setActiveMenu: Dispatch<SetStateAction<boolean>>;
  setShowDisplayViz: Dispatch<SetStateAction<boolean>>;
  setShowDisplayConfig: Dispatch<SetStateAction<boolean>>;
  nSample: number | null;
  statistics: ReturnType<typeof useStatistics>['statistics'];
}

function optionValue(option: Record<string, string | undefined>) {
  const value = sortBy(keys(option))
    .map((k) => option[k])
    .join('|');
  return value;
}

// define the component to configure selection mode
export const AnnotationModeForm: FC<AnnotationModeFormProps> = ({
  fetchNextElement,
  setActiveMenu,
  setShowDisplayViz,
  setShowDisplayConfig,
  nSample,
  statistics,
}) => {
  const {
    appContext: {
      currentScheme,
      selectionConfig,
      currentProject: project,
      activeModel,
      phase,
      currentProjection,
    },
    setAppContext,
  } = useAppContext();

  // API call to get the current model & refetch
  // TODO : MODEL SELECTION TO CHANGE
  const name = null;
  const { currentModel } = useGetQuickModel(
    project ? project.params.project_slug : null,
    name,
    project,
  );

  const [availableLabels, setAvailableLabels] = useState<string[]>([]);

  const [filterDebounced, setFilter] = useDebounceValue(selectionConfig.filter, 500);
  useEffect(() => {
    setAppContext((prev) => ({
      ...prev,
      selectionConfig: { ...prev.selectionConfig, filter: filterDebounced },
    }));
  }, [filterDebounced, setAppContext]);

  const statisticsDataset = useMemo(() => {
    if (phase === 'train') return `${statistics?.train_annotated_n}/${statistics?.train_set_n}`;
    if (phase === 'valid') return `${statistics?.valid_annotated_n}/${statistics?.valid_set_n}`;
    if (phase === 'test') return `${statistics?.test_annotated_n}/${statistics?.test_set_n}`;
    return '';
  }, [phase, statistics]);

  // keep availableLabels up to date
  useEffect(() => {
    // case where the quick model is dichotomize on a specific label
    if (currentModel && currentModel.params && currentModel.params['dichotomize']) {
      setAvailableLabels([
        currentModel.params['dichotomize'] as string,
        'not-' + currentModel.params['dichotomize'],
      ]);
    } else if (currentScheme && project?.schemes.available[currentScheme])
      setAvailableLabels(project.schemes.available[currentScheme].labels);
    else setAvailableLabels([]);
  }, [currentModel, setAvailableLabels, currentScheme, project?.schemes.available]);

  // change dataset : there should be a navigation to reset element id
  const changeDataSet = (e: ChangeEvent<HTMLSelectElement>) => {
    setAppContext((prev) => ({
      ...prev,
      phase: e.target.value,
    }));
  };

  const isValid = project?.params.valid;
  const isTest = project?.params.test;

  const selectionModeOptions: { mode: string; label_prob?: string; value: string }[] =
    useMemo(() => {
      const modes = (
        (phase === 'train' && activeModel
          ? project?.next.methods.filter((m) => m !== 'maxprob')
          : project?.next.methods_min) || []
      ).map((mode) => ({ mode, label_prob: undefined }));
      const probLabels =
        phase === 'train' && activeModel
          ? availableLabels
              .filter((l) => !activeModel.labels_excluded.includes(l))
              .flatMap((l) => [
                {
                  mode: 'maxprob',
                  label_prob: l,
                },
                {
                  mode: 'active',
                  label_prob: l,
                },
              ])
          : [];
      return [...modes, ...probLabels].map((o) => ({ ...o, value: optionValue(o) }));
    }, [phase, activeModel, project?.next.methods, project?.next.methods_min, availableLabels]);

  // reset selection mode to "fixed" when the active model is deactivated
  // and the current mode requires a model (e.g. maxprob, active)
  useEffect(() => {
    if (!activeModel && !['fixed', 'random'].includes(selectionConfig.mode)) {
      const availableModes = project?.next.methods_min || [];
      if (!availableModes.includes(selectionConfig.mode)) {
        setAppContext((prev) => ({
          ...prev,
          selectionConfig: {
            ...prev.selectionConfig,
            mode: 'fixed',
            label_prob: undefined,
          },
        }));
      }
    }
  }, [activeModel, selectionConfig.mode, project?.next.methods_min, setAppContext]);

  return (
    <form className="annotation-mode">
      {/* left container: main selectors */}
      <div>
        {/* PHASE - DATASET */}
        <div className="at-input-group">
          <label className=" small-gray">Dataset</label>
          <select
            value={phase}
            onChange={(e) => {
              changeDataSet(e);
            }}
          >
            <option value="train">train</option>
            {isValid && <option value="valid">validation</option>}
            {isTest && <option value="test">test</option>}
          </select>
        </div>
        {/* Active Mode */}
        {phase === 'train' && (
          <div className="at-input-group">
            <label className=" small-gray">Active mode</label>
            <div>
              <button
                className="button active-mode-button"
                type="button"
                onClick={() => setActiveMenu((prev) => !prev)}
              >
                <GiTigerHead
                  size={30}
                  className="activelearning"
                  style={{ color: activeModel ? 'green' : 'grey', cursor: 'pointer' }}
                  title="Active learning"
                />
                <span className="text-muted">{activeModel ? activeModel.value : 'inactive'}</span>{' '}
              </button>
            </div>
          </div>
        )}

        <div className="at-input-group">
          <label className="small-gray">Selection method</label>
          <Select
            className="react-select"
            options={selectionModeOptions}
            value={selectionModeOptions.find(
              (o) =>
                o.value ===
                optionValue({
                  mode: selectionConfig.mode,
                  label_prob: selectionConfig.label_prob,
                }),
            )}
            getOptionLabel={(o) =>
              o.mode === 'maxprob' && o.label_prob
                ? `max pred ${o.label_prob}`
                : o.mode === 'active' && o.label_prob
                  ? `active ${o.label_prob}`
                  : o.mode
            }
            onChange={(option) => {
              if (option !== null) {
                setAppContext((prev) => ({
                  ...prev,
                  selectionConfig: {
                    ...prev.selectionConfig,
                    mode: option.mode,
                    label_prob: option.label_prob,
                  },
                }));
              }
            }}
          />
        </div>
      </div>
      {/* CONTENT */}
      <div>
        <div className="at-input-group">
          <label className=" small-gray">Filter by Tag/Users</label>
          <AnnotationTagFilterSelect availableLabels={availableLabels} />
        </div>

        {
          // input validated on deselect
        }
        <div className="at-input-group">
          <label htmlFor="select_regex" className=" small-gray">
            Filter by content
            <HiOutlineQuestionMarkCircle id="regex-tooltip" />
          </label>
          <input
            className={classNames(
              'searchhelp',
              filterDebounced && !isValidRegex(filterDebounced) ? 'is-invalid' : '',
            )}
            type="text"
            id="select_regex"
            placeholder="Enter a regex"
            defaultValue={selectionConfig.filter}
            onChange={(e) => {
              setFilter(e.target.value);
            }}
          />
          <div className="invalid-feedback">Regex not valid</div>
          <Tooltip anchorSelect="#regex-tooltip">
            Use CONTEXT= or QUERY= for specific requests
          </Tooltip>
        </div>
        {currentProjection && (
          <div>
            {/* LOCK on UMAP */}

            <div className="at-input-group">
              <label className=" small-gray">Filter by Projection</label>
              <div>
                <button
                  className="button"
                  type="button"
                  onClick={() => setShowDisplayViz((p) => !p)}
                >
                  <FaMapMarkedAlt
                    size={30}
                    style={{
                      color: selectionConfig.frameSelection ? 'green' : 'grey',
                      cursor: 'pointer',
                    }}
                    title="Map frame selection"
                    id="map-icon"
                  />
                </button>
                <span className="badge info">
                  {selectionConfig.frameSelection ? 'active' : 'inactive'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
      <div className="flex-grow-1 align-items-center">
        <small className="d-flex text-muted text-end flex-column justify-content-between flex-grow-1">
          {statistics ? (
            <>
              <span>Annotated:&nbsp;{statisticsDataset}</span>
              <span>Selection:&nbsp;{nSample || 'na'}</span>
            </>
          ) : (
            'na'
          )}
        </small>

        <button
          type="button"
          className="btn-primary-action"
          onClick={() => {
            fetchNextElement();
          }}
          title="Get next element with the selection mode"
        >
          <LuRefreshCw size={20} />
          <Tooltip anchorSelect=".getelement" place="top">
            Get next element with the selection mode
          </Tooltip>
        </button>

        <button
          type="button"
          className="btn-secondary-action"
          onClick={() => {
            setShowDisplayConfig((p) => !p);
          }}
          title="Display config menu"
        >
          <MdDisplaySettings size={20} />
        </button>
      </div>
    </form>
  );
};
