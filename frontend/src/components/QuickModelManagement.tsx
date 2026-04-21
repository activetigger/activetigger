import cx from 'classnames';
import { FC, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { SubmitHandler, useForm } from 'react-hook-form';
import { FaPlusCircle } from 'react-icons/fa';
import { FaGear } from 'react-icons/fa6';
import { IoIosRefresh } from 'react-icons/io';
import { MdDriveFileRenameOutline } from 'react-icons/md';
import {
  useDeleteQuickModel,
  useGetQuickModel,
  useRenameQuickModel,
  useRetrainQuickModel,
} from '../core/api';
import { useNotifications } from '../core/notifications';
import { sortDatesAsStrings } from '../core/utils';
import { MLStatisticsModel, ModelDescriptionModel } from '../types';
import { DisplayScores } from './DisplayScores';
import { ModelParametersTab } from './ModelParametersTab';
import { ModelsPillDisplay } from './ModelsPillDisplay';
import { StopProcessButton } from './StopProcessButton';
import { ValidateButtons } from './ValidateButton';
import { QuickModelForm } from './forms/QuickModelForm';

// TODO: default values + avoid generic parameters

interface Options {
  models?: string[];
}

interface FeaturesOptions {
  fasttext?: Options;
  'sentence-embeddings'?: Options;
}

interface QuickModelManagementProps {
  projectName: string | null;
  currentScheme: string | null;
  baseQuickModels: Record<string, Record<string, number>>;
  availableQuickModels: ModelDescriptionModel[];
  availableFeatures: string[];
  availableLabels: string[];
  kindScheme: string;
  currentModel?: Record<string, never>;
  featuresOption: FeaturesOptions;
  columns: string[];
  isComputing: boolean;
}

interface renameModel {
  new_name: string;
}

export const QuickModelManagement: FC<QuickModelManagementProps> = ({
  projectName,
  currentScheme,
  baseQuickModels,
  availableQuickModels,
  availableFeatures,
  availableLabels,
  kindScheme,
  isComputing,
}) => {
  const { notify } = useNotifications();

  // available features
  const features = availableFeatures
    .map((e) => ({ value: e, label: e }))
    .sort((a, b) => a.label.localeCompare(b.label));

  // current quickmodel
  const [currentQuickModelName, setCurrentQuickModelName] = useState<string | null>(
    availableQuickModels.length > 0 ? availableQuickModels[0].name : null,
  );

  const { retrainQuickModel } = useRetrainQuickModel(projectName || null, currentScheme || null);

  // Modal rename and form to rename
  const [showRename, setShowRename] = useState(false);
  const { renameQuickModel } = useRenameQuickModel(projectName || null);
  const {
    handleSubmit: handleSubmitRename,
    register: registerRename,
    reset: resetRename,
  } = useForm<renameModel>();

  const onSubmitRename: SubmitHandler<renameModel> = async (data) => {
    if (currentQuickModelName) {
      await renameQuickModel(currentQuickModelName, data.new_name);
      resetRename();
      setShowRename(false);
    } else notify({ type: 'error', message: 'Enter a name (currently empty)' });
  };

  // get information on the quickmodel
  const { currentModel: currentModelInformations } = useGetQuickModel(
    projectName,
    currentQuickModelName,
    currentQuickModelName,
  );

  // delete quickmodel
  const { deleteQuickModel } = useDeleteQuickModel(projectName);

  // state for new feature
  const [displayNewModel, setDisplayNewModel] = useState<boolean>(false);

  const [showParameters, setShowParameters] = useState(false);

  const cleanDisplay = (listOfFeatures: string, sep?: string) => {
    if (!sep) {
      sep = ' and ';
    }
    if (listOfFeatures) {
      return listOfFeatures
        .replaceAll('"', '')
        .replaceAll('[', '')
        .replaceAll(']', '')
        .replaceAll(',', sep);
    } else {
      return 'Loading...';
    }
  };

  return (
    <>
      <ModelsPillDisplay
        modelNames={availableQuickModels
          .sort((quickModelA, quickModelB) =>
            sortDatesAsStrings(quickModelA?.time, quickModelB?.time, true),
          )
          .map((quickModel) => quickModel.name)}
        currentModelName={currentQuickModelName}
        setCurrentModelName={setCurrentQuickModelName}
        deleteModelFunction={deleteQuickModel}
      >
        <button
          onClick={() => {
            setDisplayNewModel(true);
            setCurrentQuickModelName(null);
          }}
          className={cx('model-pill ', isComputing ? 'disabled' : '')}
          id="create-new"
        >
          <FaPlusCircle size={20} /> Create new model
        </button>
      </ModelsPillDisplay>

      <hr className="my-4" />

      {isComputing && <StopProcessButton projectSlug={projectName} />}

      {currentModelInformations && currentQuickModelName && (
        <>
          <div className="horizontal wrap">
            <button className="btn-secondary-action" onClick={() => setShowParameters(true)}>
              <FaGear size={18} className="me-1" />
              Parameters
            </button>
            <button
              className="btn-secondary-action"
              onClick={() => {
                retrainQuickModel(currentQuickModelName);
                console.log('retrain', currentQuickModelName);
              }}
            >
              <IoIosRefresh size={18} className="me-1" />
              Retrain
            </button>
            <button className="btn-secondary-action" onClick={() => setShowRename(true)}>
              <MdDriveFileRenameOutline size={18} className="me-1" />
              Rename
            </button>
            <ValidateButtons
              modelName={currentQuickModelName}
              kind="quick"
              id="compute-prediction"
              buttonLabel="Compute predictions"
            />
          </div>

          <DisplayScores
            title={'Validation scores from the training data (internal validation)'}
            scores={currentModelInformations.statistics_test as MLStatisticsModel}
            projectSlug={projectName}
          />
          {currentModelInformations.statistics_cv10 && (
            <>
              <h4 className="subsection">Cross Validation results</h4>
              <DisplayScores
                title="Cross validation CV10"
                scores={
                  currentModelInformations.statistics_cv10 as unknown as Record<string, number>
                }
              />
            </>
          )}
        </>
      )}

      <Modal
        show={displayNewModel}
        id="quickmodel-modal"
        onHide={() => setDisplayNewModel(false)}
        centered
        size="xl"
      >
        <Modal.Header closeButton>
          <Modal.Title>Train a new quick model</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <QuickModelForm
            projectSlug={projectName || ''}
            currentScheme={currentScheme || ''}
            kindScheme={kindScheme}
            baseQuickModels={baseQuickModels}
            features={features}
            availableLabels={availableLabels}
            setDisplayNewModel={setDisplayNewModel}
          />
        </Modal.Body>
      </Modal>
      <Modal show={showParameters} id="parameters-modal" onHide={() => setShowParameters(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Parameters of {currentQuickModelName}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {currentModelInformations && (
            <ModelParametersTab
              params={
                {
                  'Model type': currentModelInformations?.model,
                  'Input features': cleanDisplay(
                    JSON.stringify(currentModelInformations?.features) as unknown as string,
                    ', ',
                  ),
                  ...currentModelInformations?.params,
                } as Record<string, unknown>
              }
            />
          )}
        </Modal.Body>
      </Modal>
      <Modal show={showRename} id="rename-modal" onHide={() => setShowRename(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Rename {currentQuickModelName}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <form onSubmit={handleSubmitRename(onSubmitRename)}>
            <input
              id="new_name"
              type="text"
              placeholder="New name of the model"
              {...registerRename('new_name')}
            />
            <button className="btn-submit">Rename</button>
          </form>
        </Modal.Body>
      </Modal>
    </>
  );
};
