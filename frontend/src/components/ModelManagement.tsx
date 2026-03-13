import cx from 'classnames';
import { FC, useEffect, useMemo, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { SubmitHandler, useForm } from 'react-hook-form';
import { FaPlusCircle } from 'react-icons/fa';
import { FaGear } from 'react-icons/fa6';
import { IoIosRefresh } from 'react-icons/io';
import { MdDriveFileRenameOutline } from 'react-icons/md';
import { useParams } from 'react-router-dom';
import {
  useDeleteBertModel,
  useDeleteQuickModel,
  useGetQuickModel,
  useModelInformations,
  useRenameBertModel,
  useRenameQuickModel,
  useRetrainQuickModel,
} from '../core/api';
import { useNotifications } from '../core/notifications';
import { useAppContext } from '../core/useAppContext';
import { useAuth } from '../core/useAuth';
import { sortDatesAsStrings } from '../core/utils';
import { MLStatisticsModel } from '../types';
import { DisplayScores } from './DisplayScores';
import { DisplayTrainingProcesses } from './DisplayTrainingProcesses';
import { ModelParametersTab } from './ModelParametersTab';
import { ModelsPillDisplay } from './ModelsPillDisplay';
import { ModelCreationForm } from './forms/ModelCreationForm';
import { QuickModelForm } from './forms/QuickModelForm';
import { LossChart } from './vizualisation/lossChart';

interface renameModel {
  new_name: string;
}

interface LossData {
  epoch: { [key: string]: number };
  val_loss: { [key: string]: number };
  val_eval_loss: { [key: string]: number };
}

export const ModelManagement: FC = () => {
  const { notify } = useNotifications();
  const { projectName: projectSlug } = useParams();
  const { authenticatedUser } = useAuth();
  const {
    appContext: { currentScheme, currentProject, isComputing, activeModel },
    setAppContext,
  } = useAppContext();
  const availableFeatures = currentProject?.features.available
    ? currentProject?.features.available
    : [];
  const [kindScheme] = useState<string>(
    currentScheme && currentProject && currentProject.schemes.available[currentScheme]
      ? currentProject.schemes.available[currentScheme].kind || 'multiclass'
      : 'multiclass',
  );
  const availableLabels =
    currentScheme && currentProject && currentProject.schemes.available[currentScheme]
      ? currentProject.schemes.available[currentScheme].labels
      : [];
  const features = availableFeatures.map((e) => ({ value: e, label: e }));

  // quickmodel
  const baseQuickModels = currentProject?.quickmodel.options
    ? currentProject?.quickmodel.options
    : {};
  const availableQuickModels = useMemo(
    () => currentProject?.quickmodel.available[currentScheme || ''] || [],
    [currentProject?.quickmodel, currentScheme],
  );
  const [currentQuickModelName, setCurrentQuickModelName] = useState<string | null>(null);
  const { retrainQuickModel } = useRetrainQuickModel(projectSlug || null, currentScheme || null);

  // bertmodel
  const [displayNewBertModel, setDisplayNewBertModel] = useState(false);
  const availableBertModels = useMemo(
    () => currentProject?.languagemodels.available[currentScheme || ''] || {},
    [currentProject?.languagemodels, currentScheme],
  );
  const [currentBertModel, setCurrentBertModel] = useState<string | null>(null);
  const { deleteBertModel } = useDeleteBertModel(projectSlug || null);
  const { model: currentBertModelInformations } = useModelInformations(
    projectSlug || null,
    currentBertModel || null,
    'bert',
    isComputing,
  );

  // Modal rename and form to rename
  const [showRename, setShowRename] = useState(false);
  const { renameQuickModel } = useRenameQuickModel(projectSlug || null);
  const {
    handleSubmit: handleSubmitRenameQuickModel,
    register: registerRenameQuickModel,
    reset: resetRenameQuickModel,
  } = useForm<renameModel>();

  const onSubmitRenameQuickModel: SubmitHandler<renameModel> = async (data) => {
    if (currentQuickModelName) {
      await renameQuickModel(currentQuickModelName, data.new_name);
      resetRenameQuickModel();
      setShowRename(false);
    } else notify({ type: 'error', message: 'Enter a name (currently empty)' });
  };

  const { renameBertModel } = useRenameBertModel(projectSlug || null);
  const {
    handleSubmit: handleSubmitRenameBertModel,
    register: registerRenameBertModel,
    reset: resetRenameBertModel,
  } = useForm<renameModel>();

  const onSubmitRenameBertModel: SubmitHandler<renameModel> = async (data) => {
    if (currentBertModel) {
      await renameBertModel(currentBertModel, data.new_name);
      resetRenameBertModel();
      setShowRename(false);
    } else notify({ type: 'error', message: 'Enter a name (currently empty)' });
  };

  // get information on the quickmodel
  const { currentModel: currentQuickModelInformations, reFetchQuickModel } = useGetQuickModel(
    projectSlug || null,
    currentQuickModelName,
    currentQuickModelName,
  );
  useEffect(() => {
    if (currentQuickModelInformations) {
      reFetchQuickModel();
    }
  }, [currentQuickModelInformations, isComputing, reFetchQuickModel]);

  // delete quickmodel
  const { deleteQuickModel } = useDeleteQuickModel(projectSlug || null);

  // state for new feature
  const [displayNewModel, setDisplayNewModel] = useState<boolean>(false);

  const [showParametersQuickModel, setShowParametersQuickModel] = useState(false);
  const [showParametersBertModel, setShowParametersBertModel] = useState(false);

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

  const loss = currentBertModelInformations?.loss
    ? (currentBertModelInformations?.loss as unknown as LossData)
    : null;

  // meta selector
  const [currentModel, setCurrentModel] = useState<{ name: string; kind: string } | null>(null);
  useEffect(() => {
    if (currentQuickModelName) {
      setCurrentModel({ name: currentQuickModelName, kind: 'quick' });
      setCurrentBertModel(null);
    }
  }, [currentQuickModelName]);
  useEffect(() => {
    if (currentBertModel) {
      setCurrentModel({ name: currentBertModel, kind: 'bert' });
      setCurrentQuickModelName(null);
    }
  }, [currentBertModel]);

  // deactivate currents active model if it has been deleted from the list
  useEffect(() => {
    if (
      activeModel &&
      activeModel.type === 'quick' &&
      !availableQuickModels.map((m) => m.name).includes(activeModel.value)
    ) {
      setAppContext((prev) => ({ ...prev, activeModel: null }));
      notify({
        type: 'warning',
        message: `The active model ${activeModel.value} has been deleted, it has been deactivated for active learning.`,
      });
    }
    if (
      activeModel &&
      activeModel.type === 'bert' &&
      !Object.values(availableBertModels)
        .map((m) => m?.name)
        .includes(activeModel.value)
    ) {
      setAppContext((prev) => ({ ...prev, activeModel: null }));
      notify({
        type: 'warning',
        message: `The active model ${activeModel.value} has been deleted, it has been deactivated for active learning.`,
      });
    }
  }, [
    availableQuickModels,
    availableBertModels,
    activeModel,
    setAppContext,
    notify,
    currentProject,
  ]);

  return (
    <>
      <span className="fw-semibold text-muted small">Quick Models</span>
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
          <FaPlusCircle size={20} /> Create new quick model
        </button>
      </ModelsPillDisplay>

      <span className="fw-semibold text-muted small">Bert Models</span>
      <ModelsPillDisplay
        modelNames={Object.values(availableBertModels)
          .sort((bertModelA, bertModelB) =>
            sortDatesAsStrings(bertModelA?.time, bertModelB?.time, true),
          )
          .map((model) => (model ? model.name : ''))}
        currentModelName={currentBertModel}
        setCurrentModelName={setCurrentBertModel}
        deleteModelFunction={deleteBertModel}
      >
        <button
          onClick={() => {
            setDisplayNewBertModel(true);
            setCurrentBertModel(null);
          }}
          className={cx('model-pill ', isComputing ? 'disabled' : '')}
          id="create-new"
        >
          <FaPlusCircle size={20} /> Create new BERT model
        </button>
      </ModelsPillDisplay>

      {isComputing && authenticatedUser?.username && (
        <DisplayTrainingProcesses
          projectSlug={projectSlug || null}
          processes={
            currentProject?.languagemodels.training?.[authenticatedUser.username]
              ? {
                  [authenticatedUser.username]:
                    currentProject.languagemodels.training[authenticatedUser.username],
                }
              : undefined
          }
          displayStopButton={isComputing}
        />
      )}

      <hr className="my-4" />

      {currentModel &&
        currentModel.kind === 'quick' &&
        currentQuickModelInformations &&
        currentQuickModelInformations.params && (
          <>
            <DisplayScores
              title={'Validation scores from the training data (internal validation)'}
              scores={currentQuickModelInformations.statistics_test as MLStatisticsModel}
              projectSlug={projectSlug}
              dataset="Train-Eval"
              exclude_labels={currentQuickModelInformations.exclude_labels}
            />
            <div className="horizontal wrap">
              <button
                className="btn-secondary-action"
                onClick={() => {
                  retrainQuickModel(currentQuickModelName || '');
                }}
              >
                <IoIosRefresh size={18} className="me-1" />
                Retrain
              </button>
              <button
                className="btn-secondary-action"
                onClick={() => {
                  setShowParametersQuickModel(true);
                }}
              >
                <FaGear size={18} className="me-1" />
                Parameters
              </button>

              <button className="btn-secondary-action" onClick={() => setShowRename(true)}>
                <MdDriveFileRenameOutline size={18} className="me-1" />
                Rename
              </button>
            </div>
            {currentQuickModelInformations.statistics_cv10 && (
              <>
                <h4 className="subsection">Cross Validation results</h4>
                <DisplayScores
                  title="Cross validation CV10"
                  scores={
                    currentQuickModelInformations.statistics_cv10 as unknown as Record<
                      string,
                      number
                    >
                  }
                  dataset="train test"
                />
              </>
            )}
          </>
        )}

      {currentModel && currentModel.kind === 'bert' && currentBertModelInformations && (
        <div>
          <div className="my-3"></div>
          <DisplayScores
            title={'Validation scores from the training data (internal validation)'}
            scores={currentBertModelInformations.scores.internalvalid_scores as MLStatisticsModel}
            modelName={currentBertModel || ''}
            projectSlug={projectSlug}
            dataset="Train-Eval"
          />
          <div className="horizontal wrap">
            <button
              className="btn-secondary-action"
              onClick={() => setShowParametersBertModel(true)}
            >
              <FaGear size={18} />
              Parameters
            </button>
            <button className="btn-secondary-action" onClick={() => setShowRename(true)}>
              <MdDriveFileRenameOutline size={18} className="me-1" />
              Rename
            </button>
          </div>

          <div style={{ width: '100%', height: '500px' }} className="my-4">
            <LossChart loss={loss} />
          </div>
        </div>
      )}

      {/* Modals for models */}

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
            projectSlug={projectSlug || ''}
            currentScheme={currentScheme || ''}
            kindScheme={kindScheme}
            baseQuickModels={baseQuickModels as Record<string, Record<string, number>>}
            features={features}
            availableLabels={availableLabels}
            setDisplayNewModel={setDisplayNewModel}
          />
        </Modal.Body>
      </Modal>

      <Modal
        show={showParametersQuickModel}
        id="parameters-modal"
        onHide={() => setShowParametersQuickModel(false)}
      >
        <Modal.Header closeButton>
          <Modal.Title>Parameters of {currentQuickModelInformations?.name}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {currentQuickModelInformations && (
            <ModelParametersTab
              params={
                {
                  'Model type': currentQuickModelInformations?.model,
                  'Input features': cleanDisplay(
                    JSON.stringify(currentQuickModelInformations?.features) as unknown as string,
                    ', ',
                  ),
                  'Balanced classes': currentQuickModelInformations?.balance_classes,
                  ...currentQuickModelInformations?.params,
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
          <form onSubmit={handleSubmitRenameQuickModel(onSubmitRenameQuickModel)}>
            <input
              id="new_name"
              type="text"
              placeholder="New name of the model"
              {...registerRenameQuickModel('new_name')}
            />
            <button className="btn-submit">Rename</button>
          </form>
        </Modal.Body>
      </Modal>
      <Modal
        show={displayNewBertModel}
        id="createmodel-modal"
        size="xl"
        onHide={() => setDisplayNewBertModel(false)}
      >
        <Modal.Header closeButton>
          <Modal.Title>Train a new BERT model</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <ModelCreationForm
            projectSlug={projectSlug || null}
            currentScheme={currentScheme || null}
            currentProject={currentProject || null}
            isComputing={isComputing}
            setStatusDisplay={setDisplayNewBertModel}
          />
        </Modal.Body>
      </Modal>
      <Modal
        show={showParametersBertModel}
        id="parameters-modal"
        onHide={() => setShowParametersBertModel(false)}
      >
        <Modal.Header closeButton>
          <Modal.Title>Parameters of {currentBertModel}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <ModelParametersTab
            params={currentBertModelInformations?.params as Record<string, unknown>}
          />
        </Modal.Body>
      </Modal>
      <Modal show={showRename} id="rename-modal" onHide={() => setShowRename(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Rename {currentBertModel}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <form onSubmit={handleSubmitRenameBertModel(onSubmitRenameBertModel)}>
            <input
              id="new_name"
              type="text"
              placeholder="New name of the model"
              {...registerRenameBertModel('new_name')}
            />
            <button className="btn-submit">Rename</button>
          </form>
        </Modal.Body>
      </Modal>
    </>
  );
};
