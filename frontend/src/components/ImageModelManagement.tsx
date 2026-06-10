import cx from 'classnames';
import { FC, useEffect, useMemo, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { SubmitHandler, useForm } from 'react-hook-form';
import { FaPlusCircle } from 'react-icons/fa';
import { FaGear } from 'react-icons/fa6';
import { IoIosRefresh } from 'react-icons/io';
import { MdDriveFileRenameOutline } from 'react-icons/md';
import { useParams } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import {
  useDeleteQuickModel,
  useDeleteImageModel,
  useGetQuickModel,
  useModelInformations,
  useRenameQuickModel,
  useRenameImageModel,
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
import { ValidateButtons } from './ValidateButton';
import { QuickModelForm } from './forms/QuickModelForm';
import { ImageModelCreationForm } from './forms/ImageModelCreationForm';
import { LossChart } from './vizualisation/lossChart';

interface renameModel {
  new_name: string;
}

interface LossData {
  epoch: { [key: string]: number };
  val_loss: { [key: string]: number };
  val_eval_loss: { [key: string]: number };
}

/**
 * Image-project counterpart of ModelManagement: Quick Models + image-classification fine-tuning
 * (ViT, ConvNeXt, EfficientNet, ...).
 */
export const ImageModelManagement: FC = () => {
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
  const features = availableFeatures
    .map((e) => ({ value: e, label: e }))
    .sort((a, b) => a.label.localeCompare(b.label));

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

  // image model
  const [displayNewImageModel, setDisplayNewImageModel] = useState(false);
  const availableImageModels = useMemo(
    () => currentProject?.imagemodels?.available[currentScheme || ''] || {},
    [currentProject?.imagemodels, currentScheme],
  );
  const [currentImageModel, setCurrentImageModel] = useState<string | null>(null);
  const { deleteImageModel } = useDeleteImageModel(projectSlug || null);
  const { model: currentImageModelInformations } = useModelInformations(
    projectSlug || null,
    currentImageModel || null,
    'image',
    isComputing,
  );

  // Rename
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

  const { renameImageModel } = useRenameImageModel(projectSlug || null);
  const {
    handleSubmit: handleSubmitRenameImageModel,
    register: registerRenameImageModel,
    reset: resetRenameImageModel,
  } = useForm<renameModel>();

  const onSubmitRenameImageModel: SubmitHandler<renameModel> = async (data) => {
    if (currentImageModel) {
      await renameImageModel(currentImageModel, data.new_name);
      resetRenameImageModel();
      setShowRename(false);
    } else notify({ type: 'error', message: 'Enter a name (currently empty)' });
  };

  // Quick model info
  const { currentModel: currentQuickModelInformations, reFetchQuickModel } = useGetQuickModel(
    projectSlug || null,
    currentQuickModelName,
    currentQuickModelName,
  );
  useEffect(() => {
    if (currentQuickModelName) {
      reFetchQuickModel();
    }
  }, [isComputing, currentQuickModelName, reFetchQuickModel]);

  const { deleteQuickModel } = useDeleteQuickModel(projectSlug || null);

  const [displayNewModel, setDisplayNewModel] = useState<boolean>(false);
  const [showParametersQuickModel, setShowParametersQuickModel] = useState(false);
  const [showParametersImageModel, setShowParametersImageModel] = useState(false);

  const cleanDisplay = (listOfFeatures: string, sep?: string) => {
    if (!sep) sep = ' and ';
    if (listOfFeatures) {
      return listOfFeatures
        .replaceAll('"', '')
        .replaceAll('[', '')
        .replaceAll(']', '')
        .replaceAll(',', sep);
    }
    return 'Loading...';
  };

  const loss = currentImageModelInformations?.loss
    ? (currentImageModelInformations?.loss as unknown as LossData)
    : null;

  // meta selector
  const [currentModel, setCurrentModel] = useState<{ name: string; kind: string } | null>(null);
  useEffect(() => {
    if (currentQuickModelName) {
      setCurrentModel({ name: currentQuickModelName, kind: 'quick' });
      setCurrentImageModel(null);
    }
  }, [currentQuickModelName]);
  useEffect(() => {
    if (currentImageModel) {
      setCurrentModel({ name: currentImageModel, kind: 'image' });
      setCurrentQuickModelName(null);
    }
  }, [currentImageModel]);

  // deactivate active model if deleted
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
      activeModel.type === 'imagemodel' &&
      !Object.values(availableImageModels)
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
    availableImageModels,
    activeModel,
    setAppContext,
    notify,
    currentProject,
  ]);

  return (
    <>
      {kindScheme === 'multilabel' && (
        <div className="alert alert-info" role="alert">
          Multilabel scheme: quick models are trained on dichotomized labels (one-vs-rest), while
          image-classification models support native multilabel classification.
        </div>
      )}

      <span className="fw-semibold text-muted small">Quick Models</span>
      <ModelsPillDisplay
        modelNames={availableQuickModels
          .sort((a, b) => sortDatesAsStrings(a?.time, b?.time, true))
          .map((m) => m.name)}
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
          disabled={kindScheme === 'span'}
          id="create-new"
          style={kindScheme === 'span' ? { cursor: 'not-allowed' } : {}}
        >
          <FaPlusCircle size={20} /> Create new quick model
        </button>
      </ModelsPillDisplay>

      <span className="fw-semibold text-muted small">Image Models</span>
      <ModelsPillDisplay
        modelNames={Object.values(availableImageModels)
          .sort((a, b) => sortDatesAsStrings(a?.time, b?.time, true))
          .map((m) => (m ? m.name : ''))}
        currentModelName={currentImageModel}
        setCurrentModelName={setCurrentImageModel}
        deleteModelFunction={deleteImageModel}
      >
        <button
          onClick={() => {
            setDisplayNewImageModel(true);
            setCurrentImageModel(null);
          }}
          className={cx('model-pill ', isComputing ? 'disabled' : '')}
          disabled={kindScheme === 'span'}
          id="create-new-image"
          style={kindScheme === 'span' ? { cursor: 'not-allowed' } : {}}
        >
          <FaPlusCircle size={20} /> Create new image model
        </button>
        <Tooltip anchorSelect="#create-new-image">
          {kindScheme === 'span'
            ? 'Span classification is not supported for image models'
            : 'Fine-tune an image-classification model'}
        </Tooltip>
      </ModelsPillDisplay>

      {isComputing && authenticatedUser?.username && (
        <DisplayTrainingProcesses
          projectSlug={projectSlug || null}
          processes={
            currentProject?.imagemodels?.training?.[authenticatedUser.username]
              ? {
                  [authenticatedUser.username]:
                    currentProject.imagemodels.training[authenticatedUser.username],
                }
              : undefined
          }
          displayStopButton={isComputing}
          stopKind="image"
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
                onClick={() => setShowParametersQuickModel(true)}
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

      {currentModel && currentModel.kind === 'image' && currentImageModelInformations && (
        <div>
          <ValidateButtons
            modelName={currentImageModel}
            kind="image"
            id="compute-prediction-training"
            buttonLabel="Compute predictions"
          />
          <div className="my-3"></div>
          <DisplayScores
            title={'Validation scores from the training data (internal validation)'}
            scores={currentImageModelInformations.scores.internalvalid_scores as MLStatisticsModel}
            modelName={currentImageModel || ''}
            projectSlug={projectSlug}
            dataset="Train-Eval"
          />
          <div className="horizontal wrap">
            <button
              className="btn-secondary-action"
              onClick={() => setShowParametersImageModel(true)}
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

      {/* Modals */}

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
          <Modal.Title>
            Rename {currentModel?.kind === 'image' ? currentImageModel : currentQuickModelName}
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {currentModel?.kind === 'image' ? (
            <form onSubmit={handleSubmitRenameImageModel(onSubmitRenameImageModel)}>
              <input
                id="new_name"
                type="text"
                placeholder="New name of the model"
                {...registerRenameImageModel('new_name')}
              />
              <button className="btn-submit">Rename</button>
            </form>
          ) : (
            <form onSubmit={handleSubmitRenameQuickModel(onSubmitRenameQuickModel)}>
              <input
                id="new_name"
                type="text"
                placeholder="New name of the model"
                {...registerRenameQuickModel('new_name')}
              />
              <button className="btn-submit">Rename</button>
            </form>
          )}
        </Modal.Body>
      </Modal>

      <Modal
        show={displayNewImageModel}
        id="createmodel-modal"
        size="xl"
        onHide={() => setDisplayNewImageModel(false)}
      >
        <Modal.Header closeButton>
          <Modal.Title>Train a new image model</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <ImageModelCreationForm
            projectSlug={projectSlug || null}
            currentScheme={currentScheme || null}
            currentProject={currentProject || null}
            isComputing={isComputing}
            setStatusDisplay={setDisplayNewImageModel}
          />
        </Modal.Body>
      </Modal>

      <Modal
        show={showParametersImageModel}
        id="parameters-modal"
        onHide={() => setShowParametersImageModel(false)}
      >
        <Modal.Header closeButton>
          <Modal.Title>Parameters of {currentImageModel}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <ModelParametersTab
            params={currentImageModelInformations?.params as Record<string, unknown>}
          />
        </Modal.Body>
      </Modal>
    </>
  );
};
