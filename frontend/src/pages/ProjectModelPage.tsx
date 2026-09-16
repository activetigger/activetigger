import cx from 'classnames';
import { FC, useEffect, useMemo, useState } from 'react';
import { Tab, Tabs } from 'react-bootstrap';
import { FaPlusCircle } from 'react-icons/fa';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { useParams, useSearchParams } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import { ImageModelEvaluation } from '../components/ImageModelEvaluation';
import { ImageModelManagement } from '../components/ImageModelManagement';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { ModelEvaluation } from '../components/ModelEvaluation';
import { ModelManagement } from '../components/ModelManagement';
import { ModelPredict } from '../components/ModelPredict';
import { ModelsPillDisplay } from '../components/ModelsPillDisplay';
import { QuickModelPredict } from '../components/QuickModelPredict';
import { useDeleteBertModel, useDeleteNerModel, useDeleteQuickModel } from '../core/api';
import { useAppContext } from '../core/useAppContext';
import { sortDatesAsStrings } from '../core/utils';

/**
 * Component to manage model training. Dispatches on project kind:
 * text projects get the BERT-based UI, image projects get the image-classification UI
 * (ViT, ConvNeXt, EfficientNet, ...).
 *
 * The model pills sit above the tabs: one selection shared by the
 * Training / Evaluation / Prediction views (same layout as the
 * generative panel). Image projects keep the previous per-tab layout.
 */

export const ProjectModelPage: FC = () => {
  const { projectName: projectSlug } = useParams();
  const {
    appContext: { currentProject, currentScheme, isComputing, developmentMode },
  } = useAppContext();

  // Allow deep-linking to a specific tab, e.g. /model/?tab=prediction from the Export page
  const [searchParams] = useSearchParams();
  const requestedTab = searchParams.get('tab');
  const [activeKey, setActiveKey] = useState<string>(
    requestedTab && ['models', 'evaluation', 'prediction'].includes(requestedTab)
      ? requestedTab
      : 'models',
  );
  const isImage = currentProject?.params?.kind === 'image';

  // For span schemes the "bert" slot targets the trained NER models;
  // everything else targets the BERT models. Same UI, same components.
  const kindScheme =
    currentScheme && currentProject?.schemes.available[currentScheme]
      ? currentProject.schemes.available[currentScheme].kind || 'multiclass'
      : 'multiclass';
  const isNer = kindScheme === 'span';
  const showNerCreate = isNer && developmentMode;

  // shared selection: one model at a time, across the two families
  const [selectedQuickModel, setSelectedQuickModel] = useState<string | null>(null);
  const [selectedBertModel, setSelectedBertModel] = useState<string | null>(null);
  useEffect(() => {
    if (selectedQuickModel) setSelectedBertModel(null);
  }, [selectedQuickModel]);
  useEffect(() => {
    if (selectedBertModel) setSelectedQuickModel(null);
  }, [selectedBertModel]);
  // models are scheme-bound: reset the selection when the scheme changes
  useEffect(() => {
    setSelectedQuickModel(null);
    setSelectedBertModel(null);
  }, [currentScheme]);

  // create buttons live in the pill rows; the forms live in the Training tab
  const [createRequest, setCreateRequest] = useState<'quick' | 'bert' | 'ner' | null>(null);
  const requestCreate = (kind: 'quick' | 'bert' | 'ner') => {
    setActiveKey('models');
    setCreateRequest(kind);
  };

  // available models, most recent first
  const availableQuickModels = useMemo(
    () =>
      (currentProject?.quickmodel.available[currentScheme || ''] || [])
        .slice()
        .sort((a, b) => sortDatesAsStrings(a?.time, b?.time, true))
        .map((m) => m.name),
    [currentProject?.quickmodel, currentScheme],
  );
  const bertModelsMap = isNer
    ? currentProject?.nermodels?.available?.[currentScheme || '']
    : currentProject?.languagemodels.available[currentScheme || ''];
  const availableBertModels = useMemo(
    () =>
      Object.values(bertModelsMap || {})
        .sort((a, b) => sortDatesAsStrings(a?.time, b?.time, true))
        .map((m) => (m ? m.name : '')),
    [bertModelsMap],
  );

  const { deleteQuickModel } = useDeleteQuickModel(projectSlug || null);
  const { deleteBertModel } = useDeleteBertModel(projectSlug || null);
  const { deleteNerModel } = useDeleteNerModel(projectSlug || null);

  return (
    <ProjectPageLayout projectName={projectSlug} currentAction="model">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            {isImage ? (
              <Tabs
                id="panel"
                className="mt-3"
                activeKey={activeKey}
                onSelect={(k) => setActiveKey(k || 'models')}
              >
                <Tab eventKey="models" title="Training">
                  <div className="explanations ms-3">Train quick and image models</div>
                  <ImageModelManagement />
                </Tab>
                <Tab eventKey="evaluation" title="Evaluation">
                  <div className="explanations ms-3">
                    Evaluate your models on annotations (train, eval and test)
                  </div>
                  <ImageModelEvaluation />
                </Tab>
              </Tabs>
            ) : (
              <>
                {!isNer && (
                  <div className="mt-3">
                    <span className="fw-semibold text-muted small">Quick Models</span>
                    <ModelsPillDisplay
                      modelNames={availableQuickModels}
                      currentModelName={selectedQuickModel}
                      setCurrentModelName={setSelectedQuickModel}
                      deleteModelFunction={async (name) => {
                        await deleteQuickModel(name);
                        if (selectedQuickModel === name) setSelectedQuickModel(null);
                      }}
                    >
                      <button
                        onClick={() => requestCreate('quick')}
                        className={cx('model-pill create-pill', isComputing && 'disabled')}
                        id="create-new-quick"
                      >
                        <FaPlusCircle size={20} /> Create new quick model
                      </button>
                    </ModelsPillDisplay>
                  </div>
                )}

                <div className="mt-2">
                  <span className="fw-semibold text-muted small">
                    {isNer ? 'NER Models' : 'BERT Models'}
                    {isNer && !developmentMode && ' (experimental — enable experimental mode)'}
                  </span>
                  <ModelsPillDisplay
                    modelNames={availableBertModels}
                    currentModelName={selectedBertModel}
                    setCurrentModelName={setSelectedBertModel}
                    deleteModelFunction={async (name) => {
                      await (isNer ? deleteNerModel(name) : deleteBertModel(name));
                      if (selectedBertModel === name) setSelectedBertModel(null);
                    }}
                  >
                    <button
                      onClick={() => {
                        if (isNer && (!showNerCreate || isComputing)) return;
                        requestCreate(isNer ? 'ner' : 'bert');
                      }}
                      className={cx(
                        'model-pill create-pill',
                        (isComputing || (isNer && !showNerCreate)) && 'disabled',
                      )}
                      disabled={isNer && (!showNerCreate || isComputing)}
                      id="create-new-bert"
                      style={
                        isNer && (!showNerCreate || isComputing) ? { cursor: 'not-allowed' } : {}
                      }
                    >
                      <FaPlusCircle size={20} /> Create new {isNer ? 'NER' : 'BERT'} model
                    </button>
                    <Tooltip anchorSelect="#create-new-bert">
                      {isNer && !showNerCreate
                        ? 'Enable experimental mode to train NER models'
                        : 'Train a model'}
                    </Tooltip>
                  </ModelsPillDisplay>
                </div>

                <Tabs
                  id="panel"
                  className="mt-3"
                  activeKey={activeKey}
                  onSelect={(k) => setActiveKey(k || 'models')}
                >
                  <Tab eventKey="models" title="Training">
                    <div className="explanations ms-3">Train quick and BERT models</div>
                    <ModelManagement
                      selectedQuickModel={selectedQuickModel}
                      selectedBertModel={selectedBertModel}
                      createRequest={createRequest}
                      onCreateHandled={() => setCreateRequest(null)}
                    />
                  </Tab>
                  <Tab eventKey="evaluation" title="Evaluation">
                    <div className="explanations ms-3">
                      Evaluate your models on annotations (train, eval and test){' '}
                      <a className="evaldataset">
                        <HiOutlineQuestionMarkCircle />
                      </a>
                      .
                    </div>
                    <Tooltip anchorSelect=".evaldataset" place="top">
                      Use validation statistics to choose the best model and test statistics for
                      final generalization scores of the best model (do not choose models based on
                      this)
                      <br />
                    </Tooltip>
                    <ModelEvaluation
                      selectedQuickModel={selectedQuickModel}
                      selectedBertModel={selectedBertModel}
                    />
                  </Tab>
                  <Tab eventKey="prediction" title="Prediction">
                    <div className="explanations ms-3">
                      Run a trained model on the full dataset or on an external dataset. Once a
                      prediction is computed, you can download it from the Export page.
                    </div>
                    {!selectedQuickModel && !selectedBertModel && (
                      <div className="text-muted ms-3 my-3">
                        Select a model above to run predictions
                      </div>
                    )}
                    {selectedQuickModel && <QuickModelPredict currentModel={selectedQuickModel} />}
                    {selectedBertModel && (
                      <div className="mt-3">
                        <ModelPredict
                          currentModel={selectedBertModel}
                          kind={isNer ? 'ner' : 'bert'}
                        />
                      </div>
                    )}
                  </Tab>
                </Tabs>
              </>
            )}
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
