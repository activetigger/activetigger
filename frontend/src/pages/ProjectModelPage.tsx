import { FC, useState } from 'react';
import { Tab, Tabs } from 'react-bootstrap';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { useParams } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import { ImageModelEvaluation } from '../components/ImageModelEvaluation';
import { ImageModelManagement } from '../components/ImageModelManagement';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { ModelEvaluation } from '../components/ModelEvaluation';
import { ModelManagement } from '../components/ModelManagement';
import { ModelPredict } from '../components/ModelPredict';
import { ModelsPillDisplay } from '../components/ModelsPillDisplay';
import { useAppContext } from '../core/useAppContext';

/**
 * Component to manage model training. Dispatches on project kind:
 * text projects get the BERT-based UI, image projects get the image-classification UI
 * (ViT, ConvNeXt, EfficientNet, ...).
 */

export const ProjectModelPage: FC = () => {
  const { projectName: projectSlug } = useParams();
  const {
    appContext: { currentProject, currentScheme },
  } = useAppContext();

  const [activeKey, setActiveKey] = useState<string>('models');
  const [predictionModel, setPredictionModel] = useState<string | null>(null);
  const isImage = currentProject?.params?.kind === 'image';

  // For span schemes the Prediction tab targets the trained NER models;
  // everything else targets the BERT models. Same UI, same component.
  const kindScheme =
    currentScheme && currentProject?.schemes.available[currentScheme]
      ? currentProject.schemes.available[currentScheme].kind || 'multiclass'
      : 'multiclass';
  const isNer = kindScheme === 'span';
  const predictKind = isNer ? 'ner' : 'bert';

  const availableBertModels = isNer
    ? currentScheme && currentProject?.nermodels?.available?.[currentScheme]
      ? Object.keys(currentProject.nermodels.available[currentScheme])
      : []
    : currentScheme && currentProject?.languagemodels.available[currentScheme]
      ? Object.keys(currentProject.languagemodels.available[currentScheme])
      : [];

  return (
    <ProjectPageLayout projectName={projectSlug} currentAction="model">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <Tabs
              id="panel"
              className="mt-3"
              activeKey={activeKey}
              onSelect={(k) => setActiveKey(k || 'models')}
            >
              <Tab eventKey="models" title="Training">
                <div className="explanations ms-3">
                  {isImage ? 'Train quick and image models' : 'Train quick and BERT models'}
                </div>
                {isImage ? <ImageModelManagement /> : <ModelManagement />}
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
                  Use validation statistics to choose the best model and test statistics for final
                  generalization scores of the best model (do not choose models based on this)
                  <br />
                </Tooltip>
                {isImage ? <ImageModelEvaluation /> : <ModelEvaluation />}
              </Tab>
              {!isImage && (
                <Tab eventKey="prediction" title="Prediction">
                  <div className="explanations ms-3">
                    Run a trained {isNer ? 'NER' : 'BERT'} model on the full dataset or on an
                    external dataset. Once a prediction is computed, you can download it from the
                    Export page.
                  </div>
                  <div className="ms-3 mt-3">
                    {availableBertModels.length === 0 ? (
                      <div className="text-muted small">
                        No {isNer ? 'NER' : 'BERT'} model available for the current scheme. Train
                        one in the Training tab first.
                      </div>
                    ) : (
                      <>
                        <div className="text-muted small mb-2">Select a model</div>
                        <ModelsPillDisplay
                          modelNames={availableBertModels}
                          currentModelName={predictionModel}
                          setCurrentModelName={setPredictionModel}
                        />
                        {predictionModel && (
                          <div className="mt-3">
                            <ModelPredict currentModel={predictionModel} kind={predictKind} />
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </Tab>
              )}
            </Tabs>
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
