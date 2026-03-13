import { FC, useEffect, useMemo, useState } from 'react';
import { Tab, Tabs } from 'react-bootstrap';
import { useParams } from 'react-router-dom';
import { DisplayScoresMenu } from '../components/DisplayScoresMenu';
import { DisplayTrainingProcesses } from '../components/DisplayTrainingProcesses';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { ModelsPillDisplay } from '../components/ModelsPillDisplay';
import { ValidateButtons } from '../components/ValidateButton';
import { useDeleteBertModel, useDeleteQuickModel, useModelInformations } from '../core/api';
import { useNotifications } from '../core/notifications';
import { useAppContext } from '../core/useAppContext';
import { sortDatesAsStrings } from '../core/utils';
import { MLStatisticsModel, ModelDescriptionModel } from '../types';

/**
 * Component to display the export page
 */
export const ProjectValidatePage: FC = () => {
  const { projectName } = useParams();
  const { notify } = useNotifications();

  const {
    appContext: { currentScheme, currentProject: project, isComputing },
  } = useAppContext();

  const availableBertModels = useMemo(
    () =>
      project?.languagemodels.available
        ? project?.languagemodels.available[currentScheme || '']
        : [],
    [project?.languagemodels.available, currentScheme],
  );

  const availableQuickModels = useMemo(
    () =>
      project?.quickmodel.available
        ? (project?.quickmodel.available[currentScheme || ''] as ModelDescriptionModel[])
        : [],
    [project?.quickmodel.available, currentScheme],
  );
  // model selected
  const [currentQuickModelName, setCurrentQuickModelName] = useState<string | null>(
    availableQuickModels ? availableQuickModels[0]?.name : null,
  );
  const [currentBertModelName, setCurrentBertModelName] = useState<string | null>(
    availableBertModels ? Object.keys(availableBertModels)[0] : null,
  );
  // delete quickmodel
  const { deleteQuickModel } = useDeleteQuickModel(projectName as string);
  const { deleteBertModel } = useDeleteBertModel(projectName as string);

  // get model information from api
  const { model: bertModelInformations, reFetch: reFetchBertModelInformations } =
    useModelInformations(projectName || null, currentBertModelName || null, 'bert', isComputing);
  const { model: quickModelInformations, reFetch: reFetchQuickModelInformations } =
    useModelInformations(projectName || null, currentQuickModelName || null, 'quick', isComputing);

  // refetch model informations when computation is done
  useEffect(() => {
    if (!isComputing) {
      reFetchBertModelInformations();
      reFetchQuickModelInformations();
      notify({ type: 'info', message: 'Score updated.' });
    }
  }, [isComputing, reFetchBertModelInformations, reFetchQuickModelInformations, notify]);

  console.log(isComputing);

  return (
    <ProjectPageLayout projectName={projectName} currentAction="validate">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <Tabs id="panel" className="mt-3" defaultActiveKey="quick">
              <Tab eventKey="quick" title="Quick">
                <div className="explanations">
                  Compute statistics on annotations for machine learning models
                </div>
                {availableQuickModels ? (
                  <ModelsPillDisplay
                    modelNames={availableQuickModels
                      .sort((quickModelA, quickModelB) =>
                        sortDatesAsStrings(quickModelA?.time, quickModelB?.time, true),
                      )
                      .map((quickModel) => quickModel.name)}
                    currentModelName={currentQuickModelName}
                    setCurrentModelName={setCurrentQuickModelName}
                    deleteModelFunction={deleteQuickModel}
                  ></ModelsPillDisplay>
                ) : (
                  <div className="alert alert-warning">No model available</div>
                )}
                <hr className="my-4" />
                {quickModelInformations && (
                  <>
                    <ValidateButtons
                      modelName={currentQuickModelName}
                      kind="quick"
                      id="compute-validate"
                      style={{ margin: '8px 0px', color: 'white' }}
                    />

                    <DisplayScoresMenu
                      scores={
                        quickModelInformations.scores as unknown as Record<
                          string,
                          MLStatisticsModel
                        >
                      }
                      modelName={currentQuickModelName || ''}
                      skip={['internalvalid_scores']}
                      projectSlug={projectName || null}
                    />
                  </>
                )}
              </Tab>
              <Tab eventKey="bert" title="BERT">
                <div className="explanations">
                  Compute statistics on annotations for BERT models
                </div>
                {availableBertModels ? (
                  <ModelsPillDisplay
                    modelNames={Object.values(availableBertModels || {})
                      .sort((bertModelA, bertModelB) =>
                        sortDatesAsStrings(bertModelA?.time, bertModelB?.time, true),
                      )
                      .map((model) => (model?.name ? model.name : ''))}
                    currentModelName={currentBertModelName}
                    setCurrentModelName={setCurrentBertModelName}
                    deleteModelFunction={deleteBertModel}
                  ></ModelsPillDisplay>
                ) : (
                  <div className="alert alert-warning">No model available</div>
                )}
                <hr className="my-4" />
                <div>
                  {bertModelInformations && !project?.params.test && (
                    <div className="col-12">
                      <div className="alert alert-warning m-4">
                        No testset available for this project. Please create one to compute
                        predictions on the project main page
                      </div>
                    </div>
                  )}

                  {isComputing && (
                    <DisplayTrainingProcesses
                      projectSlug={projectName || null}
                      processes={project?.languagemodels.training}
                      displayStopButton={isComputing}
                    />
                  )}

                  {bertModelInformations && (
                    <>
                      <ValidateButtons
                        modelName={currentBertModelName}
                        kind="bert"
                        id="compute-validate"
                        style={{ margin: '8px 0px', color: 'white' }}
                      />
                      <DisplayScoresMenu
                        scores={
                          bertModelInformations.scores as unknown as Record<
                            string,
                            MLStatisticsModel
                          >
                        }
                        modelName={currentQuickModelName || ''}
                        skip={['internalvalid_scores']}
                        projectSlug={projectName || null}
                      />
                    </>
                  )}
                </div>
              </Tab>
            </Tabs>
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
