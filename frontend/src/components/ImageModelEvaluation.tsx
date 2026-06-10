import { FC, useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useDeleteQuickModel, useDeleteImageModel, useModelInformations } from '../core/api';
import { useAppContext } from '../core/useAppContext';
import { sortDatesAsStrings } from '../core/utils';
import { MLStatisticsModel } from '../types';
import { DisplayScoresMenu } from './DisplayScoresMenu';
import { DisplayTrainingProcesses } from './DisplayTrainingProcesses';
import { ModelsPillDisplay } from './ModelsPillDisplay';
import { ValidateButtons } from './ValidateButton';

/**
 * Image-project counterpart of ModelEvaluation: shows scores for Quick + ViT models.
 */
export const ImageModelEvaluation: FC = () => {
  const { projectName: projectSlug } = useParams();
  const {
    appContext: { currentScheme, currentProject, isComputing },
  } = useAppContext();

  // quickmodel selector
  const availableQuickModels = useMemo(
    () => currentProject?.quickmodel.available[currentScheme || ''] || [],
    [currentProject?.quickmodel, currentScheme],
  );
  const [currentQuickModelName, setCurrentQuickModelName] = useState<string | null>(null);
  const { deleteQuickModel } = useDeleteQuickModel(projectSlug || null);

  // image model selector
  const availableImageModels = useMemo(
    () => currentProject?.imagemodels?.available[currentScheme || ''] || {},
    [currentProject?.imagemodels, currentScheme],
  );
  const [currentImageModel, setCurrentImageModel] = useState<string | null>(null);
  const { deleteImageModel } = useDeleteImageModel(projectSlug || null);

  const { model: imageModelInformations, reFetch: reFetchImageModelInformations } =
    useModelInformations(projectSlug || null, currentImageModel || null, 'image', isComputing);
  const { model: quickModelInformations, reFetch: reFetchQuickModelInformations } =
    useModelInformations(projectSlug || null, currentQuickModelName || null, 'quick', isComputing);

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

  useEffect(() => {
    if (currentImageModel) reFetchImageModelInformations();
  }, [currentImageModel, isComputing, reFetchImageModelInformations, currentProject]);
  useEffect(() => {
    if (currentQuickModelName) reFetchQuickModelInformations();
  }, [currentQuickModelName, isComputing, reFetchQuickModelInformations, currentProject]);

  return (
    <div>
      <div>
        <span className="fw-semibold text-muted small">Quick Models</span>
        <ModelsPillDisplay
          modelNames={availableQuickModels
            .sort((a, b) => sortDatesAsStrings(a?.time, b?.time, true))
            .map((m) => m.name)}
          currentModelName={currentQuickModelName}
          setCurrentModelName={setCurrentQuickModelName}
          deleteModelFunction={deleteQuickModel}
        />
      </div>
      <div>
        <span className="fw-semibold text-muted small">Image Models</span>
        <ModelsPillDisplay
          modelNames={Object.values(availableImageModels)
            .sort((a, b) => sortDatesAsStrings(a?.time, b?.time, true))
            .map((m) => (m ? m.name : ''))}
          currentModelName={currentImageModel}
          setCurrentModelName={setCurrentImageModel}
          deleteModelFunction={deleteImageModel}
        />
      </div>

      {isComputing && (
        <DisplayTrainingProcesses
          projectSlug={projectSlug || null}
          processes={currentProject?.imagemodels?.training}
          displayStopButton={isComputing}
          stopKind="image"
        />
      )}

      <hr className="my-4" />

      {quickModelInformations && currentModel && (
        <>
          <ValidateButtons
            modelName={currentQuickModelName}
            kind="quick"
            id="compute-validate"
            style={{ margin: '8px 0px', color: 'white' }}
          />
          <DisplayScoresMenu
            scores={quickModelInformations.scores as unknown as Record<string, MLStatisticsModel>}
            modelName={currentQuickModelName || ''}
            skip={['internalvalid_scores']}
            projectSlug={projectSlug || null}
            exclude_labels={
              (quickModelInformations?.params?.exclude_labels as unknown as string[]) || []
            }
          />
        </>
      )}

      {imageModelInformations && currentModel && (
        <>
          <ValidateButtons
            modelName={currentImageModel}
            kind="image"
            id="compute-validate"
            style={{ margin: '8px 0px', color: 'white' }}
          />
          <DisplayScoresMenu
            scores={imageModelInformations.scores as unknown as Record<string, MLStatisticsModel>}
            modelName={currentImageModel || ''}
            skip={['internalvalid_scores']}
            projectSlug={projectSlug || null}
          />
        </>
      )}
    </div>
  );
};
