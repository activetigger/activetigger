import { FC, useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  useDeleteBertModel,
  useDeleteNerModel,
  useDeleteQuickModel,
  useModelInformations,
} from '../core/api';
import { useAppContext } from '../core/useAppContext';
import { sortDatesAsStrings } from '../core/utils';
import { MLStatisticsModel } from '../types';
import { DisplayNerScores } from './DisplayNerScores';
import { DisplayScoresMenu } from './DisplayScoresMenu';
import { DisplayTrainingProcesses } from './DisplayTrainingProcesses';
import { ModelsPillDisplay } from './ModelsPillDisplay';
import { ValidateButtons } from './ValidateButton';

type NerSplitBundle = {
  training_kind?: string;
  exact?: MLStatisticsModel;
  partial?: MLStatisticsModel;
} | null;

// User-facing labels for the score slots the backend exposes. Wording is
// kept aligned with DisplayScoresMenu (multiclass/multilabel selector) so
// the same vocabulary shows up in both places.
const NER_SPLIT_LABELS: Array<{ slot: string; label: string }> = [
  { slot: 'train_scores', label: 'Train (all)' },
  { slot: 'internalvalid_scores', label: 'Internal validation' },
  { slot: 'outofsample_scores', label: 'Train (eval)' },
  { slot: 'valid_scores', label: 'Validation set' },
  { slot: 'test_scores', label: 'Test set' },
];

/**
 * NER-specific score viewer: picks one split (train/valid/test/…) at a time
 * so the page doesn't stack a tall column of identical tables, then defers
 * to DisplayNerScores for the in-split flavor selector.
 */
const NerSplitSelector: FC<{
  modelScores: Record<string, NerSplitBundle>;
  modelName: string;
  projectSlug: string | null;
}> = ({ modelScores, modelName, projectSlug }) => {
  const availableSlots = NER_SPLIT_LABELS.filter(({ slot }) => Boolean(modelScores[slot]));
  const [slot, setSlot] = useState<string>(availableSlots[0]?.slot || 'train_scores');
  useEffect(() => {
    if (!modelScores[slot] && availableSlots.length > 0) setSlot(availableSlots[0].slot);
  }, [modelScores, slot, availableSlots]);

  if (availableSlots.length === 0) {
    return <div className="text-muted my-3">No NER metrics available yet.</div>;
  }
  const block = modelScores[slot] ?? null;

  return (
    <div className="my-3">
      <div className="horizontal">
        <label htmlFor="ner-statistics" style={{ marginRight: '10px' }}>
          Scores{' '}
        </label>
        <select
          id="ner-statistics"
          value={slot}
          onChange={(e) => setSlot(e.target.value)}
          style={{ maxWidth: '200px' }}
        >
          {availableSlots.map(({ slot: s, label }) => (
            <option key={s} value={s}>
              {label}
            </option>
          ))}
        </select>
      </div>
      <DisplayNerScores
        title={null}
        scores={block}
        modelName={modelName}
        projectSlug={projectSlug}
        dataset={slot.replace('_scores', '')}
      />
    </div>
  );
};

export const ModelEvaluation: FC = () => {
  const { projectName: projectSlug } = useParams();
  const {
    appContext: { currentScheme, currentProject, isComputing },
  } = useAppContext();

  const kindScheme =
    currentScheme && currentProject && currentProject.schemes.available[currentScheme]
      ? currentProject.schemes.available[currentScheme].kind || 'multiclass'
      : 'multiclass';
  const isNer = kindScheme === 'span';

  // quickmodel selector
  const availableQuickModels = useMemo(
    () => currentProject?.quickmodel.available[currentScheme || ''] || [],
    [currentProject?.quickmodel, currentScheme],
  );
  const [currentQuickModelName, setCurrentQuickModelName] = useState<string | null>(null);
  const { deleteQuickModel } = useDeleteQuickModel(projectSlug || null);

  // bertmodel selector — for span schemes this slot displays NER models instead
  const availableBertModels = useMemo(
    () =>
      isNer
        ? currentProject?.nermodels?.available?.[currentScheme || ''] || {}
        : currentProject?.languagemodels.available[currentScheme || ''] || {},
    [currentProject?.languagemodels, currentProject?.nermodels, currentScheme, isNer],
  );
  const [currentBertModel, setCurrentBertModel] = useState<string | null>(null);
  const { deleteBertModel } = useDeleteBertModel(projectSlug || null);
  const { deleteNerModel } = useDeleteNerModel(projectSlug || null);

  // get model information from api
  const { model: bertModelInformations, reFetch: reFetchBertModelInformations } =
    useModelInformations(
      projectSlug || null,
      currentBertModel || null,
      isNer ? 'ner' : 'bert',
      isComputing,
    );
  const { model: quickModelInformations, reFetch: reFetchQuickModelInformations } =
    useModelInformations(projectSlug || null, currentQuickModelName || null, 'quick', isComputing);

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

  // reFetch when model or isComputing change
  useEffect(() => {
    if (currentBertModel) reFetchBertModelInformations();
  }, [currentBertModel, isComputing, reFetchBertModelInformations, currentProject]);
  useEffect(() => {
    if (currentQuickModelName) reFetchQuickModelInformations();
  }, [currentQuickModelName, isComputing, reFetchQuickModelInformations, currentProject]);

  return (
    <div>
      {/* Display all the models */}
      {!isNer && (
        <div>
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
          />
        </div>
      )}
      <div>
        <span className="fw-semibold text-muted small">{isNer ? 'NER Models' : 'BERT Models'}</span>
        <ModelsPillDisplay
          modelNames={Object.values(availableBertModels)
            .sort((bertModelA, bertModelB) =>
              sortDatesAsStrings(bertModelA?.time, bertModelB?.time, true),
            )
            .map((model) => (model ? model.name : ''))}
          currentModelName={currentBertModel}
          setCurrentModelName={setCurrentBertModel}
          deleteModelFunction={isNer ? deleteNerModel : deleteBertModel}
        />
      </div>

      {isComputing && (
        <DisplayTrainingProcesses
          projectSlug={projectSlug || null}
          processes={currentProject?.languagemodels.training}
          displayStopButton={isComputing}
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
            batchInput={false}
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

      {bertModelInformations && currentModel && (
        <>
          <ValidateButtons
            modelName={currentBertModel}
            kind={isNer ? 'ner' : 'bert'}
            id="compute-validate"
            style={{ margin: '8px 0px', color: 'white' }}
          />
          {isNer ? (
            <NerSplitSelector
              modelScores={
                bertModelInformations.scores as unknown as Record<
                  string,
                  {
                    training_kind?: string;
                    exact?: MLStatisticsModel;
                    partial?: MLStatisticsModel;
                  } | null
                >
              }
              modelName={currentBertModel || ''}
              projectSlug={projectSlug || null}
            />
          ) : (
            <DisplayScoresMenu
              scores={bertModelInformations.scores as unknown as Record<string, MLStatisticsModel>}
              modelName={currentQuickModelName || ''}
              skip={['internalvalid_scores']}
              projectSlug={projectSlug || null}
            />
          )}
        </>
      )}
    </div>
  );
};
