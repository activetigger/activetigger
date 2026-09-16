import { FC, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useModelInformations } from '../core/api';
import { useAppContext } from '../core/useAppContext';
import { MLStatisticsModel } from '../types';
import { DisplayNerScores } from './DisplayNerScores';
import { DisplayScoresMenu } from './DisplayScoresMenu';
import { DisplayTrainingProcesses } from './DisplayTrainingProcesses';
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

/**
 * Evaluation view of the model selected in the pills above the tabs.
 * For span schemes the "bert" slot holds a NER model.
 */
export const ModelEvaluation: FC<{
  selectedQuickModel: string | null;
  selectedBertModel: string | null;
}> = ({ selectedQuickModel, selectedBertModel }) => {
  const { projectName: projectSlug } = useParams();
  const {
    appContext: { currentScheme, currentProject, isComputing },
  } = useAppContext();

  const kindScheme =
    currentScheme && currentProject && currentProject.schemes.available[currentScheme]
      ? currentProject.schemes.available[currentScheme].kind || 'multiclass'
      : 'multiclass';
  const isNer = kindScheme === 'span';
  const currentQuickModelName = selectedQuickModel;
  const currentBertModel = selectedBertModel;

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

  // reFetch when model or isComputing change
  useEffect(() => {
    if (currentBertModel) reFetchBertModelInformations();
  }, [currentBertModel, isComputing, reFetchBertModelInformations, currentProject]);
  useEffect(() => {
    if (currentQuickModelName) reFetchQuickModelInformations();
  }, [currentQuickModelName, isComputing, reFetchQuickModelInformations, currentProject]);

  return (
    <div>
      {isComputing && (
        <DisplayTrainingProcesses
          projectSlug={projectSlug || null}
          processes={currentProject?.languagemodels.training}
          displayStopButton={isComputing}
        />
      )}

      {!currentQuickModelName && !currentBertModel && (
        <div className="text-muted my-3">Select a model above to see its evaluation scores</div>
      )}

      {quickModelInformations && currentQuickModelName && (
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

      {bertModelInformations && currentBertModel && (
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
