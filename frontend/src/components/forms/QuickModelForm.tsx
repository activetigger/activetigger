import { FC, useState } from 'react';
import { Controller, SubmitHandler, useForm } from 'react-hook-form';
import Select from 'react-select';
import { useTrainQuickModel } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { getRandomName } from '../../core/utils';
import { QuickModelInModel } from '../../types';
import { ButtonNewFeature } from '../ButtonNewFeature';

interface QuickModelFormProps {
  projectSlug: string;
  currentScheme: string;
  kindScheme: string;
  features: { value: string; label: string }[];
  availableLabels: string[];
  baseQuickModels: Record<string, Record<string, number>>;
  setDisplayNewModel: (display: boolean) => void;
}

// build default features selected
type Feature = {
  label: string;
  value: string;
};

export const QuickModelForm: FC<QuickModelFormProps> = ({
  projectSlug,
  currentScheme,
  kindScheme,
  features,
  availableLabels,
  baseQuickModels,
  setDisplayNewModel,
}) => {
  const { notify } = useNotifications();

  // hooks to update
  const { trainQuickModel } = useTrainQuickModel(projectSlug, currentScheme);

  const filterFeatures = (features: Feature[]) => {
    const filtered = features.filter((e) =>
      /sentence-embeddings|embeddings|fasttext/i.test(e.label),
    );
    const predictFeature = features.find((e) => /predict/i.test(e.label)); // Trouve le premier "predict"
    const embeddingsFeature = features.find((e) => /sentence-embeddings|embeddings/i.test(e.label)); // Trouve le premier "sentence-embeddings"

    if (embeddingsFeature) {
      filtered.push(embeddingsFeature);
    } else if (predictFeature) {
      filtered.push(predictFeature);
    }

    return filtered;
  };
  const existingLabels = Object.entries(availableLabels).map(([key, value]) => ({
    value: key,
    label: value,
  }));
  const predictions = filterFeatures(features);
  const defaultFeatures = predictions.length > 0 ? [predictions[predictions.length - 1]] : [];

  const createDefaultValues = () => ({
    name: getRandomName('QuickModel'),
    model: 'logistic-l2',
    scheme: currentScheme || undefined,
    params: {
      costLogL1: 1,
      costLogL2: 1,
      n_neighbors: 3,
      alpha: 1,
      n_estimators: 500,
      max_features: null,
    },
    balance_classes: false,
    cv10: false,
    test_size: 0.2,
    dichotomize: kindScheme == 'multilabel' ? availableLabels[0] : undefined,
    features: defaultFeatures.map((e) => e.value),
  });

  // create form
  const { register, handleSubmit, control, watch } = useForm<QuickModelInModel>({
    defaultValues: createDefaultValues(),
  });

  // state for the model selected to modify parameters
  const selectedModel = watch('model');

  // action when form validated
  const onSubmit: SubmitHandler<QuickModelInModel> = async (formData) => {
    const watchedFeatures = watch('features');
    if (watchedFeatures.length == 0) {
      notify({ type: 'error', message: 'Please select at least one feature' });
      return;
    }
    if (availableLabels.length - formData.exclude_labels?.length < 2) {
      notify({
        type: 'error',
        message: 'You need at least 2 labels to start training.',
      });
      return;
    }
    await trainQuickModel(formData);
    setDisplayNewModel(false);
  };

  const [formSelectedFeatures, setFormSelectedFeatures] = useState<string[]>(
    defaultFeatures.map((e) => e.value),
  );

  const selectedFeaturesContainsBERTFeatures = () => {
    return formSelectedFeatures
      .map((feature) => feature?.slice(0, 8) === 'predict_')
      .includes(true);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <label htmlFor="name">Model name</label>
      <input type="text" id="name" placeholder="Model name" {...register('name')} />

      <label htmlFor="features">Features used to predict (X)</label>
      <ButtonNewFeature projectSlug={projectSlug} />
      <Controller
        name="features"
        control={control}
        render={({ field: { onChange, value } }) => (
          <>
            {' '}
            <Select
              options={features}
              isMulti
              value={features.filter((option) => value.includes(option.value))}
              onChange={(selectedOptions) => {
                onChange(selectedOptions ? selectedOptions.map((option) => option.value) : []);
                setFormSelectedFeatures(
                  selectedOptions ? selectedOptions.map((option) => option.value) : [],
                );
              }}
            />
          </>
        )}
      />
      {selectedFeaturesContainsBERTFeatures() && (
        <a className="explanations">
          ⚠️ Warning: using BERT predictions as features results in strongly upward-biased quality
          metrics on the train set.
        </a>
      )}

      {kindScheme == 'multilabel' && (
        <>
          <label htmlFor="dichotomize">Dichotomize on the label</label>
          <select id="dichotomize" {...register('dichotomize')}>
            {Object.values(availableLabels).map((e) => (
              <option key={e}>{e}</option>
            ))}{' '}
          </select>
        </>
      )}

      <details>
        <summary>Advanced parameters</summary>

        <label htmlFor="model">Select a model</label>
        <select id="model" {...register('model')}>
          {Object.keys(baseQuickModels).map((e) => (
            <option key={e}>{e}</option>
          ))}{' '}
        </select>
        {
          //generate_config(selectedQuickModel)
          (selectedModel == 'logistic-l2' && (
            <div key="logistic-l2">
              <label htmlFor="costLogL2">Cost</label>
              <input
                type="number"
                step="1"
                id="logistic-l2"
                {...register('params.costLogL2', { valueAsNumber: true })}
              ></input>
              <label htmlFor="balance_classes">
                <input type="checkbox" id="balance_classes" {...register('balance_classes')} />
                Automatically balance classes
              </label>
            </div>
          )) ||
            (selectedModel == 'knn' && (
              <div key="knn">
                <label htmlFor="n_neighbors">Number of neighbors</label>
                <input
                  type="number"
                  step="1"
                  id="n_neighbors"
                  {...register('params.n_neighbors', { valueAsNumber: true })}
                ></input>
              </div>
            )) ||
            (selectedModel == 'logistic-l1' && (
              <div key="logistic-l1">
                <label htmlFor="costLogL1">Cost</label>
                <input
                  type="number"
                  step="1"
                  id="logistic-l1"
                  {...register('params.costLogL1', { valueAsNumber: true })}
                ></input>
                <label htmlFor="balance_classes">
                  <input type="checkbox" id="balance_classes" {...register('balance_classes')} />
                  Automatically balance classes
                </label>
              </div>
            )) ||
            (selectedModel == 'multi_naivebayes' && (
              <div key="multi_naivebayes">
                <label htmlFor="alpha">Alpha</label>
                <input
                  type="number"
                  id="alpha"
                  {...register('params.alpha', { valueAsNumber: true })}
                ></input>
                <label htmlFor="fit_prior">
                  <input type="checkbox" id="fit_prior" {...register('params.fit_prior')} checked />
                  Fit prior
                </label>
              </div>
            )) ||
            (selectedModel == 'randomforest' && (
              <div key="randomforest">
                <label htmlFor="n_estimators">Number of estimators</label>
                <input
                  type="number"
                  step="1"
                  id="n_estimators"
                  {...register('params.n_estimators', { valueAsNumber: true })}
                ></input>
                <label htmlFor="max_features">Max features</label>
                <input
                  type="number"
                  step="1"
                  id="max_features"
                  {...register('params.max_features', { valueAsNumber: true })}
                ></input>
                <label htmlFor="balance_classes">
                  <input type="checkbox" id="balance_classes" {...register('balance_classes')} />
                  Automatically balance classes
                </label>
              </div>
            ))
        }

        <label htmlFor="test_size">Train-eval split size</label>
        <input
          type="number"
          step="0.1"
          min="0"
          max="0.9"
          id="test_size"
          {...register('test_size', { valueAsNumber: true })}
        />

        <label htmlFor="cv10">
          <input type="checkbox" id="cv10" {...register('cv10')} />
          10-fold cross validation
        </label>
      </details>
      <details>
        <summary>Advanced parameters for the data</summary>
        <label>Labels to ignore</label>
        <Controller
          name="exclude_labels"
          control={control}
          render={({ field: { onChange } }) => (
            <Select
              options={existingLabels}
              isMulti
              onChange={(selectedOptions) => {
                onChange(selectedOptions ? selectedOptions.map((option) => option.label) : []);
              }}
            />
          )}
        />
      </details>

      <button className="btn-submit">Train quick model</button>
    </form>
  );
};
