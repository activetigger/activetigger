import { Dispatch, FC, SetStateAction } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { Tooltip } from 'react-tooltip';
import { useComputeBertopic, useStopProcesses } from '../../core/api';
import { useAppContext } from '../../core/useAppContext';
import { getRandomName } from '../../core/utils';
import { ComputeBertopicModel } from '../../types';

interface BertopicCreationFormProps {
  projectSlug: string | null;
  bindableFeatures: string[];
  isComputing?: boolean;
  setStatusDisplay?: Dispatch<SetStateAction<boolean>>;
}

export const BertopicForm: FC<BertopicCreationFormProps> = ({
  projectSlug,
  bindableFeatures,
  isComputing = false,
  setStatusDisplay,
}) => {
  const { computeBertopic } = useComputeBertopic(projectSlug);
  const { stopProcesses } = useStopProcesses(projectSlug);
  const {
    appContext: { currentScheme },
  } = useAppContext();

  const { handleSubmit: handleSubmitNewModel, register } = useForm<ComputeBertopicModel>({
    defaultValues: {
      name: getRandomName('BERTopic'),
      outlier_reduction: true,
      hdbscan_min_cluster_size: 15,
      umap_n_neighbors: 30,
      umap_n_components: 5,
      existing_feature: bindableFeatures[0] ?? '',
      filter_text_length: 50,
      input_datasets: 'train',
      scheme: currentScheme,
    },
  });

  const onSubmitNewModel: SubmitHandler<ComputeBertopicModel> = async (data) => {
    await computeBertopic(data);
    if (setStatusDisplay) setStatusDisplay(false);
  };

  if (bindableFeatures.length === 0) {
    return (
      <div className="alert alert-warning">
        BERTopic reuses embeddings from a project feature. Compute a sentence-embeddings feature
        first in the <b>Features</b> page, then come back here to run BERTopic on it.
      </div>
    );
  }

  return (
    <div>
      <form onSubmit={handleSubmitNewModel(onSubmitNewModel)}>
        <label htmlFor="name">Name</label>
        <input id="name" type="text" {...register('name')} />
        <label htmlFor="existing_feature">
          Embeddings feature
          <a className="existing_feature">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".existing_feature" place="top">
            BERTopic reuses embeddings already computed for this project.
            <br />
            Pick a sentence-embeddings feature; to use a different embedding
            <br />
            model, add it from the Features page first.
          </Tooltip>
        </label>
        <select id="existing_feature" {...register('existing_feature', { required: true })}>
          {bindableFeatures.map((feature) => (
            <option key={feature} value={feature}>
              {feature}
            </option>
          ))}
        </select>
        <label htmlFor="umap_n_neighbors">
          Number of neighbors (dimension reduction parameter)
          <a className="umap_n_neighbors">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".umap_n_neighbors" place="top">
            (UMAP) Choose small values to focus on local structures (ie specific topics) and large
            <br />
            values to focus on broader structures (ie broad topics)
            <br />
            <i>This value depends on how many elements you have in your corpus</i>
          </Tooltip>
        </label>
        <input
          className="form-control"
          id="umap_n_neighbors"
          type="number"
          {...register('umap_n_neighbors')}
        />
        <label htmlFor="min_topic_size">
          Min topic size (clustering parameter)
          <a className="min_topic_size">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".min_topic_size" place="top">
            (HDBSCAN) Minimum number of elements in a group to be considered a cluster, otherwise,
            <br />
            it's considered as noise.
            <br />
            Small values will generate many small topics whereas larger values will generate few
            <br />
            topics and a lot of noise.
            <br />
            <i>This value depends on how many elements you have in your corpus</i>
          </Tooltip>
        </label>
        <input id="minTopicSize" type="number" {...register('hdbscan_min_cluster_size')} />
        <details>
          <summary>Advanced parameters</summary>
          <div className="explanations">Using UMAP (reduction) and HDBSCAN (clustering)</div>
          <div>
            <input id="outlier_reduction" type="checkbox" {...register('outlier_reduction')} />
            <label htmlFor="outlier_reduction">Outlier reduction</label>
          </div>
          <label htmlFor="input_datasets">Input dataset</label>
          <select {...register('input_datasets')}>
            {/* TODO: Add the number of element in each option */}
            <option key="train" value="train">
              Train
            </option>
            <option key="all_sets" value="all_sets">
              All sets (train + test + valid)
            </option>
          </select>
          <label htmlFor="filter_text_length">Filter out texts of length lower than</label>
          <input id="filter_text_length" type="number" {...register('filter_text_length')} />
          <label htmlFor="umap_n_components">
            Number of components (dimension reduction parameter)
            <a className="umap_n_components">
              <HiOutlineQuestionMarkCircle />
            </a>
            <Tooltip anchorSelect=".umap_n_components" place="top">
              The number of dimensions to reduce the embedding space to.
              <br />
              There is not a quick way of tuning it. The lower the value the "flatter", ie the
              <br />
              embedding will lose information, however increasing this value does not guarantee
              <br />
              better results. Try changing the embedding model first.
            </Tooltip>
          </label>
          <input id="umap_n_components" type="number" {...register('umap_n_components')} />
        </details>

        {!isComputing && <button className="btn-submit">Compute Bertopic</button>}
      </form>
      {isComputing && (
        // TODO: AXEL refactor this button
        <button className="btn btn-primary w-100" onClick={() => stopProcesses('all')}>
          Stop computation
        </button>
      )}
    </div>
  );
};
