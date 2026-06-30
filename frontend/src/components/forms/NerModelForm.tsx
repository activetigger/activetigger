import { Dispatch, FC, SetStateAction } from 'react';
import { Controller, SubmitHandler, useForm } from 'react-hook-form';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import Select from 'react-select';
import { Tooltip } from 'react-tooltip';
import { useTrainNerModel } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { getRandomName } from '../../core/utils';
import { newNerModel, ProjectStateModel } from '../../types';

interface NerModelFormProps {
  projectSlug: string | null;
  currentScheme: string | null;
  currentProject: ProjectStateModel | null;
  isComputing: boolean;
  setStatusDisplay?: Dispatch<SetStateAction<boolean>>;
}

type NerBaseModel = {
  name: string;
  priority: number;
  comment: string;
  language: string;
};

/**
 * Train form for NER (token-classification) models on span schemes.
 * Drops classification-specific fields (loss, dichotomize, class_balance,
 * class_min_freq, exclude_labels) — BIO tagging on a span scheme has no
 * "per-class" knobs to balance or ignore.
 */
export const NerModelForm: FC<NerModelFormProps> = ({
  projectSlug,
  currentScheme,
  currentProject,
  isComputing,
  setStatusDisplay,
}) => {
  const { trainNerModel } = useTrainNerModel(projectSlug || null, currentScheme || null);
  const { notify } = useNotifications();

  const filteredModels = (
    ((currentProject?.nermodels?.options ??
      currentProject?.languagemodels.options) as unknown as NerBaseModel[]) ?? []
  )
    .sort((a, b) => b.priority - a.priority)
    .sort((a, b) => {
      const aMatch = a.language === currentProject?.params.language ? -1 : 1;
      const bMatch = b.language === currentProject?.params.language ? -1 : 1;
      return aMatch - bMatch;
    });
  const availableBaseModels = filteredModels.map((e) => ({
    value: e.name as string,
    label: `[${e.language as string}] ${e.name as string}`,
  }));

  const availableLabels =
    currentScheme &&
    currentProject &&
    currentProject.schemes.available &&
    currentProject.schemes.available[currentScheme]
      ? currentProject.schemes.available[currentScheme].labels
      : [];

  const defaultValues = (): newNerModel => ({
    name: getRandomName('nermodel'),
    base: availableBaseModels?.[0]?.value ?? '',
    test_size: 0.2,
    max_length: 512,
    parameters: {
      batchsize: 16,
      gradacc: 1.0,
      epochs: 5,
      lrate: 5e-5,
      wdecay: 0.01,
      best: true,
      eval: 9,
      gpu: true,
      adapt: false,
    },
  });

  const { handleSubmit, register, control } = useForm<newNerModel>({
    defaultValues: defaultValues(),
  });

  const onSubmit: SubmitHandler<newNerModel> = async (data) => {
    if (Object.keys(availableLabels).length < 1) {
      notify({ type: 'error', message: 'Define at least one tag in the scheme before training.' });
      return;
    }
    await trainNerModel(data);
    if (setStatusDisplay) setStatusDisplay(false);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div className="alert alert-warning" role="alert">
        Experimental: NER fine-tuning uses BIO tagging on word boundaries. Annotation spans that
        don't align with word boundaries may shift by a character or two.
      </div>

      <label>Name for the model</label>
      <input type="text" {...register('name')} placeholder="Name the model" />

      <label>
        Model base{' '}
        <a className="basemodel-ner">
          <HiOutlineQuestionMarkCircle />
        </a>
        <Tooltip anchorSelect=".basemodel-ner" place="top">
          Pre-trained encoder fine-tuned for token classification.
        </Tooltip>
      </label>
      <Controller
        name="base"
        control={control}
        defaultValue={availableBaseModels?.[0]?.value}
        render={({ field }) => (
          <Select
            {...field}
            options={availableBaseModels}
            classNamePrefix="react-select"
            value={availableBaseModels.find((o) => o.value === field.value)}
            onChange={(opt) => field.onChange(opt?.value)}
          />
        )}
      />

      <label>
        Context window size (in tokens){' '}
        <a className="max_length-ner">
          <HiOutlineQuestionMarkCircle />
        </a>
        <Tooltip anchorSelect=".max_length-ner" place="top">
          Documents longer than this are truncated; spans past the cutoff are dropped.
        </Tooltip>
      </label>
      <input type="number" step="1" {...register('max_length')} />

      <label>Epochs</label>
      <input type="number" {...register('parameters.epochs')} min={1} />

      <label>Learning rate</label>
      <input type="number" step="0.00001" min={0} {...register('parameters.lrate')} />

      <label>Weight decay</label>
      <input type="number" step="0.001" min={0} {...register('parameters.wdecay')} />

      <label>
        <input type="checkbox" {...register('parameters.gpu')} />
        Use GPU
      </label>

      <details className="custom-details">
        <summary>Advanced parameters</summary>

        <label>Batch size</label>
        <input type="number" {...register('parameters.batchsize')} min={1} />

        <label>Gradient accumulation</label>
        <input type="number" step="1" {...register('parameters.gradacc')} min={1} />

        <label>Eval frequency</label>
        <input type="number" {...register('parameters.eval')} />

        <label>Train-eval split size</label>
        <input type="number" step="0.1" min="0" max="0.9" {...register('test_size')} />

        <label>
          <input type="checkbox" {...register('parameters.best')} />
          Keep the best model (lowest eval loss)
        </label>
      </details>

      <button key="start" className="btn-submit" disabled={isComputing}>
        Train the NER model
      </button>
    </form>
  );
};
