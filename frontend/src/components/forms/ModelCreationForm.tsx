import { Dispatch, FC, SetStateAction, useEffect, useState } from 'react';
import { Controller, SubmitHandler, useForm } from 'react-hook-form';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import Select from 'react-select';
import { Tooltip } from 'react-tooltip';
import { useTrainBertModel } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { getRandomName } from '../../core/utils';
import { newBertModel, ProjectStateModel } from '../../types';
interface ModelCreationFormProps {
  projectSlug: string | null;
  currentScheme: string | null;
  currentProject: ProjectStateModel | null;
  isComputing: boolean;
  setStatusDisplay?: Dispatch<SetStateAction<boolean>>;
}

type BertModel = {
  name: string;
  priority: number;
  comment: string;
  language: string;
};

export const ModelCreationForm: FC<ModelCreationFormProps> = ({
  projectSlug,
  currentScheme,
  currentProject,
  isComputing,
  setStatusDisplay,
}) => {
  // form to train a model
  const { trainBertModel } = useTrainBertModel(projectSlug || null, currentScheme || null);
  const [disableMaxLengthInput, setDisableMaxLengthInput] = useState<boolean>(true);
  const [disableLabelsToIgnore, setDisableLabelsToIgnore] = useState<boolean>(false);
  // const { gpu } = useGetServer(currentProject || null);
  const { notify } = useNotifications();
  // available base models suited for the project : sorted by language + priority
  const filteredModels = ((currentProject?.languagemodels.options as unknown as BertModel[]) ?? [])
    .sort((a, b) => b.priority - a.priority)
    .sort((a, b) => {
      const aHasFr = a.language === currentProject?.params.language ? -1 : 1;
      const bHasFr = b.language === currentProject?.params.language ? -1 : 1;
      return aHasFr - bHasFr;
    });
  const availableBaseModels = filteredModels.map((e) => ({
    value: e.name as string,
    label: `[${e.language as string}] ${e.name as string}`,
  }));
  // available labels from context
  const availableLabels =
    currentScheme &&
    currentProject &&
    currentProject.schemes.available &&
    currentProject.schemes.available[currentScheme]
      ? currentProject.schemes.available[currentScheme].labels
      : [];
  const existingLabels = Object.entries(availableLabels).map(([key, value]) => ({
    value: key,
    label: value,
  }));

  const kindScheme =
    currentScheme && currentProject && currentProject.schemes.available[currentScheme]
      ? currentProject.schemes.available[currentScheme].kind
      : 'multiclass';

  const createDefaultValues = () => ({
    name: getRandomName('bertmodel'),
    class_balance: false,
    loss: 'cross_entropy',
    class_min_freq: 1,
    test_size: 0.2,
    max_length: 512,
    auto_max_length: false,
    parameters: {
      batchsize: 16,
      gradacc: 1.0,
      epochs: 3,
      lrate: 3e-5,
      wdecay: 0.01,
      best: true,
      eval: 9,
      gpu: true,
      adapt: false,
    },
    exclude_labels: [],
    dichotomize: 'No dichotomization',
  });

  const {
    handleSubmit: handleSubmitNewModel,
    register: registerNewModel,
    watch,
    control,
  } = useForm<newBertModel>({
    defaultValues: createDefaultValues(),
  });

  const autoMaxLengthValue = watch('auto_max_length');
  useEffect(() => {
    setDisableMaxLengthInput(autoMaxLengthValue);
  }, [autoMaxLengthValue]);
  const dichotomization = watch('dichotomize');
  useEffect(() => {
    if (dichotomization === 'No dichotomization') setDisableLabelsToIgnore(false);
    else setDisableLabelsToIgnore(true);
  }, [dichotomization]);

  const onSubmitNewModel: SubmitHandler<newBertModel> = async (data) => {
    // setActiveKey('models');
    // Retrieve existing labels and prevent training if only one label
    if (availableLabels.length - data.exclude_labels?.length < 2) {
      notify({
        type: 'error',
        message: 'You need at least 2 labels to start a training.',
      });
      return;
    } else {
      console.log(data);
      await trainBertModel(data);
      if (setStatusDisplay) setStatusDisplay(false);
    }
  };
  return (
    <form onSubmit={handleSubmitNewModel(onSubmitNewModel)}>
      <label htmlFor="new-model-type"></label>
      <label>Name for the model</label>
      <input type="text" {...registerNewModel('name')} placeholder="Name the model" />
      <label>
        Model base{' '}
        <a className="basemodel">
          <HiOutlineQuestionMarkCircle />
        </a>
        <Tooltip anchorSelect=".basemodel" place="top">
          The pre-trained model to be used for fine-tuning.
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
            value={availableBaseModels.find((option) => option.value === field.value)}
            onChange={(selectedOption) => field.onChange(selectedOption?.value)}
          />
        )}
      />

      <div style={{ display: 'flex', flexWrap: 'wrap', marginTop: '10px' }}>
        <label style={{ flex: '1 1 auto' }}>
          <input type="checkbox" {...registerNewModel('auto_max_length')} />
          Auto adjust Max context window{' '}
          <a className="optimum_max_length">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".optimum_max_length" place="top">
            Automatically sets the the context window size to the maximum number of tokens in one
            <br />
            element of your corpus.
          </Tooltip>
        </label>
        <div style={{ display: 'flex', flex: '3 1 auto' }}>
          <label style={{ flex: '3 0 300px' }}>
            Context window size (in tokens){' '}
            <a className="max_length">
              <HiOutlineQuestionMarkCircle />
            </a>
            <Tooltip anchorSelect=".max_length" place="top">
              Number of tokens before truncation (depends on the model)
            </Tooltip>
          </label>
          <input
            type="number"
            step="1"
            {...registerNewModel('max_length')}
            disabled={disableMaxLengthInput}
            style={{ flex: '1 1 400px' }}
          />
        </div>
      </div>

      <label>
        Epochs{' '}
        <a className="epochs">
          <HiOutlineQuestionMarkCircle />
        </a>
        <Tooltip anchorSelect=".epochs" place="top">
          number of complete pass through the entire training dataset
        </Tooltip>
      </label>
      <input type="number" {...registerNewModel('parameters.epochs')} min={0} />
      <label>
        Learning Rate{' '}
        <a className="learningrate">
          <HiOutlineQuestionMarkCircle />
        </a>
        <Tooltip anchorSelect=".learningrate" place="top">
          step size at which the model updates its weights during training (use a factor 3 to change
          <br />
          it)
        </Tooltip>
      </label>
      <input type="number" step="0.00001" min={0} {...registerNewModel('parameters.lrate')} />
      <label>
        Weight Decay{' '}
        <a className="weightdecay">
          <HiOutlineQuestionMarkCircle />
        </a>
        <Tooltip anchorSelect=".weightdecay" place="top">
          regularization technique that reduces model weights over time to prevent overfitting
        </Tooltip>
      </label>
      <input type="number" step="0.001" min={0} {...registerNewModel('parameters.wdecay')} />

      <label>
        <input type="checkbox" {...registerNewModel('parameters.gpu')} />
        Use GPU
        <a className="gpu">
          <HiOutlineQuestionMarkCircle />
        </a>
        <Tooltip anchorSelect=".gpu" place="top">
          Compute the training on GPU.
        </Tooltip>
      </label>

      <details className="custom-details">
        <summary>Advanced parameters for the model</summary>

        <label>
          Batch Size{' '}
          <a className="batchsize">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".batchsize" place="top">
            How many samples are processed simultaneously. With small GPU, keep it around 4.
          </Tooltip>
        </label>
        <input type="number" {...registerNewModel('parameters.batchsize')} min={1} />

        <label>
          Gradient Accumulation{' '}
          <a className="gradientacc">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".gradientacc" place="top">
            summing gradients over multiple steps before updating the model weights
          </Tooltip>
        </label>
        <input type="number" step="1" {...registerNewModel('parameters.gradacc')} min={1} />

        <label>
          Eval{' '}
          <a className="evalstep">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".evalstep" place="top">
            how often (in terms of training steps) the evaluation of the model on the validation
            dataset is performed during training
          </Tooltip>
        </label>
        <input type="number" {...registerNewModel('parameters.eval')} />

        <label>
          Train-eval split size{' '}
          <a className="test_size">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".test_size" place="top">
            Eval size for the dev test to compute metrics.
          </Tooltip>
        </label>
        <input type="number" step="0.1" min="0" max="0.9" {...registerNewModel('test_size')} />

        <label>
          Class threshold{' '}
          <a className="class_min_freq">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".class_min_freq" place="top">
            Drop classses with less than this number of elements
          </Tooltip>
        </label>
        <input type="number" step="1" {...registerNewModel('class_min_freq')} />

        <label>
          <input type="checkbox" {...registerNewModel('class_balance')} />
          Balance classes
          <a className="class_balance">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".class_balance" place="top">
            Downsize classes to the lowest one.
          </Tooltip>
        </label>

        <label className="horizontal">
          Loss
          <a className="loss">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".loss" place="top">
            Use a specific loss function
          </Tooltip>
          <select {...registerNewModel('loss')} className="mx-2">
            <option value="cross_entropy">Cross Entropy</option>
            <option value="weighted_cross_entropy">Weighted Cross Entropy</option>
          </select>{' '}
        </label>

        <label>
          <input type="checkbox" {...registerNewModel('parameters.best')} />
          Keep the best model
          <a className="best">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".best" place="top">
            Keep the model with the lowest validation loss.
          </Tooltip>
        </label>
      </details>
      <details className="custom-details">
        <summary>Advanced parameters for the data</summary>

        {kindScheme == 'multilabel' && (
          <div>
            <label htmlFor="dichotomize">Dichotomize on a specific label (yes/no) </label>
            <select id="dichotomize" {...registerNewModel('dichotomize')}>
              <option key="none">No dichotomization</option>
              {Object.values(availableLabels).map((e) => (
                <option key={e}>{e}</option>
              ))}{' '}
            </select>
          </div>
        )}

        <label>
          Labels to ignore{' '}
          <a className="ignore">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".ignore" place="top">
            Elements with those labels will be ignored during training
          </Tooltip>
        </label>

        <Controller
          name="exclude_labels"
          control={control}
          disabled={disableLabelsToIgnore}
          render={({ field: { onChange } }) => (
            <Select
              options={existingLabels}
              isMulti
              isDisabled={disableLabelsToIgnore}
              onChange={(selectedOptions) => {
                onChange(selectedOptions ? selectedOptions.map((option) => option.label) : []);
              }}
            />
          )}
        />
      </details>
      <button key="start" className="btn-submit" disabled={isComputing}>
        Train the model
      </button>
    </form>
  );
};
