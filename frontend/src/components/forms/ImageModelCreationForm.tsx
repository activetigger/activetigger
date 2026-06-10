import { Dispatch, FC, SetStateAction } from 'react';
import { Controller, SubmitHandler, useForm } from 'react-hook-form';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import Select from 'react-select';
import { Tooltip } from 'react-tooltip';
import { useTrainImageModel } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { getRandomName } from '../../core/utils';
import { newImageModel, ProjectStateModel } from '../../types';

interface ImageModelCreationFormProps {
  projectSlug: string | null;
  currentScheme: string | null;
  currentProject: ProjectStateModel | null;
  isComputing: boolean;
  setStatusDisplay?: Dispatch<SetStateAction<boolean>>;
}

type ImageBaseModel = {
  name: string;
  priority: number;
  comment?: string;
  parameters?: number;
  image_size?: number;
};

export const ImageModelCreationForm: FC<ImageModelCreationFormProps> = ({
  projectSlug,
  currentScheme,
  currentProject,
  isComputing,
  setStatusDisplay,
}) => {
  const { trainImageModel } = useTrainImageModel(projectSlug || null, currentScheme || null);
  const { notify } = useNotifications();

  // available ViT base models, sorted by priority
  const filteredModels = (
    (currentProject?.imagemodels?.options as unknown as ImageBaseModel[]) ?? []
  )
    .slice()
    .sort((a, b) => b.priority - a.priority);
  const availableBaseModels = filteredModels.map((e) => ({
    value: e.name,
    label: e.image_size ? `${e.name} (${e.image_size}px)` : e.name,
  }));

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

  // ViT-Large/384 effective-128 defaults (batch 16 x gradacc 8 with fp16).
  const createDefaultValues = (): newImageModel => ({
    name: getRandomName('imagemodel'),
    base: availableBaseModels?.[0]?.value ?? 'google/vit-large-patch16-384',
    class_balance: false,
    loss: 'cross_entropy',
    class_min_freq: 1,
    test_size: 0.2,
    parameters: {
      batchsize: 16,
      gradacc: 8,
      epochs: 10,
      lrate: 1e-4,
      wdecay: 1e-4,
      best: true,
      eval: 10,
      gpu: true,
      adapt: false,
    },
    exclude_labels: [],
    fp16: true,
  });

  const {
    handleSubmit: handleSubmitNewModel,
    register: registerNewModel,
    control,
  } = useForm<newImageModel>({ defaultValues: createDefaultValues() });

  const onSubmitNewModel: SubmitHandler<newImageModel> = async (data) => {
    if (availableLabels.length - (data.exclude_labels?.length ?? 0) < 2) {
      notify({
        type: 'error',
        message: 'You need at least 2 labels to start a training.',
      });
      return;
    }
    await trainImageModel(data);
    if (setStatusDisplay) setStatusDisplay(false);
  };

  return (
    <form onSubmit={handleSubmitNewModel(onSubmitNewModel)}>
      <label>Name for the model</label>
      <input type="text" {...registerNewModel('name')} placeholder="Name the model" />

      <label>
        Model base{' '}
        <a className="basemodel-image">
          <HiOutlineQuestionMarkCircle />
        </a>
        <Tooltip anchorSelect=".basemodel-image" place="top">
          The pre-trained backbone to fine-tune (ViT, ConvNeXt, EfficientNet, ...). ViT-Large@384 is
          the default; smaller variants need less GPU memory.
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

      <label>
        Epochs{' '}
        <a className="epochs-image">
          <HiOutlineQuestionMarkCircle />
        </a>
        <Tooltip anchorSelect=".epochs-image" place="top">
          Number of complete passes through the training set.
        </Tooltip>
      </label>
      <input type="number" {...registerNewModel('parameters.epochs')} min={0} />

      <label>
        Learning Rate{' '}
        <a className="learningrate-image">
          <HiOutlineQuestionMarkCircle />
        </a>
        <Tooltip anchorSelect=".learningrate-image" place="top">
          Step size for weight updates. Default 1e-4 for ViT fine-tuning.
        </Tooltip>
      </label>
      <input type="number" step="0.00001" min={0} {...registerNewModel('parameters.lrate')} />

      <label>
        Weight Decay{' '}
        <a className="weightdecay-image">
          <HiOutlineQuestionMarkCircle />
        </a>
        <Tooltip anchorSelect=".weightdecay-image" place="top">
          L2 regularization on the weights.
        </Tooltip>
      </label>
      <input type="number" step="0.0001" min={0} {...registerNewModel('parameters.wdecay')} />

      <label>
        <input type="checkbox" {...registerNewModel('parameters.gpu')} />
        Use GPU{' '}
        <a className="gpu-image">
          <HiOutlineQuestionMarkCircle />
        </a>
        <Tooltip anchorSelect=".gpu-image" place="top">
          ViT-Large/384 needs ≳12 GB of VRAM with batch 16 + fp16.
        </Tooltip>
      </label>

      <label>
        <input type="checkbox" {...registerNewModel('fp16')} />
        Mixed precision (fp16){' '}
        <a className="fp16-image">
          <HiOutlineQuestionMarkCircle />
        </a>
        <Tooltip anchorSelect=".fp16-image" place="top">
          Halves activation memory on CUDA; ignored on CPU.
        </Tooltip>
      </label>

      <details className="custom-details">
        <summary>Advanced parameters for the model</summary>

        <label>
          Batch Size{' '}
          <a className="batchsize-image">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".batchsize-image" place="top">
            Per-device images per step. Reduce if you OOM on ViT-Large/384.
          </Tooltip>
        </label>
        <input type="number" {...registerNewModel('parameters.batchsize')} min={1} />

        <label>
          Gradient Accumulation{' '}
          <a className="gradientacc-image">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".gradientacc-image" place="top">
            Effective batch = batch_size × gradient_accumulation. Default 16 × 8 = 128.
          </Tooltip>
        </label>
        <input type="number" step="1" {...registerNewModel('parameters.gradacc')} min={1} />

        <label>
          Eval steps{' '}
          <a className="evalstep-image">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".evalstep-image" place="top">
            How often validation runs during training.
          </Tooltip>
        </label>
        <input type="number" {...registerNewModel('parameters.eval')} />

        <label>
          Train-eval split size{' '}
          <a className="test_size-image">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".test_size-image" place="top">
            Fraction of training data held out for internal validation.
          </Tooltip>
        </label>
        <input type="number" step="0.1" min="0" max="0.9" {...registerNewModel('test_size')} />

        <label>
          Label threshold{' '}
          <a className="class_min_freq-image">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".class_min_freq-image" place="top">
            Drop classes with fewer than this many annotated images.
          </Tooltip>
        </label>
        <input type="number" step="1" {...registerNewModel('class_min_freq')} />

        <label>
          <input type="checkbox" {...registerNewModel('class_balance')} />
          Balance labels{' '}
          <a className="class_balance-image">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".class_balance-image" place="top">
            Downsample majority classes to the size of the smallest.
          </Tooltip>
        </label>

        <label className="horizontal">
          Loss{' '}
          <a className="loss-image">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".loss-image" place="top">
            Choose a loss function.
          </Tooltip>
          <select {...registerNewModel('loss')} className="mx-2">
            <option value="cross_entropy">Cross Entropy</option>
            <option value="weighted_cross_entropy">Weighted Cross Entropy</option>
          </select>
        </label>

        <label>
          <input type="checkbox" {...registerNewModel('parameters.best')} />
          Keep the best model{' '}
          <a className="best-image">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".best-image" place="top">
            Keep the checkpoint with the lowest validation loss.
          </Tooltip>
        </label>
      </details>

      <details className="custom-details">
        <summary>Advanced parameters for the data</summary>
        <label>
          Labels to ignore{' '}
          <a className="ignore-image">
            <HiOutlineQuestionMarkCircle />
          </a>
          <Tooltip anchorSelect=".ignore-image" place="top">
            Images with these labels are skipped during training.
          </Tooltip>
        </label>
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

      <button key="start" className="btn-submit" disabled={isComputing}>
        Train the model
      </button>
    </form>
  );
};
