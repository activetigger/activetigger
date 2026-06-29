import { ChangeEvent, FC, useEffect, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { fetchOllamaModels, fetchOpenAICompatibleModels, useGetGenModels } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { GenerationModelApi, GenModel, SupportedAPI } from '../../types';

type FormValues = { model: string; name?: string; endpoint?: string; credentials?: string };

export const GenModelSetupForm: FC<{
  add: (model: Omit<GenModel & { api: SupportedAPI }, 'id'>) => void;
  cancel: () => void;
}> = ({ add }) => {
  const { notify } = useNotifications();
  const [availableAPIs, setAvailableAPIs] = useState<GenerationModelApi[]>([]);
  const [selectedAPI, setSelectedAPI] = useState<GenerationModelApi>(availableAPIs[0]);
  const [modelName, setModelName] = useState<string>('');
  const [ollamaEndpoint, setOllamaEndpoint] = useState<string>('');
  const [ollamaModels, setOllamaModels] = useState<Array<{ slug: string; name: string }>>([]);
  const [ollamaLoading, setOllamaLoading] = useState(false);
  const [openaiEndpoint, setOpenaiEndpoint] = useState<string>('');
  const [openaiCredentials, setOpenaiCredentials] = useState<string>('');
  const [openaiModels, setOpenaiModels] = useState<Array<{ slug: string; name: string }>>([]);
  const [openaiLoading, setOpenaiLoading] = useState(false);
  const { models } = useGetGenModels();
  const { register, handleSubmit, setValue } = useForm<FormValues>();
  useEffect(() => {
    const fetchModels = async () => {
      setAvailableAPIs(await models());
    };
    fetchModels();
  }, [models]);
  useEffect(() => {
    setSelectedAPI(availableAPIs[0]);
  }, [availableAPIs]);

  const onAPIChange = (e: ChangeEvent<HTMLSelectElement>) => {
    const index = parseInt(e.target.value);
    if (index >= availableAPIs.length) {
      throw new Error(`Invalid index choice: ${index}`);
    }
    setSelectedAPI(availableAPIs[index]);
    setOllamaModels([]);
    setOllamaEndpoint('');
    setOpenaiModels([]);
    setOpenaiEndpoint('');
    setOpenaiCredentials('');
  };

  const onSubmit: SubmitHandler<FormValues> = (data: FormValues) => {
    const slug = data.model;
    const name = modelName;
    let endpoint: string | undefined;
    let credentials: string | undefined;
    if (selectedAPI?.name === 'Ollama') {
      endpoint = ollamaEndpoint;
      credentials = undefined;
    } else if (selectedAPI?.name === 'OpenAICompatible') {
      endpoint = openaiEndpoint;
      credentials = openaiCredentials || undefined;
    } else {
      endpoint = data.endpoint;
      credentials = data.credentials;
    }
    if (slug === null || slug === '') {
      notify({ type: 'error', message: 'You must select a model' });
      return;
    }
    if (name === null || name === '') {
      notify({ type: 'error', message: 'You must select a name' });
      return;
    }
    if (selectedAPI?.name === 'Ollama' && (!endpoint || endpoint === '')) {
      notify({ type: 'error', message: 'You must provide an Ollama endpoint' });
      return;
    }
    if (selectedAPI?.name === 'OpenAICompatible' && (!endpoint || endpoint === '')) {
      notify({ type: 'error', message: 'You must provide an endpoint URL' });
      return;
    }
    add({
      slug,
      name,
      api: selectedAPI.name as SupportedAPI,
      endpoint,
      credentials,
    });
  };

  const onNameChange = (e: ChangeEvent<HTMLInputElement>) => {
    setModelName(e.target.value);
  };

  const onModelChange = (e: ChangeEvent<HTMLSelectElement | HTMLInputElement>) => {
    setModelName(selectedAPI.name + '-' + e.target.value);
  };

  const handleFetchOllamaModels = async () => {
    if (!ollamaEndpoint) {
      notify({ type: 'error', message: 'Please enter an Ollama endpoint URL' });
      return;
    }
    setOllamaLoading(true);
    try {
      const models = await fetchOllamaModels(ollamaEndpoint);
      setOllamaModels(models);
      if (models.length > 0) {
        setValue('model', models[0].slug);
        setModelName(selectedAPI.name + '-' + models[0].slug);
      }
    } catch (e) {
      notify({ type: 'error', message: `Failed to fetch models: ${e}` });
      setOllamaModels([]);
    } finally {
      setOllamaLoading(false);
    }
  };

  const handleFetchOpenAIModels = async () => {
    if (!openaiEndpoint) {
      notify({ type: 'error', message: 'Please enter an endpoint URL' });
      return;
    }
    setOpenaiLoading(true);
    try {
      const models = await fetchOpenAICompatibleModels(openaiEndpoint, openaiCredentials);
      setOpenaiModels(models);
      if (models.length > 0) {
        setValue('model', models[0].slug);
        setModelName(selectedAPI.name + '-' + models[0].slug);
      }
    } catch (e) {
      notify({ type: 'error', message: `Failed to fetch models: ${e}` });
      setOpenaiModels([]);
    } finally {
      setOpenaiLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <label htmlFor="api">API </label>
      <select id="api" defaultValue={0} onChange={onAPIChange} name="api">
        {availableAPIs.map((api, index) => (
          <option key={index} value={index}>
            {api.name}
          </option>
        ))}
      </select>
      {(() => {
        const inputs = [];
        if (selectedAPI !== undefined) {
          if (selectedAPI.name === 'Ollama') {
            inputs.push(
              <div key="endpoint">
                <label htmlFor="endpoint">Endpoint</label>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  <input
                    type="text"
                    id="endpoint"
                    placeholder="enter the url of the Ollama server"
                    value={ollamaEndpoint}
                    onChange={(e) => setOllamaEndpoint(e.target.value)}
                    style={{ flex: 1 }}
                  />
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={handleFetchOllamaModels}
                    disabled={ollamaLoading}
                  >
                    {ollamaLoading ? 'Loading...' : 'Fetch models'}
                  </button>
                </div>
              </div>,
            );
            if (ollamaModels.length > 0) {
              inputs.push(
                <div key="model">
                  <label htmlFor="model">Model</label>
                  <select id="model" {...register('model', { onChange: onModelChange })}>
                    {ollamaModels.map((model) => (
                      <option key={model.slug} value={model.slug}>
                        {model.name}
                      </option>
                    ))}
                  </select>
                </div>,
              );
            }
          } else if (selectedAPI.name === 'OpenAICompatible') {
            inputs.push(
              <div key="endpoint">
                <label htmlFor="endpoint">Endpoint</label>
                <input
                  type="text"
                  id="endpoint"
                  placeholder="e.g. https://api.example.com/v1"
                  value={openaiEndpoint}
                  onChange={(e) => setOpenaiEndpoint(e.target.value)}
                />
              </div>,
            );
            inputs.push(
              <div key="credentials">
                <label htmlFor="credentials">API Credentials</label>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  <input
                    type="text"
                    id="credentials"
                    placeholder="API key (optional)"
                    autoComplete="off"
                    value={openaiCredentials}
                    onChange={(e) => setOpenaiCredentials(e.target.value)}
                    style={{ flex: 1 }}
                  />
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={handleFetchOpenAIModels}
                    disabled={openaiLoading}
                  >
                    {openaiLoading ? 'Loading...' : 'Fetch models'}
                  </button>
                </div>
              </div>,
            );
            if (openaiModels.length > 0) {
              inputs.push(
                <div key="model">
                  <label htmlFor="model">Model</label>
                  <select id="model" {...register('model', { onChange: onModelChange })}>
                    {openaiModels.map((model) => (
                      <option key={model.slug} value={model.slug}>
                        {model.name}
                      </option>
                    ))}
                  </select>
                </div>,
              );
            } else {
              inputs.push(
                <div key="model">
                  <label htmlFor="model">Model</label>
                  <input
                    type="text"
                    id="model"
                    placeholder="ID of the model"
                    {...register('model', { onChange: onModelChange })}
                  />
                </div>,
              );
            }
          } else {
            inputs.push(
              <div key="model">
                <label htmlFor="model">Model</label>
                <input
                  type="text"
                  id="model"
                  placeholder="ID of the model"
                  {...register('model', { onChange: onModelChange })}
                />
              </div>,
            );
            if (selectedAPI.name !== 'OpenRouter')
              inputs.push(
                <div key="endpoint">
                  <label htmlFor="endpoint">Endpoint</label>
                  <input
                    type="text"
                    id="endpoint"
                    placeholder="enter the url of the endpoint"
                    {...register('endpoint')}
                  />
                </div>,
              );
          }

          if (selectedAPI.name !== 'Ollama' && selectedAPI.name !== 'OpenAICompatible')
            inputs.push(
              <div key="credentials">
                <label htmlFor="credentials">API Credentials</label>
                <input
                  type="text"
                  id="credentials"
                  placeholder="API key"
                  autoComplete="off"
                  {...register('credentials')}
                />
              </div>,
            );
        }
        return inputs;
      })()}
      <label htmlFor="name">Name</label>
      <input
        type="text"
        id="name"
        value={modelName}
        {...register('name', { onChange: onNameChange })}
      />

      <button type="submit" className="btn-submit">
        Add
      </button>
    </form>
  );
};
