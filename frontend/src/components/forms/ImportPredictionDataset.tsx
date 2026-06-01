//import { omit } from 'lodash';
import { FC, useEffect, useMemo, useState } from 'react';
import DataTable from 'react-data-table-component';
import { Controller, SubmitHandler, useForm, useWatch } from 'react-hook-form';

import { omit } from 'lodash';
import { FaCloudDownloadAlt } from 'react-icons/fa';
import Select from 'react-select';
import { useAddFile, useGetPredictionsFile, usePredictOnDataset } from '../../core/api';
import { formatUploadError } from '../../core/HTTPError';
import { useNotifications } from '../../core/notifications';
import { loadFile } from '../../core/utils';
import { TextDatasetModel } from '../../types';
import { UploadProgressBar } from '../UploadProgressBar';

// format of the data table
export interface DataType {
  headers: string[];
  data: Record<string, string | number | bigint>[];
  filename: string;
}

export interface ImportPredictionDatasetProps {
  projectSlug: string;
  scheme: string;
  modelName: string;
  availablePredictionExternal?: boolean;
}

// component
export const ImportPredictionDataset: FC<ImportPredictionDatasetProps> = ({
  projectSlug,
  scheme,
  modelName,
  availablePredictionExternal,
}) => {
  const maxSizeMB = 300;
  const maxSize = maxSizeMB * 1024 * 1024; // 100 MB in bytes

  const { getPredictionsFile } = useGetPredictionsFile(projectSlug || null);

  // form management
  const { register, control, handleSubmit, reset } = useForm<
    TextDatasetModel & { files: FileList }
  >({
    defaultValues: { cols_text: [] },
  });
  const { addFile, progression, cancel } = useAddFile();
  const predict = usePredictOnDataset(); // API call
  const { notify } = useNotifications();
  const [importingDataset, setImportingDataset] = useState<boolean>(false); // state for the data
  const [phase, setPhase] = useState<'uploading' | 'finalizing' | 'queuing' | null>(null);
  const [data, setData] = useState<DataType | null>(null);
  const files = useWatch({ control, name: 'files' });
  // available columns
  const columns = data?.headers.map((h) => (
    <option key={h} value={h}>
      {h}
    </option>
  ));
  const availableFields = useMemo(
    () => data?.headers.filter((h) => !!h).map((h) => ({ value: h, label: h })) ?? [],
    [data],
  );

  // convert paquet file in csv if needed when event on files
  useEffect(() => {
    if (files && files.length > 0) {
      const file = files[0];
      if (file.size > maxSize) {
        notify({
          type: 'error',
          message: `File is too large (maximum size: ${maxSizeMB} MB)`,
        });
        return;
      }
      loadFile(file).then((data) => {
        if (data === null) {
          notify({ type: 'error', message: 'Error reading the file.' });
          return;
        }
        setData(data);
      });
    }
  }, [files, maxSize, notify, setData]);
  // action when form validated
  const onSubmit: SubmitHandler<TextDatasetModel & { files: FileList }> = async (formData) => {
    if (data) {
      if (!formData.id || !formData.cols_text || formData.cols_text.length === 0) {
        notify({ type: 'error', message: 'Please fill all the fields.' });
        return;
      }
      const file = formData.files[0];
      setImportingDataset(true);
      setPhase('uploading');
      let uploaded = false;
      try {
        // first upload file — if this fails we must NOT call predict, otherwise
        // the backend returns a misleading 404 because the file is not on disk.
        await addFile(projectSlug, file);
        uploaded = true;
        setPhase('queuing');
        await predict(projectSlug, scheme, modelName, {
          ...omit(formData, 'files'),
          filename: data.filename,
        });
        setData(null);
        reset();
      } catch (error) {
        let message: string;
        if (uploaded) {
          // The upload reached the server; the failure is in the prediction kickoff.
          const raw = error instanceof Error ? error.message : String(error);
          if (/timeout/i.test(raw)) {
            message =
              'Prediction request timed out. The server accepted the file but the kickoff is taking longer than expected — it may still be queued. Refresh the page in a minute to check, or retry on a smaller file.';
          } else if (/unreachable|network/i.test(raw)) {
            message =
              'Could not reach the server when starting the prediction. The file was uploaded — check your connection and retry.';
          } else {
            message = `Prediction kickoff failed: ${raw}`;
          }
        } else {
          message = formatUploadError(error, file.size);
        }
        notify({ type: 'error', message });
        console.error('Prediction on imported dataset failed:', error);
      } finally {
        setImportingDataset(false);
        setPhase(null);
      }
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit(onSubmit)}>
        <div className="explanations">
          One predicted, you can export them in Export as the external dataset. If you predict on a
          new dataset, it will erase the previous one.
        </div>
        {availablePredictionExternal && (
          <div className="alert alert-warning">
            You already have a prediction for this model.{' '}
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault();
                getPredictionsFile(modelName, 'csv', 'external', scheme);
              }}
              className="text-blue-600 hover:underline"
            >
              You can export it <FaCloudDownloadAlt />.
            </a>{' '}
            If you continue, it will be replaced.
          </div>
        )}
        <div className="container">
          <div className="w-75">
            <label htmlFor="csvFile">Import text dataset to predict</label>
            <input className="form-control" id="csvFile" type="file" {...register('files')} />
            {
              // display datable if data available
              data !== null && (
                <>
                  <div className="explanations">Preview</div>
                  <div>
                    Size of the dataset : <b>{data.data.length - 1}</b>
                  </div>
                  <DataTable<Record<DataType['headers'][number], string | number>>
                    responsive
                    columns={data.headers.map((h) => ({
                      name: h,
                      selector: (row) => row[h],
                      format: (row) => {
                        const v = row[h];
                        return typeof v === 'bigint' ? Number(v) : v;
                      },
                      grow: 1,
                    }))}
                    data={
                      data.data.slice(0, 5) as Record<keyof DataType['headers'], string | number>[]
                    }
                  />
                </>
              )
            }
          </div>
        </div>

        {
          // only display if data
          data != null && (
            <div className="w-50">
              <div>
                <label htmlFor="col_id">
                  Column for id (they need to be unique, otherwise replaced by a number)
                </label>
                <select id="col_id" disabled={data === null} {...register('id')}>
                  {columns}
                </select>
              </div>
              <div>
                <label htmlFor="cols_text">
                  Text columns (selected fields will be concatenated)
                </label>
                <Controller
                  name="cols_text"
                  control={control}
                  defaultValue={[]}
                  render={({ field: { value, onChange } }) => (
                    <Select
                      inputId="cols_text"
                      options={availableFields}
                      isMulti
                      isDisabled={data === null}
                      value={
                        value
                          ? value
                              .map((v: string) => availableFields.find((opt) => opt.value === v))
                              .filter(Boolean)
                          : []
                      }
                      onChange={(selectedOptions) => {
                        onChange(
                          selectedOptions ? selectedOptions.map((option) => option?.value) : [],
                        );
                      }}
                    />
                  )}
                />
              </div>
              <button type="submit" className="btn-submit">
                Launch the prediction on the imported dataset
              </button>
            </div>
          )
        }
      </form>
      {data &&
        importingDataset &&
        (() => {
          const bytesDone =
            phase === 'uploading' &&
            !!progression.loaded &&
            !!progression.total &&
            progression.loaded >= progression.total;
          const statusMessage =
            phase === 'queuing'
              ? 'Starting prediction on the server…'
              : bytesDone
                ? 'Finalizing upload on the server (writing file to disk)…'
                : 'Uploading dataset';
          const showProgress = phase === 'uploading' && !bytesDone;
          return (
            <UploadProgressBar
              progression={progression}
              cancel={phase === 'uploading' ? cancel : undefined}
              statusMessage={statusMessage}
              showProgress={showProgress}
            />
          );
        })()}
    </div>
  );
};
