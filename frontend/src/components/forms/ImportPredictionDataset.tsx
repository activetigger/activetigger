//import { omit } from 'lodash';
import { FC, useEffect, useState } from 'react';
import DataTable from 'react-data-table-component';
import { SubmitHandler, useForm, useWatch } from 'react-hook-form';

import { omit } from 'lodash';
import { FaCloudDownloadAlt } from 'react-icons/fa';
import { useAddFile, useGetPredictionsFile, usePredictOnDataset } from '../../core/api';
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
    defaultValues: {},
  });
  const { addFile, progression, cancel } = useAddFile();
  const predict = usePredictOnDataset(); // API call
  const { notify } = useNotifications();
  const [importingDataset, setImportingDataset] = useState<boolean>(false); // state for the data
  const [data, setData] = useState<DataType | null>(null);
  const files = useWatch({ control, name: 'files' });
  // available columns
  const columns = data?.headers.map((h) => (
    <option key={h} value={h}>
      {h}
    </option>
  ));

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
      if (!formData.id || !formData.text) {
        notify({ type: 'error', message: 'Please fill all the fields.' });
        return;
      }
      setImportingDataset(true);
      // first upload file
      await addFile(projectSlug, formData.files[0]);
      // then launch prediction
      await predict(projectSlug, scheme, modelName, {
        ...omit(formData, 'files'),
        filename: data.filename,
      });
      setData(null);
      setImportingDataset(false);
      reset();
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
                <label htmlFor="col_text">Column for text</label>
                <select id="col_text" disabled={data === null} {...register('text')}>
                  <option key="none"></option>
                  {columns}
                </select>
              </div>
              <button type="submit" className="btn-submit">
                Launch the prediction on the imported dataset
              </button>
            </div>
          )
        }
      </form>
      {data && importingDataset && <UploadProgressBar progression={progression} cancel={cancel} />}
    </div>
  );
};
