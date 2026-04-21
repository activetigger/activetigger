import { stringify } from 'csv-stringify/browser/esm/sync';
import { omit } from 'lodash';
import { FC, useEffect, useState } from 'react';
import DataTable from 'react-data-table-component';
import { SubmitHandler, useForm, useWatch } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import { usePostAnnotationsFile } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { loadFile } from '../../core/utils';

interface ImportPropos {
  projectName: string | null;
  currentScheme: string | null;
}

export interface DataSetAnnotations {
  col_id: string;
  col_label: string;
}

export interface DataType {
  headers: string[];
  data: Record<string, string | number | bigint>[];
  filename: string;
}

export const ImportAnnotations: FC<ImportPropos> = ({ projectName, currentScheme }) => {
  const maxSizeMB = 100;
  const maxSize = maxSizeMB * 1024 * 1024; // 100 MB in bytes
  const { notify } = useNotifications();
  const postAnnotationsFile = usePostAnnotationsFile(projectName || null);
  const navigate = useNavigate();

  // form management
  const { register, control, handleSubmit, reset } = useForm<
    DataSetAnnotations & { files: FileList }
  >({
    defaultValues: {},
  });
  const [data, setData] = useState<DataType | null>(null);
  const files = useWatch({ control, name: 'files' }); // watch the files entry

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
  const onSubmit: SubmitHandler<DataSetAnnotations & { files: FileList }> = async (formData) => {
    if (data) {
      const csv = stringify(data.data, { header: true, columns: data.headers.filter(Boolean) });
      await postAnnotationsFile({
        ...omit(formData, 'files'),
        csv: csv,
        filename: data.filename,
        scheme: currentScheme || '',
      });
      setData(null);
      reset();
      notify({
        type: 'success',
        message: 'Annotations imported successfully!',
      });
      navigate(0);
    }
  };

  // available columns to display
  const columns = data?.headers.map((h) => (
    <option key={h} value={h}>
      {h}
    </option>
  ));

  return (
    <>
      <h4 className="subsection">Import annotations</h4>
      <form onSubmit={handleSubmit(onSubmit)} className="row">
        <div className="col-lg-6">
          <div className="alert alert-warning">
            You can import annotations for existing elements in the train set. Make sure to maintain
            a consistent identification system of the elements. If elements are already labelled,
            this annotation will prevail. Labels are not checked for existance.
            {/* TODO: Axel: Precise what happens: are unexisting labels created? are they skipped? */}
          </div>
          <label htmlFor="csvFile">File to upload</label>
          <input className="form-control" id="csvFile" type="file" {...register('files')} />
          {
            // display datable if data available
            data !== null && (
              <>
                <div className="explanations">Preview</div>
                <div>
                  Size of the dataset : <b>{data.data.length - 1}</b>
                </div>

                <div>
                  <DataTable<Record<DataType['headers'][number], string | number>>
                    columns={data.headers.map((h) => ({
                      name: h,
                      selector: (row) => row[h],
                      format: (row) => {
                        const v = row[h];
                        return typeof v === 'bigint' ? Number(v) : v;
                      },
                      width: '200px',
                    }))}
                    data={
                      data.data.slice(0, 5) as Record<keyof DataType['headers'], string | number>[]
                    }
                  />
                </div>

                <label htmlFor="col_id">
                  Column for id (they need to match exactly the original data)
                </label>
                <select id="col_id" disabled={data === null} {...register('col_id')}>
                  {columns}
                </select>

                <label htmlFor="col_label">
                  Column annotations (rows without annotation will be ignored)
                </label>
                <select id="col_label" disabled={data === null} {...register('col_label')}>
                  {columns}
                </select>

                <button type="submit" className="btn-submit">
                  Import annotations
                </button>
              </>
            )
          }
        </div>
      </form>
    </>
  );
};
