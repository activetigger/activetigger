import { FC, useEffect, useState } from 'react';
import DataTable from 'react-data-table-component';
import { Controller, SubmitHandler, useForm, useWatch } from 'react-hook-form';
import Select from 'react-select';

import { omit } from 'lodash';
import { unparse } from 'papaparse';
import { useCreateValidSet, useDropEvalSet } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { loadFile } from '../../core/utils';

import { Modal } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { EvalSetModel } from '../../types';

// format of the data table
export interface DataType {
  headers: string[];
  data: Record<string, string | number | bigint>[];
  filename: string;
}

export interface EvalSetsManagementModel {
  projectSlug: string;
  currentScheme: string;
  dataset: string;
  exist: boolean;
}

// component
export const EvalSetsManagement: FC<EvalSetsManagementModel> = ({
  projectSlug,
  currentScheme,
  dataset,
  exist,
}) => {
  // form management
  const datasetCleanForPrinting = dataset == 'test' ? 'Test' : 'Validation';
  const { register, control, handleSubmit, setValue } = useForm<EvalSetModel & { files: FileList }>(
    {
      defaultValues: { scheme: currentScheme },
    },
  );

  const createValidSet = useCreateValidSet(); // API call
  const { notify } = useNotifications();

  const dropEvalSet = useDropEvalSet(projectSlug); // API call to drop existing test set
  const navigate = useNavigate(); // for navigation after drop

  const [alertDrop, setAlertDrop] = useState<boolean>(false);

  const [data, setData] = useState<DataType | null>(null);
  const files = useWatch({ control, name: 'files' });
  // available columns
  const columns = data?.headers.map((h) => (
    <option key={`${h}`} value={`${h}`}>
      {h}
    </option>
  ));

  // convert paquet file in csv if needed when event on files
  useEffect(() => {
    console.log('checking file', files);
    if (files && files.length > 0) {
      const file = files[0];
      loadFile(file).then((data) => {
        if (data === null) {
          notify({ type: 'error', message: 'Error reading the file' });
          return;
        }
        setData(data);
        setValue('n_eval', data.data.length - 1);
      });
    }
  }, [files, setValue, notify]);

  // action when form validated
  const onSubmit: SubmitHandler<EvalSetModel & { files: FileList }> = async (formData) => {
    if (data) {
      if (!formData.col_id || !formData.cols_text || !formData.n_eval) {
        notify({ type: 'error', message: 'Please fill all the fields.' });
        return;
      }
      const csv = data ? unparse(data.data, { header: true, columns: data.headers }) : '';
      formData.scheme = currentScheme;
      await createValidSet(projectSlug, dataset, {
        ...omit(formData, 'files'),
        csv,
        filename: data.filename,
      });
    }
  };

  const capFirstLetter = (word: string) => {
    return word.charAt(0).toUpperCase() + word.slice(1);
  };

  return (
    <div>
      <h4 className="subsection">{capFirstLetter(dataset)} set</h4>
      {exist && (
        <button
          className="btn-drop-dataset"
          onClick={() => {
            setAlertDrop(true);
          }}
        >
          Drop {datasetCleanForPrinting} set
        </button>
      )}

      {!exist && (
        <form onSubmit={handleSubmit(onSubmit)}>
          <div className="col-lg-6">
            <div className="explanations">
              No {datasetCleanForPrinting} data set has been created. You can upload a{' '}
              {datasetCleanForPrinting} set. Careful : all features will be dropped and need to be
              computed again, and id will be modified with "imported-". You are responsible to check
              that the elements are not already in the train set.
            </div>
            <label htmlFor="csvFile">File to upload</label>
            <input id="csvFile" className="form-control" type="file" {...register('files')} />
            {
              // display datable if data available
              data !== null && (
                <div>
                  <div className="explanations">Preview</div>
                  <div>
                    Size of the dataset : <b>{data.data.length - 1}</b>
                  </div>
                  <DataTable<Record<DataType['headers'][number], string | number>>
                    customStyles={{responsiveWrapper: {
                      style: {
                              maxWidth: '600px',
                              overflowX: 'auto',
                            },
                          },
                        }}
                    columns={data.headers.map((h) => ({
                      name: h,
                      selector: (row) => row[h],
                      format: (row) => {
                        const v = row[h];
                        return typeof v === 'bigint' ? Number(v) : v;
                      },
                      width: '200px',
                      wrap:true,
                    }))}
                    data={
                      data.data.slice(0, 5) as Record<keyof DataType['headers'], string | number>[]
                    }
                  />
                </div>
              )
            }
            {
              // only display if data
              data != null && (
                <div>
                  <label htmlFor="col_id">ID column (IDs must be unique)</label>
                  <select id="col_id" disabled={data === null} {...register('col_id')}>
                    {columns}
                  </select>

                  <label htmlFor="cols_text">
                    Text columns (all the selected fields will be concatenated)
                  </label>
                  <Controller
                    name="cols_text"
                    control={control}
                    render={({ field: { onChange } }) => (
                      <Select
                        options={(data?.headers || []).map((e) => ({ value: e, label: e }))}
                        isMulti
                        onChange={(selectedOptions) => {
                          onChange(
                            selectedOptions ? selectedOptions.map((option) => option.value) : [],
                          );
                        }}
                      />
                    )}
                  />
                  <label htmlFor="col_label">
                    Column(s) for existing annotations (optional, labels must already exist in the
                    current scheme, otherwise they are ignored)
                  </label>
                  <select id="col_label" disabled={data === null} {...register('col_label')}>
                    <option key="none" value="">
                      No label
                    </option>

                    {columns}
                  </select>
                  <label htmlFor="n_test">Number of rows to import</label>
                  <input id="n_test" type="number" {...register('n_eval')} />

                  <button type="submit" className="btn-submit">
                    Create
                  </button>
                </div>
              )
            }
          </div>
        </form>
      )}
      <Modal show={alertDrop} onHide={() => setAlertDrop(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Drop the validation set</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          Do you really want to drop the {dataset == 'test' ? 'Test' : 'Validation'} set? All
          features and quick models will be dropped and need to be recomputed.
          <div className="horizontal">
            <button onClick={() => setAlertDrop(false)} style={{ flex: '1 1 auto' }}>
              Cancel
            </button>
            <button
              className="btn-danger"
              onClick={() => {
                dropEvalSet(dataset).then(() => {
                  navigate(0);
                });
              }}
              style={{ flex: '1 1 auto' }}
            >
              Confirm
            </button>
          </div>
        </Modal.Body>
      </Modal>
    </div>
  );
};
