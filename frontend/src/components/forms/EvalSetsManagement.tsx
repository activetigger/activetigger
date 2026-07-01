import { FC, useCallback, useEffect, useRef, useState } from 'react';
import DataTable from 'react-data-table-component';
import { Controller, SubmitHandler, useForm, useWatch } from 'react-hook-form';
import Select from 'react-select';

import { omit } from 'lodash';
import { unparse } from 'papaparse';
import { Modal } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import { useCreateValidSet, useDropEvalSet, useStopProcesses } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { useAppContext } from '../../core/useAppContext';
import { loadFile } from '../../core/utils';
import { EvalSetModel } from '../../types';

import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { UploadProgressBar } from '../UploadProgressBar';

// format of the data table
export interface DataType {
  headers: string[];
  data: Record<string, string | number | bigint>[];
  filename: string;
}

export interface EvalSetsManagementModel {
  projectSlug: string;
  dataset: string;
  exist: boolean;
}

// component
export const EvalSetsManagement: FC<EvalSetsManagementModel> = ({
  projectSlug,
  dataset,
  exist,
}) => {
  // form management
  const datasetCleanForPrinting = dataset == 'test' ? 'Test' : 'Validation';
  const { register, control, handleSubmit, setValue } = useForm<EvalSetModel & { files: FileList }>(
    {
      defaultValues: { cols_label: [] },
    },
  );
  const {
    appContext: { currentProject },
  } = useAppContext();
  const proj_errors = currentProject?.errors || [];
  const availableSchemes = Object.keys(currentProject?.schemes.available || {});

  const { progression, createValidSet, cancel } = useCreateValidSet(); // API call
  const { notify } = useNotifications();

  const dropEvalSet = useDropEvalSet(projectSlug); // API call to drop existing test set
  const { stopProcesses } = useStopProcesses(projectSlug); //API call to stop the current process of adding evalset
  const navigate = useNavigate(); // for navigation after drop

  const [alertDrop, setAlertDrop] = useState<boolean>(false);

  const [data, setData] = useState<DataType | null>(null);
  const files = useWatch({ control, name: 'files' });
  //Local storage variables
  const add_eval_storageKey = `add-evalset-${dataset}-${projectSlug}`; //
  //const processIdKey = `evalset-process-id-${dataset}-${projectSlug}`;//will be unused

  //Uploading State
  const [uploading, setUploading] = useState<boolean>(
    () => sessionStorage.getItem(add_eval_storageKey) === 'true',
  );
  const uploadingRef = useRef(uploading);
  const cancelRef = useRef(cancel);
  //Controller state
  const [displayCancel, setDisplayCancel] = useState<AbortController | undefined>(undefined);
  //Task Errors Management
  const errorCountAtSubmit = useRef(0);
  //Set Max Duration
  const maxDuration = 5 * 60 * 1000; // 5 minutes in milliseconds

  //no old error that may block the workflow

  //handle uploading
  const isUploading = useCallback(
    (val: boolean) => {
      if (!val) {
        cancelRef.current = undefined;
      }
      val
        ? sessionStorage.setItem(add_eval_storageKey, 'true')
        : sessionStorage.removeItem(add_eval_storageKey);
      setUploading(val);
    },
    [add_eval_storageKey],
  );

  //sync uploadref with state
  useEffect(() => {
    uploadingRef.current = uploading;
  }, [uploading]);

  useEffect(() => {
    if (exist && uploadingRef.current) {
      isUploading(false);
    }
  }, [exist]); // eslint-disable-line react-hooks/exhaustive-deps

  //Error Management : handle errors from backend
  useEffect(() => {
    if (!uploading) return;
    const len_new_errors = proj_errors.length - errorCountAtSubmit.current;
    if (len_new_errors > 0) {
      const newErrors = proj_errors.slice(-len_new_errors);
      const evalsetError = [...newErrors]
        .reverse()
        .find((e) => Array.isArray(e) && (e[0] as string).includes(`add_evalset_${dataset}`));
      if (evalsetError) {
        notify({ type: 'error', message: (evalsetError as string[]).join('-') });
        isUploading(false);
        //setTimeout(() => { navigate(0); }, 750);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [proj_errors, uploading]);

  //handle Timeout :if task surpasses 5 minutes (Case of server resrtart or shut down)
  useEffect(() => {
    if (!uploading) return;

    const timer = setTimeout(() => {
      console.warn('Upload timeout reached');
      isUploading(false);
    }, maxDuration);

    return () => clearTimeout(timer);
  }, [uploading, isUploading, maxDuration]);

  //setting the id
  //useEffect(() => {if (id) sessionStorage.setItem(processIdKey, id);}, [id]);

  //Cancel
  useEffect(() => {
    cancelRef.current = cancel;
    setDisplayCancel(cancel);
  }, [cancel]);

  //Handle cancel signal
  useEffect(() => {
    if (!uploading) {
      return;
    }
    const stop = () => {
      //const currentId = sessionStorage.getItem(processIdKey);
      stopProcesses(`add_evalset_${dataset}`);
      //sessionStorage.removeItem(processIdKey);
      isUploading(false);
      //refresh
      setTimeout(() => {
        navigate(0);
      }, 750);
    };
    if (cancel?.signal) {
      const signal = cancel.signal;
      signal.addEventListener('abort', stop);
      return () => {
        signal.removeEventListener('abort', stop);
      };
    }
    const n_cancel = new AbortController();
    cancelRef.current = n_cancel;
    setDisplayCancel(n_cancel);
    n_cancel.signal.addEventListener('abort', stop);
    return () => {
      n_cancel.signal.removeEventListener('abort', stop);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cancel, uploading]);

  // available columns
  const columns = data?.headers.map((h) => (
    <option key={`${h}`} value={`${h}`}>
      {h}
    </option>
  ));

  // convert paquet file in csv if needed when event on files
  useEffect(() => {
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
      // check that the selected ID column contains unique values
      if (formData.col_id && formData.col_id !== 'row_number' && data) {
        const idValues = data.data.map((row) => row[formData.col_id]);
        const uniqueValues = new Set(idValues);
        if (uniqueValues.size !== idValues.length) {
          const nDuplicates = idValues.length - uniqueValues.size;
          notify({
            type: 'error',
            message: `The selected ID column contains ${nDuplicates} duplicate values. Please choose a column with unique values or use 'Row number'.`,
          });
          return;
        }
      }
      if (!formData.col_id || !formData.cols_text || !formData.n_eval) {
        notify({ type: 'error', message: 'Please fill all the fields.' });
        return;
      }
      // each selected label column name must match an existing scheme
      const cols_label = formData.cols_label ?? [];
      const unknownSchemes = cols_label.filter((c) => !availableSchemes.includes(c));
      if (unknownSchemes.length > 0) {
        notify({
          type: 'error',
          message: `The selected label column(s) do not match any existing scheme: ${unknownSchemes.join(', ')}. Column names must match a scheme name.`,
        });
        return;
      }
      const csv = data ? unparse(data.data, { header: true, columns: data.headers }) : '';
      errorCountAtSubmit.current = proj_errors.length;
      isUploading(true);
      try {
        const res = await createValidSet(projectSlug, dataset, {
          ...omit(formData, 'files'),
          csv,
          filename: data.filename,
        });
        if (!res) {
          isUploading(false);
        } //sessionStorage.removeItem(processIdKey);}
      } catch (e) {
        isUploading(false);
        notify({ type: 'error', message: 'Failed to start the process' });
      }
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
          </div>
          {
            // display datable if data available (pulled out of col-lg-6 so long rows can breathe)
            data !== null && (
              <div className="my-3">
                <div className="explanations">Preview</div>
                <div>
                  Size of the dataset : <b>{data.data.length - 1}</b>
                </div>
                <DataTable<Record<DataType['headers'][number], string | number>>
                  customStyles={{
                    responsiveWrapper: {
                      style: {
                        maxWidth: '100%',
                        overflowX: 'auto',
                      },
                    },
                    cells: {
                      style: {
                        maxHeight: '120px',
                        overflowY: 'auto',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                        paddingTop: '8px',
                        paddingBottom: '8px',
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
                    minWidth: '250px',
                    wrap: true,
                  }))}
                  data={
                    data.data.slice(0, 5) as Record<keyof DataType['headers'], string | number>[]
                  }
                />
              </div>
            )
          }
          <div className="col-lg-6">
            {
              // only display if data
              data != null && (
                <div>
                  <label htmlFor="col_id">ID column (IDs must be unique)</label>
                  <select id="col_id" disabled={data === null} {...register('col_id')}>
                    <option key="row_number" value="row_number">
                      Row number
                    </option>
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
                  <label htmlFor="cols_label">
                    Columns for existing annotations (optional)
                    <HiOutlineQuestionMarkCircle className="search" />
                    <Tooltip anchorSelect=".search">
                      Select one or more columns whose name matches an existing scheme name: each
                      column will be imported as annotations for the scheme with the same name.
                      Labels must already exist in the matching scheme.
                    </Tooltip>
                    {availableSchemes.length > 0 && (
                      <div className="explanations">
                        Available schemes: {availableSchemes.join(', ')}
                      </div>
                    )}
                  </label>
                  <Controller
                    name="cols_label"
                    control={control}
                    render={({ field: { onChange, value } }) => (
                      <Select
                        inputId="cols_label"
                        options={(data?.headers || []).map((e) => ({
                          value: e,
                          label: availableSchemes.includes(e) ? e : `${e} (not a scheme)`,
                          isDisabled: !availableSchemes.includes(e),
                        }))}
                        isMulti
                        value={(value || []).map((v: string) => ({ value: v, label: v }))}
                        onChange={(selectedOptions) => {
                          onChange(
                            selectedOptions ? selectedOptions.map((option) => option.value) : [],
                          );
                        }}
                      />
                    )}
                  />
                  <label htmlFor="n_test">Number of rows to import</label>
                  <input id="n_test" type="number" {...register('n_eval')} />

                  <button type="submit" className="btn-submit" disabled={uploading}>
                    {uploading ? 'Uploading File ...' : 'Create'}
                  </button>
                </div>
              )
            }
          </div>
        </form>
      )}
      {uploading && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            zIndex: 10,
          }}
        >
          <UploadProgressBar progression={progression} cancel={cancel || displayCancel} />
        </div>
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
