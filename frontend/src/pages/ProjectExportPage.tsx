import { FC, useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';

import { Tab, Tabs } from 'react-bootstrap';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { ModelPredict } from '../components/ModelPredict';
import {
  useGetAnnotationsFile,
  useGetFeaturesFile,
  useGetModelFile,
  useGetPredictionsFile,
  useGetProjectionFile,
  useGetRawDataFile,
  useGetStaticUrls,
} from '../core/api';
import config from '../core/config';
import { useAppContext } from '../core/useAppContext';
import { useAuth } from '../core/useAuth';

/**
 * Component to display the export page
 */
export const ProjectExportPage: FC = () => {
  const { projectName } = useParams();

  // get the current state of the project
  const {
    appContext: { currentProject: project, currentScheme },
  } = useAppContext();
  const { authenticatedUser } = useAuth();

  const [format, setFormat] = useState<string>('csv');
  const [features, setFeatures] = useState<string[] | null>(null);
  const [model, setModel] = useState<string | null>(null);

  const availableFeatures = project?.features.available ? project?.features.available : [];
  const availableProjection =
    authenticatedUser?.username && project?.projections.available[authenticatedUser?.username]
      ? project?.projections.available[authenticatedUser?.username]
      : null;
  const availableModels =
    currentScheme && project?.languagemodels.available[currentScheme]
      ? Object.keys(project?.languagemodels.available[currentScheme])
      : [];
  const availablePredictionAll =
    (currentScheme &&
      model &&
      project?.languagemodels?.available?.[currentScheme]?.[model]?.['predicted']) ??
    false;
  const availablePredictionTest =
    (currentScheme &&
      model &&
      project?.languagemodels?.available?.[currentScheme]?.[model]?.['tested']) ??
    false;
  const availablePredictionExternal =
    (currentScheme &&
      model &&
      project?.languagemodels?.available?.[currentScheme]?.[model]?.['predicted_external']) ??
    false;

  const { getFeaturesFile } = useGetFeaturesFile(projectName || null);
  const { getAnnotationsFile } = useGetAnnotationsFile(projectName || null);
  const { getPredictionsFile } = useGetPredictionsFile(projectName || null);
  const { getModelFile } = useGetModelFile(projectName || null);
  const { getRawDataFile } = useGetRawDataFile(projectName || null);
  // const { getPredictionsQuickModelFile } = useGetPredictionsQuickmodelFile(projectName || null);
  const { getProjectionFile } = useGetProjectionFile(projectName || null);
  const { staticUrls, reFetchUrl } = useGetStaticUrls(projectName || null, model);

  useEffect(() => {
    reFetchUrl();
  }, [model, reFetchUrl]);

  return (
    <ProjectPageLayout projectName={projectName} currentAction="export">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <div className="explanations">Predict and export annotations and models</div>
            <div className="d-flex align-items-center mt-2">
              Select a file format
              <select
                className="form-select w-25 mx-2"
                onChange={(e) => {
                  setFormat(e.currentTarget.value);
                }}
              >
                <option key="csv">csv</option>
                <option key="xlsx">xlsx</option>
                <option key="parquet">parquet</option>
              </select>
            </div>
            <Tabs id="panel" className="mt-3" defaultActiveKey="annotations">
              <Tab eventKey="annotations" title="Annotations">
                <div>
                  <button
                    className="btn-secondary-action"
                    onClick={() => {
                      if (currentScheme) getAnnotationsFile(currentScheme, format, 'train');
                    }}
                  >
                    Tags: train
                  </button>
                  {project?.params.valid && (
                    <button
                      className="btn-secondary-action"
                      onClick={() => {
                        if (currentScheme) getAnnotationsFile(currentScheme, format, 'valid');
                      }}
                    >
                      Tags: validation
                    </button>
                  )}
                  {project?.params.test && (
                    <button
                      className="btn-secondary-action"
                      onClick={() => {
                        if (currentScheme) getAnnotationsFile(currentScheme, format, 'test');
                      }}
                    >
                      Tags: test
                    </button>
                  )}
                </div>
                <div>
                  <button
                    className="btn-secondary-action"
                    onClick={() => {
                      if (currentScheme) getAnnotationsFile('all', format, 'train');
                    }}
                  >
                    All annotations/schemes
                  </button>
                </div>
              </Tab>
              <Tab eventKey="features" title="Features">
                <div>
                  <select
                    className="form-select"
                    onChange={(e) => {
                      setFeatures(Array.from(e.target.selectedOptions, (option) => option.value));
                    }}
                    multiple
                  >
                    {(availableFeatures || []).map((e) => (
                      <option key={e}>{e}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <button
                    className="btn-secondary-action"
                    onClick={() => {
                      if (features) {
                        getFeaturesFile(features, format);
                      }
                    }}
                  >
                    Export selected features
                  </button>
                </div>
                {availableProjection && (
                  <div>
                    <button
                      className="btn-secondary-action"
                      onClick={() => {
                        if (availableProjection) {
                          getProjectionFile(format);
                        }
                      }}
                    >
                      Export current projection
                    </button>
                  </div>
                )}
              </Tab>
              <Tab eventKey="models" title="Models">
                <div>
                  <div>BERT models</div>
                  <div>
                    <select
                      className="form-select"
                      onChange={(e) => {
                        setModel(e.target.value);
                      }}
                    >
                      <option></option>
                      {(availableModels || []).map((e) => (
                        <option key={e}>{e}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <ModelPredict currentModel={model} />
                  </div>
                  <div>
                    {availablePredictionAll && (
                      <button
                        className="btn-secondary-action"
                        onClick={() => {
                          if (model) {
                            getPredictionsFile(model, format, 'all', currentScheme);
                          }
                        }}
                      >
                        Export prediction complete dataset
                      </button>
                    )}
                  </div>
                  <div>
                    {availablePredictionTest && (
                      <button
                        className="btn-secondary-action"
                        onClick={() => {
                          if (model) {
                            getPredictionsFile(model, format, 'test', currentScheme);
                          }
                        }}
                      >
                        Export prediction testset
                      </button>
                    )}
                  </div>
                  <div>
                    {availablePredictionExternal && (
                      <button
                        className="btn-secondary-action mt-4"
                        onClick={() => {
                          if (model) {
                            getPredictionsFile(model, format, 'external', currentScheme);
                          }
                        }}
                      >
                        Export prediction external dataset
                      </button>
                    )}
                  </div>
                  {/*
            small fix for the direct link when no nging
            */}
                  <div>
                    {model &&
                      (staticUrls && staticUrls.model ? (
                        <Link
                          to={
                            config.api.url.replace(/\/$/, '') + '/static/' + staticUrls.model.path
                          }
                          target="_blank"
                          download
                          className="btn btn-secondary mt-3"
                        >
                          Export fine-tuned model (large file)
                        </Link>
                      ) : (
                        <button
                          className="btn-secondary-action"
                          onClick={() => {
                            getModelFile(model);
                          }}
                        >
                          Export fine-tuned model
                        </button>
                      ))}
                  </div>
                </div>
              </Tab>
            </Tabs>
          </div>
          <hr className="mt-3" />

          {/*
            small fix for the direct link when no nging
            */}
          {staticUrls ? (
            <>
              <a
                href={config.api.url.replace(/\/$/, '') + '/static/' + staticUrls.dataset.path}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-3"
              >
                Static link to the raw dataset (click right, save as...)
              </a>
            </>
          ) : (
            <button
              className="btn btn-primary mt-3"
              onClick={() => {
                getRawDataFile();
              }}
            >
              Export raw dataset in parquet
            </button>
          )}
        </div>
      </div>
    </ProjectPageLayout>
  );
};
