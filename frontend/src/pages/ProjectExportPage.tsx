import cx from 'classnames';
import { FC, useState } from 'react';
import { Link, useParams } from 'react-router-dom';

import PulseLoader from 'react-spinners/PulseLoader';
import { ProjectPageLayout } from '../components/layout/ProjectPageLayout';
import { ModelsPillDisplay } from '../components/ModelsPillDisplay';
import {
  useGetAnnotationsFile,
  useGetFeaturesFile,
  useGetModelFile,
  useGetPredictionsFile,
  useGetProjectionFile,
  useGetProjectSummary,
  useGetRawDataFile,
} from '../core/api';
import { downloadSummaryJson, downloadSummaryMd, ProjectSummary } from '../core/projectSummary';
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
  const [features, setFeatures] = useState<string[]>([]);
  const [model, setModel] = useState<string | null>(null);
  const [predictionLoading, setPredictionLoading] = useState<string | null>(null);

  const toggleFeature = (name: string) => {
    setFeatures((prev) => (prev.includes(name) ? prev.filter((f) => f !== name) : [...prev, name]));
  };

  // For span schemes the exports come from NER models; everything else
  // from BERT models. Same flag names, just a different manager in state.
  const kindScheme =
    currentScheme && project?.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].kind || 'multiclass'
      : 'multiclass';
  const isNer = kindScheme === 'span';
  const exportKind = isNer ? 'ner' : 'bert';
  const modelAvailabilityMap = isNer
    ? project?.nermodels?.available
    : project?.languagemodels?.available;

  const downloadPrediction = async (dataset: 'all' | 'test' | 'external') => {
    if (!model) return;
    setPredictionLoading(dataset);
    try {
      await getPredictionsFile(model, format, dataset, currentScheme, exportKind);
    } finally {
      setPredictionLoading(null);
    }
  };

  const availableFeatures = project?.features.available ? project?.features.available : [];
  const availableProjection =
    authenticatedUser?.username && project?.projections.available[authenticatedUser?.username]
      ? project?.projections.available[authenticatedUser?.username]
      : null;
  const availableModels =
    currentScheme && modelAvailabilityMap?.[currentScheme]
      ? Object.keys(modelAvailabilityMap[currentScheme])
      : [];
  const availablePredictionAll =
    (currentScheme && model && modelAvailabilityMap?.[currentScheme]?.[model]?.['predicted_all']) ??
    false;
  const availablePredictionTest =
    (currentScheme && model && modelAvailabilityMap?.[currentScheme]?.[model]?.['tested']) ?? false;
  const availablePredictionExternal =
    (currentScheme &&
      model &&
      modelAvailabilityMap?.[currentScheme]?.[model]?.['predicted_external']) ??
    false;

  const { getFeaturesFile } = useGetFeaturesFile(projectName || null);
  const { getAnnotationsFile } = useGetAnnotationsFile(projectName || null);
  const { getPredictionsFile } = useGetPredictionsFile(projectName || null);
  const { getModelFile } = useGetModelFile(projectName || null);
  const { getRawDataFile } = useGetRawDataFile(projectName || null);
  const { getProjectionFile } = useGetProjectionFile(projectName || null);
  const getProjectSummary = useGetProjectSummary();
  const [summaryLoading, setSummaryLoading] = useState<'json' | 'md' | null>(null);

  const downloadSummary = async (kind: 'json' | 'md') => {
    if (!projectName) return;
    setSummaryLoading(kind);
    try {
      const data = (await getProjectSummary(projectName)) as ProjectSummary;
      if (kind === 'json') downloadSummaryJson(data, projectName);
      else downloadSummaryMd(data, projectName);
    } finally {
      setSummaryLoading(null);
    }
  };

  return (
    <ProjectPageLayout projectName={projectName} currentAction="export">
      <div className="container-fluid">
        <div className="row">
          <div className="col-12">
            <div className="d-flex justify-content-between align-items-center mt-2">
              <div className="explanations mb-0">Predict and export annotations and models</div>
              <div className="d-flex align-items-center">
                <label className="me-2 text-muted small mb-0">Format</label>
                <select
                  className="form-select form-select-sm"
                  style={{ width: 'auto' }}
                  value={format}
                  onChange={(e) => setFormat(e.currentTarget.value)}
                >
                  <option value="csv">csv</option>
                  <option value="xlsx">xlsx</option>
                  <option value="parquet">parquet</option>
                </select>
              </div>
            </div>

            <section className="mt-4">
              <h5 className="fw-semibold">Annotations</h5>
              <hr className="mt-1" />
              <div className="d-flex flex-wrap gap-2">
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
                <button
                  className="btn-secondary-action"
                  onClick={() => {
                    if (currentScheme) getAnnotationsFile('all', format, 'train');
                  }}
                >
                  All annotations / schemes
                </button>
              </div>
            </section>

            <section className="mt-4">
              <h5 className="fw-semibold">Features</h5>
              <hr className="mt-1" />
              {availableFeatures.length === 0 ? (
                <div className="text-muted small">No features available.</div>
              ) : (
                <>
                  <div className="text-muted small mb-2">Click pills to select features</div>
                  <div className="model-pill-selection">
                    {availableFeatures.map((name) => (
                      <button
                        key={name}
                        className={cx('ms-0 model-pill', features.includes(name) && 'selected')}
                        onClick={() => toggleFeature(name)}
                      >
                        {name}
                      </button>
                    ))}
                  </div>
                </>
              )}
              <div className="d-flex flex-wrap gap-2 mt-2">
                <button
                  className="btn-secondary-action"
                  disabled={features.length === 0}
                  onClick={() => {
                    if (features.length > 0) getFeaturesFile(features, format);
                  }}
                >
                  Export selected features
                </button>
                {availableProjection && (
                  <button
                    className="btn-secondary-action"
                    onClick={() => getProjectionFile(format)}
                  >
                    Export current projection
                  </button>
                )}
              </div>
            </section>

            <section className="mt-4">
              <h5 className="fw-semibold">{isNer ? 'NER models' : 'BERT models'}</h5>
              <hr className="mt-1" />
              {availableModels.length === 0 ? (
                <div className="text-muted small">No models available for the current scheme.</div>
              ) : (
                <>
                  <div className="text-muted small mb-2">Select a model</div>
                  <ModelsPillDisplay
                    modelNames={availableModels}
                    currentModelName={model}
                    setCurrentModelName={setModel}
                  />
                </>
              )}

              {model && (
                <>
                  {!availablePredictionAll && (
                    <div className="alert alert-info mt-3 py-2 small mb-0" role="alert">
                      No prediction available on the complete dataset for this model. Run one from
                      the <Link to={`/projects/${projectName}/model/`}>Prediction tab</Link> first
                      to enable the export.
                    </div>
                  )}
                  <div className="d-flex flex-wrap gap-2 mt-3">
                    {availablePredictionAll && (
                      <button
                        className="btn-secondary-action"
                        disabled={predictionLoading !== null}
                        onClick={() => downloadPrediction('all')}
                      >
                        Export prediction complete dataset (+ imported)
                        {predictionLoading === 'all' && (
                          <PulseLoader color="white" size={6} className="ms-2" />
                        )}
                      </button>
                    )}
                    {availablePredictionTest && (
                      <button
                        className="btn-secondary-action"
                        disabled={predictionLoading !== null}
                        onClick={() => downloadPrediction('test')}
                      >
                        Export prediction testset
                        {predictionLoading === 'test' && (
                          <PulseLoader color="white" size={6} className="ms-2" />
                        )}
                      </button>
                    )}
                    {availablePredictionExternal && (
                      <button
                        className="btn-secondary-action"
                        disabled={predictionLoading !== null}
                        onClick={() => downloadPrediction('external')}
                      >
                        Export prediction external dataset
                        {predictionLoading === 'external' && (
                          <PulseLoader color="white" size={6} className="ms-2" />
                        )}
                      </button>
                    )}
                    <button className="btn-secondary-action" onClick={() => getModelFile(model)}>
                      Export fine-tuned model
                    </button>
                  </div>
                  {predictionLoading === 'all' && (
                    <div className="text-muted small mt-2">
                      Preparing file, this may take a while for large datasets...
                    </div>
                  )}
                </>
              )}
            </section>

            <section className="mt-4">
              <h5 className="fw-semibold">Raw dataset</h5>
              <hr className="mt-1" />
              <button className="btn-secondary-action" onClick={() => getRawDataFile()}>
                Export raw dataset in parquet
              </button>
            </section>

            <section className="mt-4">
              <h5 className="fw-semibold">Project summary</h5>
              <hr className="mt-1" />
              <div className="text-muted small mb-2">
                Lab-notebook style snapshot of the project: parameters, schemes, annotation counts,
                features, models, users. JSON is suitable for the CLI client; Markdown is meant to
                be read.
              </div>
              <div className="d-flex flex-wrap gap-2">
                <button
                  className="btn-secondary-action"
                  disabled={summaryLoading !== null}
                  onClick={() => downloadSummary('md')}
                >
                  Download Markdown notebook
                  {summaryLoading === 'md' && (
                    <PulseLoader color="white" size={6} className="ms-2" />
                  )}
                </button>
                <button
                  className="btn-secondary-action"
                  disabled={summaryLoading !== null}
                  onClick={() => downloadSummary('json')}
                >
                  Download JSON
                  {summaryLoading === 'json' && (
                    <PulseLoader color="white" size={6} className="ms-2" />
                  )}
                </button>
              </div>
            </section>
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
