import chroma from 'chroma-js';
import cx from 'classnames';
import { FC, useCallback, useEffect, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { FaCloudDownloadAlt, FaPlusCircle } from 'react-icons/fa';
import { FaGear } from 'react-icons/fa6';
import { RiFileTransferLine } from 'react-icons/ri';
import { SlSizeFullscreen } from 'react-icons/sl';
import { useParams } from 'react-router-dom';
import { DisplayTableTopics, Row } from '../components/DisplayTableTopics';
import { BertopicForm } from '../components/forms/BertopicForm';
import { ModelsPillDisplay } from '../components/ModelsPillDisplay';
import { BertopicVizSigma } from '../components/ProjectionVizSigma/BertopicVizSigma';
import {
  useDeleteBertopic,
  useDownloadBertopicClusters,
  useDownloadBertopicEmbeddings,
  useDownloadBertopicReport,
  useDownloadBertopicTopics,
  useExportTopicsToScheme,
  useGetBertopicProjection,
  useGetBertopicTopics,
  useGetElementById,
} from '../core/api';
import { useAppContext } from '../core/useAppContext';
import { sortDatesAsStrings } from '../core/utils';

export const BertopicPage: FC = () => {
  const { projectName } = useParams();
  const {
    appContext: { currentProject, isComputing },
  } = useAppContext();
  const deleteBertopic = useDeleteBertopic(projectName || null);
  const exportTopicsToScheme = useExportTopicsToScheme(projectName || null);
  const { downloadBertopicTopics } = useDownloadBertopicTopics(projectName || null);
  const { downloadBertopicClusters } = useDownloadBertopicClusters(projectName || null);
  const { downloadBertopicReport } = useDownloadBertopicReport(projectName || null);
  const { downloadBertopicEmbeddings } = useDownloadBertopicEmbeddings(projectName || null);
  const availableBertopic = currentProject ? currentProject.bertopic.available : [];
  const [currentBertopic, setCurrentBertopic] = useState<string | null>(null);
  const { getElementById } = useGetElementById();

  const { topics, parameters, reFetchTopics } = useGetBertopicTopics(
    projectName || null,
    currentBertopic,
  );
  const { projection, reFetchProjection } = useGetBertopicProjection(
    projectName || null,
    currentBertopic,
  );
  const currentTraining = currentProject ? Object.entries(currentProject.bertopic.training) : null;
  const availableModels = currentProject ? currentProject.bertopic.models : [];
  useEffect(() => {
    reFetchTopics();
    reFetchProjection();

    //Reset states
    setClusterHighlight(undefined);
    setCurrentText(null);
  }, [currentBertopic, reFetchTopics, reFetchProjection]);

  // Action if clicked
  const [currentText, setCurrentText] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | undefined>(undefined);
  const setSelectedIdAfterClick = useCallback(
    // For the moment, only get something if it is the trainset
    (id?: string) => {
      if (id) {
        setSelectedId(id);
        getElementById(id, 'train').then((res) => {
          setCurrentText(String(id) + ': ' + res?.text || null);
        });
      } else {
        setSelectedId(undefined);
        setCurrentText(null);
      }
    },
    [getElementById],
  );
  // Action if double click
  const [clusterHighlight, setClusterHighlight] = useState<string | undefined>(undefined);
  const setClusterHighlightAfterDoubleClick = useCallback(
    (id?: string) => {
      if (id && projection) {
        const selected_node = projection.nodes.find((o) => o.node_id === id);
        if (selected_node) {
          setClusterHighlight(selected_node.cluster_id.toString());
          return;
        }
      }
      setClusterHighlight(undefined);
    },
    [projection],
  );

  const uniqueLabels = projection
    ? (Object.values(projection.cluster_id_label_mapper) as string[])
    : [];
  const colormap = chroma.scale('Paired').colors(uniqueLabels.length);
  const clusterIdColorMapping = uniqueLabels.reduce<Record<string, string>>(
    (acc, label, index: number) => {
      acc[label as string] = colormap[index];
      return acc;
    },
    {},
  );

  const exportBertopicAsAnnotation = async (topicModelName: string | null) => {
    if (topicModelName) {
      exportTopicsToScheme(topicModelName);
    }
  };

  const [showComputeNewBertopic, setShowComputeNewBertopic] = useState(false);
  const [showParameters, setShowParameters] = useState<boolean>(false);
  const [figSize, setFigSize] = useState<number>(40);
  const toogleFigSize = () => {
    if (figSize === 40) {
      setFigSize(80);
    } else {
      setFigSize(40);
    }
  };

  return (
    // <ProjectPageLayout projectName={projectName} currentAction="explore">
    <div className="row">
      <div className="col-12">
        <div className="d-flex my-2" style={{ zIndex: 100 }}>
          <ModelsPillDisplay
            modelNames={Object.values(availableBertopic)
              .sort((bertopicA, bertopicB) =>
                sortDatesAsStrings(bertopicA?.time, bertopicB?.time, true),
              )
              .map((model) => (model && model.name ? model.name : ''))}
            currentModelName={currentBertopic}
            setCurrentModelName={setCurrentBertopic}
            deleteModelFunction={deleteBertopic}
          >
            <button
              onClick={() => setShowComputeNewBertopic(true)}
              className={cx('model-pill ', isComputing ? 'disabled' : '')}
              id="create-new"
            >
              <FaPlusCircle size={20} /> Compute new BERTopic
            </button>
          </ModelsPillDisplay>
        </div>
        {currentTraining && currentTraining?.length > 0 && (
          <div className="alert alert-info m-2">
            Current computation
            <ul>
              {Object.values(currentTraining).map(([k, v]) => (
                <li key={k}>
                  User {k} : {(v as unknown as { progress: string })?.progress}
                </li>
              ))}
            </ul>
          </div>
        )}
        <Modal
          show={showComputeNewBertopic}
          onHide={() => setShowComputeNewBertopic(false)}
          size="xl"
          id="viz-modal"
        >
          <Modal.Header closeButton>
            <Modal.Title>
              Compute Bertopic on the train dataset to identify the main topics
            </Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <BertopicForm
              projectSlug={projectName || null}
              availableModels={availableModels}
              isComputing={isComputing}
              setStatusDisplay={setShowComputeNewBertopic}
            />
          </Modal.Body>
        </Modal>

        <hr className="my-4" />

        {currentBertopic && (
          <>
            <Modal
              show={showParameters}
              id="parameters-modal"
              onHide={() => setShowParameters(false)}
            >
              <Modal.Header closeButton>
                <Modal.Title>Parameters of the current visualisation</Modal.Title>
              </Modal.Header>
              <Modal.Body>
                <div
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    justifyContent: 'space-evenly',
                  }}
                >
                  <div style={{ width: '400px' }}>
                    <h6 className="subsection">General parameters</h6>
                    <table className="table-statistics">
                      <tbody>
                        <tr>
                          <td>Language</td>
                          <td>{parameters?.bertopic_params.language}</td>
                        </tr>
                        <tr>
                          <td>Embedding model</td>
                          <td>{parameters?.bertopic_params.embedding_model}</td>
                        </tr>
                        <tr>
                          <td>Number of keywords</td>
                          <td>{parameters?.bertopic_params.top_n_words}</td>
                        </tr>
                        <tr>
                          <td>Keywords n-grams</td>
                          <td>{parameters?.bertopic_params.n_gram_range}</td>
                        </tr>
                        <tr>
                          <td>Outlier reduction</td>
                          <td>
                            {parameters?.bertopic_params.outlier_reduction ? 'True' : 'False'}
                          </td>
                        </tr>
                        <tr>
                          <td>Minimum number of characters of texts</td>
                          <td>{parameters?.bertopic_params.filter_text_length}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <div>
                    <div style={{ width: '400px' }}>
                      <h6 className="subsection">Dimension reduction parameters (UMAP)</h6>
                      <table className="table-statistics">
                        <tbody>
                          <tr>
                            <td>Number of neighbors</td>
                            <td>{parameters?.bertopic_params.umap_n_neighbors}</td>
                          </tr>
                          <tr>
                            <td>Number of components</td>
                            <td>{parameters?.bertopic_params.umap_n_components}</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                    <div style={{ width: '400px' }}>
                      <h6 className="subsection">Clustering parameters (HDBSCAN)</h6>
                      <table className="table-statistics">
                        <tbody>
                          <tr>
                            <td>Clusters' mininum size</td>
                            <td>{parameters?.bertopic_params.hdbscan_min_cluster_size}</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </Modal.Body>
            </Modal>
          </>
        )}
        {projection && (
          <>
            <div style={{ height: `${figSize}vh`, width: '100%' }}>
              <BertopicVizSigma
                className={`col-12 border h-100`}
                nodes={projection.nodes}
                selectedId={selectedId}
                setSelectedIdAfterClick={setSelectedIdAfterClick}
                clusterIdColorMapping={clusterIdColorMapping}
                clusterHighlight={clusterHighlight}
                setClusterHighlightAfterDoubleClick={setClusterHighlightAfterDoubleClick}
              />
            </div>
            {currentText && (
              <div
                className="col-12"
                style={{
                  height: '200px',
                  overflow: 'hidden',
                  overflowY: 'scroll',
                  backgroundColor: '#f5f5f5',
                }}
              >
                {currentText}
              </div>
            )}
          </>
        )}
        {topics && (
          <>
            <div
              style={{ margin: '10px 0px' }}
              className="d-flex gap-2 align-items-center flex-wrap"
            >
              <button className="btn" onClick={() => toogleFigSize()}>
                <SlSizeFullscreen />
              </button>
              <button
                className="btn-secondary-action"
                style={{ whiteSpace: 'nowrap' }}
                onClick={() => setShowParameters(true)}
              >
                <FaGear size={18} />
                Parameters
              </button>
              <button
                className="btn-secondary-action"
                style={{ whiteSpace: 'nowrap' }}
                onClick={() => exportBertopicAsAnnotation(currentBertopic)}
              >
                Convert to scheme <RiFileTransferLine size={20} />
              </button>
              <button
                className="btn-secondary-action"
                style={{ whiteSpace: 'nowrap' }}
                id="download-topics"
                onClick={() => (currentBertopic ? downloadBertopicTopics(currentBertopic) : null)}
              >
                Export topics <FaCloudDownloadAlt size={20} />
              </button>
              <button
                className="btn-secondary-action"
                style={{ whiteSpace: 'nowrap' }}
                id="download-clusters"
                onClick={() => (currentBertopic ? downloadBertopicClusters(currentBertopic) : null)}
              >
                Export topic per text <FaCloudDownloadAlt size={20} />
              </button>
              <button
                className="btn-secondary-action"
                style={{ whiteSpace: 'nowrap' }}
                id="download-clusters"
                onClick={() => (currentBertopic ? downloadBertopicReport(currentBertopic) : null)}
              >
                Topic model report <FaCloudDownloadAlt size={20} />
              </button>
              <button
                className="btn-secondary-action"
                style={{ whiteSpace: 'nowrap' }}
                id="download-embeddings"
                onClick={() =>
                  currentBertopic ? downloadBertopicEmbeddings(currentBertopic) : null
                }
              >
                Embeddings <FaCloudDownloadAlt size={20} />
              </button>
            </div>
            <div
              style={{
                height: `${80 * (1 + topics.length)}px`,
                margin: '15px 0px',
                overflowX: 'auto',
              }}
            >
              <DisplayTableTopics data={(topics as unknown as Row[]) || []} />
            </div>
          </>
        )}
      </div>
    </div>
    //  </ProjectPageLayout>
  );
};
