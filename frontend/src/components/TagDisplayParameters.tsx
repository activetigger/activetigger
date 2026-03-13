import { FC } from 'react';
import { useAppContext } from '../core/useAppContext';

export const TagDisplayParameters: FC = () => {
  const {
    appContext: { displayConfig },
    setAppContext,
  } = useAppContext();
  return (
    <div className="d-flex flex-column">
      <label>
        <input
          type="checkbox"
          checked={displayConfig.displayHistory}
          onChange={(_) => {
            setAppContext((prev) => ({
              ...prev,
              displayConfig: {
                ...displayConfig,
                displayHistory: !displayConfig.displayHistory,
              },
            }));
          }}
        />
        Annotation history
      </label>
      <label>
        <input
          type="checkbox"
          checked={displayConfig.displayAnnotation}
          onChange={(_) => {
            setAppContext((prev) => ({
              ...prev,
              displayConfig: {
                ...displayConfig,
                displayAnnotation: !displayConfig.displayAnnotation,
              },
            }));
          }}
        />
        Existing annotation
      </label>

      <label>
        <input
          type="checkbox"
          checked={displayConfig.displayPrediction}
          onChange={(_) => {
            setAppContext((prev) => ({
              ...prev,
              displayConfig: {
                ...displayConfig,
                displayPrediction: !displayConfig.displayPrediction,
              },
            }));
          }}
        />
        Show Prediction
      </label>
      {displayConfig.displayPrediction && (
        <label>
          <input
            type="checkbox"
            checked={displayConfig.displayPredictionStat}
            onChange={(_) => {
              setAppContext((prev) => ({
                ...prev,
                displayConfig: {
                  ...displayConfig,
                  displayPredictionStat: !displayConfig.displayPredictionStat,
                },
              }));
            }}
          />
          Show Prediction Stats
        </label>
      )}

      {/* <label>
        <input
          type="checkbox"
          checked={displayConfig.displayContext}
          onChange={(_) => {
            setAppContext((prev) => ({
              ...prev,
              displayConfig: {
                ...displayConfig,
                displayContext: !displayConfig.displayContext,
              },
            }));
          }}
        />
        Contextual information
      </label> */}

      <label>
        <input
          type="checkbox"
          checked={displayConfig.displayElementHistory}
          onChange={(_) => {
            setAppContext((prev) => ({
              ...prev,
              displayConfig: {
                ...displayConfig,
                displayElementHistory: !displayConfig.displayElementHistory,
              },
            }));
          }}
        />
        Element history
      </label>

      <label>Tokens (approx. 4 c / token)</label>
      <input
        type="number"
        min="100"
        max="10000"
        value={displayConfig.numberOfTokens}
        onChange={(e) => {
          setAppContext((prev) => ({
            ...prev,
            displayConfig: {
              ...displayConfig,
              numberOfTokens: Number(e.target.value),
            },
          }));
        }}
        style={{ marginRight: '10px' }}
      />

      <label>Text frame size</label>
      <div className="horizontal">
        <span className="text-nowrap me-1">Height {displayConfig.textFrameHeight}%</span>
        <input
          type="range"
          min="25"
          max="90"
          value={displayConfig.textFrameHeight}
          onChange={(e) => {
            setAppContext((prev) => ({
              ...prev,
              displayConfig: {
                ...displayConfig,
                textFrameHeight: Number(e.target.value),
              },
            }));
          }}
          style={{ marginRight: '10px' }}
        />
      </div>
      <div className="horizontal">
        <span className="text-nowrap me-1">Width {displayConfig.textFrameWidth}%</span>
        <input
          type="range"
          min="25"
          max="100"
          value={displayConfig.textFrameWidth}
          onChange={(e) => {
            setAppContext((prev) => ({
              ...prev,
              displayConfig: {
                ...displayConfig,
                textFrameWidth: Number(e.target.value),
              },
            }));
          }}
          style={{ marginRight: '10px' }}
        />
      </div>

      <label>
        <input
          type="checkbox"
          checked={displayConfig.forceOneColumnLayout}
          onChange={(_) => {
            setAppContext((prev) => ({
              ...prev,
              displayConfig: {
                ...displayConfig,
                forceOneColumnLayout: !displayConfig.forceOneColumnLayout,
              },
            }));
          }}
        />
        Force bottom labels
      </label>

      <label>Highlight words in the text</label>
      <textarea
        placeholder="Line break to separate"
        // onChange={(e) => setWordsToHighlight(e.target.value)}
        value={displayConfig.highlightText}
        onChange={(e) => {
          setAppContext((prev) => ({
            ...prev,
            displayConfig: {
              ...displayConfig,
              highlightText: String(e.target.value),
            },
          }));
        }}
      />
    </div>
  );
};
