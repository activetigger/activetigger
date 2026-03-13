import { useContext } from 'react';
import { AppContext, AppContextValue } from './context';

export function useAppContext() {
  return useContext(AppContext);
}
export const DEFAULT_CONTEXT: AppContextValue = {
  currentScheme: undefined,
  notifications: [],
  displayConfig: {
    interfaceType: 'default',
    displayAnnotation: true,
    displayContext: true,
    displayPrediction: true,
    displayPredictionStat: true,
    displayHistory: true,
    displayElementHistory: false,
    textFrameHeight: 50,
    textFrameWidth: 40,
    highlightText: '',
    numberOfTokens: 512,
    forceOneColumnLayout: false,
    displayFormat: 'cards',
  },
  selectionConfig: {
    mode: 'fixed',
    sample: 'all',
    frameSelection: false,
    frame: [],
  },
  generateConfig: { n_batch: 1, selectionMode: 'all' },
  history: [],
  selectionHistory: {},
  freqRefreshQuickModel: 0,
  phase: 'train',
  isComputing: false,
  developmentMode: false,
};
