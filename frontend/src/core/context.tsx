import { FC, PropsWithChildren, createContext, useCallback, useEffect, useState } from 'react';

import {
  ActiveModel,
  DisplayConfig,
  ElementHistoryPoint,
  GenerateConfig,
  NotificationType,
  ProjectStateModel,
  ProjectionOutModel,
  SelectionConfig,
} from '../types';
import { DEFAULT_CONTEXT } from './useAppContext';

// Context content
export type AppContextValue = {
  notifications: NotificationType[]; // manage notification
  selectionConfig: SelectionConfig; // selection for the next element
  generateConfig: GenerateConfig;
  displayConfig: DisplayConfig; // config for the visual
  currentProject?: ProjectStateModel | null; // current project selected
  currentScheme?: string; // scheme selected to annotate
  currentProjection?: ProjectionOutModel;
  labelColorMapping?: { [key: string]: string };
  activeModel?: ActiveModel | null;
  freqRefreshQuickModel: number; // freq to refresh active learning model
  history: ElementHistoryPoint[]; // element annotated
  selectionHistory: Record<string, string>; // history of the selection
  reFetchCurrentProject?: () => void; // update the state of the project
  phase: string;
  isComputing: boolean;
  developmentMode: boolean;
};

export const CONTEXT_LOCAL_STORAGE_KEY = 'activeTigger.context';

const storedContext = localStorage.getItem(CONTEXT_LOCAL_STORAGE_KEY);

// type of the context
export type AppContextType = {
  appContext: AppContextValue;
  setAppContext: React.Dispatch<React.SetStateAction<AppContextValue>>;
  resetContext: () => void;
};

export const AppContext = createContext<AppContextType>(null as unknown as AppContextType);

const _useAppContext = () => {
  const [appContext, setAppContext] = useState<AppContextValue>(
    storedContext ? (JSON.parse(storedContext) as AppContextValue) : DEFAULT_CONTEXT,
  );

  //store context in localstorage
  useEffect(() => {
    localStorage.setItem(CONTEXT_LOCAL_STORAGE_KEY, JSON.stringify(appContext));
  }, [appContext]);

  // Function to reset the context
  const resetContext = useCallback(() => {
    const newContext = DEFAULT_CONTEXT;
    // keep the interface type
    newContext.displayConfig.interfaceType = appContext.displayConfig.interfaceType;
    setAppContext(newContext);
  }, [appContext.displayConfig.interfaceType, setAppContext]);

  return {
    appContext,
    setAppContext,
    resetContext,
  };
};

export const AppContextProvider: FC<PropsWithChildren> = ({ children }) => {
  const context = _useAppContext();

  return <AppContext.Provider value={context}>{children}</AppContext.Provider>;
};
