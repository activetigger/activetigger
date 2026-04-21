import { ReactNode } from 'react';

import { components } from './generated/openapi';

/**
 * API data types
 * this file rename API data types from the generated ones to make our code cleaner
 * Types are generated in a components['schemas'] map which is hideous
 * we extract what we use little by little when needed
 */

export type UserModel = components['schemas']['UserModel'];

export type ProjectModel = components['schemas']['ProjectModel'];
export type ProjectBaseModel = components['schemas']['ProjectBaseModel'];
export type ProjectStateModel = components['schemas']['ProjectStateModel'];
export type ElementOutModel = components['schemas']['ElementOutModel'];

export type AvailableProjectsModel = {
  created_by: string;
  created_at: string;
  parameters: ProjectModel;
  size: number;
};
export type LoginParams = components['schemas']['Body_login_for_access_token_token_post'];

/**
 * Notifications
 */
export interface NotificationData {
  title?: ReactNode;
  message: ReactNode;
  type: 'success' | 'info' | 'warning' | 'error';
}

export type NotificationType = NotificationData & { id: number; createdAt: Date };

export type SchemeModel = components['schemas']['SchemeModel'];

export type FeatureModel = components['schemas']['FeatureModel'];

export type RequestNextModel = components['schemas']['NextInModel'];

export type AnnotationModel = components['schemas']['AnnotationModel'];
export type TableAnnotationsModel = components['schemas']['TableAnnotationsModel'];

export type QuickModelInModel = components['schemas']['QuickModelInModel'];

export type LMParametersModel = components['schemas']['LMParametersModel'];

export type EvalSetDataModel = components['schemas']['EvalSetDataModel'];

export type AnnotationsDataModel = components['schemas']['AnnotationsDataModel'];

export type ProjectionParametersModel = components['schemas']['ProjectionParametersModel'];

export type GenerationModelApi = components['schemas']['GenerationModelApi'];

export type ProjectUpdateModel = components['schemas']['ProjectUpdateModel'];

export type TextDatasetModel = components['schemas']['TextDatasetModel'];

export type MLStatisticsModel = components['schemas']['MLStatisticsModel'];

export type ModelInformationsModel = components['schemas']['ModelInformationsModel'];

export type ModelScoresModel = components['schemas']['ModelScoresModel'];

export type ComputeBertopicModel = components['schemas']['ComputeBertopicModel'];

export type ModelDescriptionModel = components['schemas']['ModelDescriptionModel'];

export type FeaturesProjectStateModel = components['schemas']['FeaturesProjectStateModel'];

export type ProjectionOutModel = components['schemas']['ProjectionOutModel'];

export type FeatureDescriptionModelOut = components['schemas']['FeatureDescriptionModelOut'];

export type ActiveModel = components['schemas']['ActiveModel'];

export type ElementHistoryPoint = Omit<components['schemas']['AnnotationModel'], 'user'> & {
  element_text: string;
  skip?: boolean;
};

export interface FeatureDfmParameters {
  dfm_tfidf: string;
  ngrams: number;
  dfm_ngrams: number;
  dfm_min_term_freq: number;
  dfm_max_term_freq: number;
  dfm_norm: string;
  dfm_log: string;
}

export interface FeatureRegexParameters {
  value: string;
  regex_count?: boolean;
}

export interface FeatureDatasetParameters {
  dataset_type: string;
  dataset_col: string;
}

export interface FeatureFasttextParameters {
  model?: string;
}

export interface FeatureSbertParameters {
  model?: string;
  max_length_tokens?: number;
  batch_size?: number;
}

export interface FeatureModelExtended {
  name: string;
  type: string;
  parameters:
    | null
    | FeatureSbertParameters
    | FeatureDfmParameters
    | FeatureRegexParameters
    | FeatureDatasetParameters
    | FeatureFasttextParameters;
}

export interface SelectionConfig {
  mode: string;
  sample: string;
  labels?: string[];
  label_prob?: string; // label to use for maxprob selection
  frame?: number[];
  frameSelection?: boolean; // true/false to use frame to select
  filter?: string;
  users?: string[];
}

export interface GenerateConfig {
  api?: string;
  endpoint?: string;
  token?: string;
  prompt?: string;
  promptId?: string;
  n_batch?: number;
  selectionMode?: string;
  dataset?: string;
  selectedModel?: GenModel & { api: string };
}

export interface DisplayConfig {
  interfaceType: string;
  displayAnnotation: boolean;
  displayPrediction: boolean;
  displayPredictionStat: boolean;
  displayContext: boolean;
  displayHistory: boolean;
  displayElementHistory: boolean;
  numberOfTokens: number;
  textFrameHeight: number;
  textFrameWidth: number;
  highlightText: string;
  labelsOrder?: string[];
  forceOneColumnLayout?: boolean;
  forceCompleteLabel?: boolean;
  displayFormat?: 'table' | 'cards';
}

export interface newBertModel {
  name?: string;
  base: string;
  parameters: LMParametersModel;
  dichotomize?: string;
  class_balance?: boolean;
  loss?: string;
  class_min_freq?: number;
  test_size?: number;
  exclude_labels: string[];
  max_length?: number;
  auto_max_length: boolean;
}

export interface EvalSetModel {
  col_id: string;
  cols_text: string[];
  col_label?: string | null;
  scheme?: string | null;
  n_eval: number;
}

export type SupportedAPI = 'Ollama' | 'OpenAI' | 'HuggingFace';

export type GenModelAPI = { models: GenModel[] } & (
  | {
      name: 'Ollama';
      endpoint: string;
    }
  | {
      name: 'OpenAI';
      credentials: string;
    }
  | {
      name: 'HuggingFace';
      endpoint: string;
      credentials: string;
    }
);

export interface GenModel {
  id: number;
  slug: string;
  name: string;
  endpoint?: string;
  credentials?: string;
}
