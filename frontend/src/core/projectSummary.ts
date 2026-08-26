import { saveAs } from 'file-saver';

/**
 * Project summary payload returned by GET /export/summary.
 * Kept as an open shape — the canonical schema lives in the backend
 * (Project.export_summary in api/activetigger/project.py) and the CLI
 * client consumes the same JSON.
 */
export type ProjectSummary = Record<string, unknown> & {
  format_version?: string;
  generated_at?: string;
  project?: Record<string, unknown> & {
    slug?: string;
    name?: string;
    kind?: string;
    language?: string;
    created_at?: string | null;
    created_by?: string | null;
    modified_at?: string | null;
    col_id?: string;
    cols_text?: string[];
    cols_context?: string[];
    n_total?: number | null;
    n_train?: number;
    n_test?: number;
    n_valid?: number;
    parameters?: Record<string, unknown>;
  };
  users?: { username: string; role: string }[];
  schemes?: Array<{
    name: string;
    kind: string;
    labels: string[];
    created_at?: string | null;
    modified_at?: string | null;
    created_by?: string | null;
    n_annotations_total: number;
    n_annotations_per_dataset: Record<string, number>;
    n_distinct_elements_annotated: number;
    annotations_per_label: Record<string, number>;
    annotations_per_user: Record<string, number>;
  }>;
  features?: Array<{
    name: string;
    kind?: string;
    user?: string;
    time?: string | null;
    parameters?: Record<string, unknown>;
  }>;
  language_models?: Array<{
    name: string;
    scheme: string;
    time?: string | null;
    predicted?: boolean;
    predicted_all?: boolean;
    predicted_external?: boolean;
    tested?: boolean;
    exclude_labels?: string[];
  }>;
  quick_models?: Array<{
    name: string;
    scheme: string;
    kind: string;
    time?: string | null;
    parameters?: Record<string, unknown>;
  }>;
};

const fmtDate = (iso?: string | null): string => (iso ? iso.replace('T', ' ').slice(0, 19) : '—');

const escapeMd = (s: string): string => s.replace(/\|/g, '\\|');

const kvTable = (entries: [string, string | number | null | undefined][]): string => {
  const rows = entries
    .filter(([, v]) => v !== undefined && v !== null && v !== '')
    .map(([k, v]) => `| ${escapeMd(k)} | ${escapeMd(String(v))} |`);
  if (rows.length === 0) return '';
  return ['| Key | Value |', '|---|---|', ...rows].join('\n');
};

const countsTable = (header: [string, string], obj: Record<string, number> | undefined): string => {
  if (!obj || Object.keys(obj).length === 0) return '_none_';
  const rows = Object.entries(obj)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `| ${escapeMd(k)} | ${v} |`);
  return [`| ${header[0]} | ${header[1]} |`, '|---|---:|', ...rows].join('\n');
};

/**
 * Render a Markdown "laboratory notebook" view of the project summary.
 */
export const renderSummaryMd = (s: ProjectSummary): string => {
  const lines: string[] = [];
  const p = s.project ?? {};

  lines.push(`# Project notebook — ${p.name ?? p.slug ?? 'untitled'}`);
  lines.push('');
  lines.push(`_Generated at ${fmtDate(s.generated_at)} — format ${s.format_version ?? '?'}_`);
  lines.push('');

  // Project header
  lines.push('## Project');
  lines.push('');
  lines.push(
    kvTable([
      ['Slug', p.slug ?? null],
      ['Name', p.name ?? null],
      ['Kind', p.kind ?? null],
      ['Language', p.language ?? null],
      ['Created at', fmtDate(p.created_at)],
      ['Created by', p.created_by ?? null],
      ['Last modified', fmtDate(p.modified_at)],
      ['ID column', p.col_id ?? null],
      ['Text columns', (p.cols_text ?? []).join(', ') || null],
      ['Context columns', (p.cols_context ?? []).join(', ') || null],
      ['Total rows', p.n_total ?? null],
      ['Train rows', p.n_train ?? null],
      ['Test rows', p.n_test ?? null],
      ['Validation rows', p.n_valid ?? null],
    ]),
  );
  lines.push('');

  // Users
  lines.push('## Users');
  lines.push('');
  if (!s.users || s.users.length === 0) {
    lines.push('_none_');
  } else {
    lines.push('| Username | Role |');
    lines.push('|---|---|');
    for (const u of s.users) lines.push(`| ${escapeMd(u.username)} | ${escapeMd(u.role)} |`);
  }
  lines.push('');

  // Schemes
  lines.push('## Schemes');
  lines.push('');
  if (!s.schemes || s.schemes.length === 0) {
    lines.push('_none_');
    lines.push('');
  } else {
    for (const sc of s.schemes) {
      lines.push(`### ${sc.name}`);
      lines.push('');
      lines.push(
        kvTable([
          ['Kind', sc.kind],
          ['Labels', sc.labels.join(', ')],
          ['Created at', fmtDate(sc.created_at)],
          ['Created by', sc.created_by ?? null],
          ['Last modified', fmtDate(sc.modified_at)],
          ['Annotations total', sc.n_annotations_total],
          ['Annotations train', sc.n_annotations_per_dataset?.train ?? 0],
          ['Annotations test', sc.n_annotations_per_dataset?.test ?? 0],
          ['Annotations valid', sc.n_annotations_per_dataset?.valid ?? 0],
          ['Distinct elements annotated', sc.n_distinct_elements_annotated],
        ]),
      );
      lines.push('');
      lines.push('**Annotations per label**');
      lines.push('');
      lines.push(countsTable(['Label', 'Count'], sc.annotations_per_label));
      lines.push('');
      lines.push('**Annotations per user**');
      lines.push('');
      lines.push(countsTable(['User', 'Count'], sc.annotations_per_user));
      lines.push('');
    }
  }

  // Features
  lines.push('## Features');
  lines.push('');
  if (!s.features || s.features.length === 0) {
    lines.push('_none_');
  } else {
    lines.push('| Name | Kind | User | Time |');
    lines.push('|---|---|---|---|');
    for (const f of s.features) {
      lines.push(
        `| ${escapeMd(f.name)} | ${escapeMd(f.kind ?? '')} | ${escapeMd(f.user ?? '')} | ${fmtDate(
          f.time,
        )} |`,
      );
    }
  }
  lines.push('');

  // Language models
  lines.push('## Language models (BERT)');
  lines.push('');
  if (!s.language_models || s.language_models.length === 0) {
    lines.push('_none_');
  } else {
    lines.push('| Name | Scheme | Time | Predicted (all) | Tested | Predicted (external) |');
    lines.push('|---|---|---|:-:|:-:|:-:|');
    for (const m of s.language_models) {
      lines.push(
        `| ${escapeMd(m.name)} | ${escapeMd(m.scheme)} | ${fmtDate(m.time)} | ${
          m.predicted_all ? '✓' : ''
        } | ${m.tested ? '✓' : ''} | ${m.predicted_external ? '✓' : ''} |`,
      );
    }
  }
  lines.push('');

  // Quick models
  lines.push('## Quick models');
  lines.push('');
  if (!s.quick_models || s.quick_models.length === 0) {
    lines.push('_none_');
  } else {
    lines.push('| Name | Scheme | Kind | Time |');
    lines.push('|---|---|---|---|');
    for (const m of s.quick_models) {
      lines.push(
        `| ${escapeMd(m.name)} | ${escapeMd(m.scheme)} | ${escapeMd(m.kind)} | ${fmtDate(m.time)} |`,
      );
    }
  }
  lines.push('');

  // Raw parameters (collapsed)
  if (p.parameters) {
    lines.push('## Raw project parameters');
    lines.push('');
    lines.push('```json');
    lines.push(JSON.stringify(p.parameters, null, 2));
    lines.push('```');
    lines.push('');
  }

  return lines.join('\n');
};

const todayStamp = (): string => new Date().toISOString().slice(0, 10);

export const downloadSummaryJson = (summary: ProjectSummary, slug: string): void => {
  const blob = new Blob([JSON.stringify(summary, null, 2)], {
    type: 'application/json;charset=utf-8',
  });
  saveAs(blob, `summary_${slug}_${todayStamp()}.json`);
};

export const downloadSummaryMd = (summary: ProjectSummary, slug: string): void => {
  const blob = new Blob([renderSummaryMd(summary)], { type: 'text/markdown;charset=utf-8' });
  saveAs(blob, `summary_${slug}_${todayStamp()}.md`);
};
