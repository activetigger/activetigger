{{- define "activetigger.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "activetigger.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "activetigger.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "activetigger.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "activetigger.selectorLabels" -}}
app.kubernetes.io/name: {{ include "activetigger.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "activetigger.secretName" -}}
{{- if .Values.secrets.existingSecret }}
{{- .Values.secrets.existingSecret }}
{{- else }}
{{- printf "%s-secret" (include "activetigger.fullname" .) }}
{{- end }}
{{- end }}

{{- define "activetigger.apiServiceName" -}}
{{- printf "%s-api" (include "activetigger.fullname" .) }}
{{- end }}

{{- define "activetigger.frontendServiceName" -}}
{{- printf "%s-frontend" (include "activetigger.fullname" .) }}
{{- end }}

{{- define "activetigger.postgresqlServiceName" -}}
{{- printf "%s-postgresql" (include "activetigger.fullname" .) }}
{{- end }}

{{- define "activetigger.databaseUrl" -}}
{{- if .Values.postgresql.trustAuth }}
{{- printf "postgresql://%s@%s:%v/%s" .Values.postgresql.username (include "activetigger.postgresqlServiceName" .) .Values.postgresql.port .Values.postgresql.database }}
{{- else }}
{{- printf "postgresql://%s:%s@%s:%v/%s" .Values.postgresql.username .Values.secrets.postgresqlPassword (include "activetigger.postgresqlServiceName" .) .Values.postgresql.port .Values.postgresql.database }}
{{- end }}
{{- end }}
