{{/*
vllm.fullname 헬퍼를 정의합니다.
차트 이름과 릴리스 이름을 조합하여 리소스의 전체 이름을 생성합니다.
*/}}
{{- define "vllm.fullname" -}}
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

{{/*
vllm.chart 헬퍼를 정의합니다.
차트 이름과 버전 정보를 조합하여 "차트-버전" 형식의 문자열을 만듭니다.
*/}}
{{- define "vllm.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
vllm.labels 헬퍼를 정의합니다.
모든 리소스에 공통적으로 사용될 표준 레이블을 생성합니다.
*/}}
{{- define "vllm.labels" -}}
helm.sh/chart: {{ include "vllm.chart" . }}
{{ include "vllm.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
vllm.selectorLabels 헬퍼를 정의합니다.
Deployment, Service 등에서 리소스를 선택하기 위한 selector용 레이블을 생성합니다.
*/}}
{{- define "vllm.selectorLabels" -}}
app.kubernetes.io/name: {{ include "vllm.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}