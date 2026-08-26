# ActiveTigger Helm Charts

This folder contains a first Helm chart for deploying ActiveTigger on Kubernetes, with the current prototype tested on the ENSAE/CREST Onyxia instance.


## Chart layout

The main chart is in `charts/activetigger/`.

It deploys:

- a FastAPI backend API;
- a React frontend served through nginx;
- an optional PostgreSQL database;
- PersistentVolumeClaims for API data and PostgreSQL data;
- a Kubernetes Ingress for external access;
- a Kubernetes Secret for application/database credentials.

Main files:

- `Chart.yaml`: chart metadata;
- `values.yaml`: default configuration;
- `templates/api-deployment.yaml`: API workload;
- `templates/frontend-deployment.yaml`: frontend workload;
- `templates/postgresql-deployment.yaml`: embedded PostgreSQL workload;
- `templates/ingress.yaml`: public routing;
- `templates/secret.yaml`: development Secret generation, disabled when using an external Secret.

## Current prototype deployment

The current prototype was deployed on an ENSAE Kubernetes namespace with one GPU and exposed through:

```text
https://activetigger.lab.groupe-genes.fr/
```

The chart currently defaults to this ENSAE test ingress:

```yaml
ingress:
  enabled: true
  className: nginx
  host: activetigger.lab.groupe-genes.fr
```

These values should be changed before using the chart in another namespace, domain, or Onyxia instance.

## Docker images

The prototype currently uses images published on the dedicated Docker Hub account for the project:

```text
activetigger/activetigger-api
activetigger/activetigger-frontend
```

The same Dockerfiles serve both the Docker Compose stack and Kubernetes:

- `docker/api/Dockerfile`: API image. The CPU/GPU choice is made at runtime by `entrypoint.sh` (`CPU_ONLY`/`GPU` env vars), so a single image and tag serve both deployments. Compose bind-mounts the live checkout over the baked-in code and starts the container as root to chown its named volumes; Kubernetes runs the baked-in code directly as uid 1000 via the pod securityContext.
- `docker/frontend/Dockerfile`: frontend image used by the chart (nginx serving the built React app);
- `docker/frontend/nginx.prototype.conf.template`: nginx template used by the frontend image to route `/api` to the API service.

Both images are built from the repository root (a root `.dockerignore` keeps local data and caches out). When building on Apple Silicon, target `linux/amd64` explicitly — the cluster nodes are amd64:

```bash
docker buildx build --platform linux/amd64 -f docker/api/Dockerfile \
  -t activetigger/activetigger-api:prototype --push .
docker buildx build --platform linux/amd64 -f docker/frontend/Dockerfile \
  -t activetigger/activetigger-frontend:prototype --push .
```

The chart defaults to these repositories and the `prototype` tag in `charts/activetigger/values.yaml`.

## Install examples

Set the namespace before running Helm. In Onyxia, the namespace is available from the Kubernetes settings of your account. Configure `kubectl` locally with the cluster URL, OIDC credentials, context, and namespace, following the instruction "Connect to the Kubernetes cluster".


Reuse the namespace variable in Helm commands:

```bash
export NAMESPACE=<onyxia-namespace>
```

Validate the chart locally:

```bash
helm lint charts/activetigger
helm template activetigger charts/activetigger
```

Install or upgrade with the default CPU values:

```bash
helm upgrade --install activetigger ./charts/activetigger \
  --namespace "$NAMESPACE" \
  --server-side=false \
  --timeout 25m
```

Deploy the GPU prototype (same image, GPU behavior enabled through values):

```bash
helm upgrade --install activetigger ./charts/activetigger \
  --namespace "$NAMESPACE" \
  --server-side=false \
  --timeout 25m \
  --set api.gpu.enabled=true \
  --set api.env.CPU_ONLY=false \
  --set api.env.N_WORKERS_GPU=1 \
  --set api.env.HF_HOME=/data/models/huggingface \
  --set api.env.TRANSFORMERS_CACHE=/data/models/huggingface/transformers \
  --set api.env.SENTENCE_TRANSFORMERS_HOME=/data/models/sentence-transformers
```

On the ENSAE cluster, `--server-side=false` was needed because server-side apply requests for some workloads were blocked by the gateway/WAF.

The HuggingFace/SentenceTransformers cache variables are not required for startup, but they are recommended. They store downloaded embedding models under `/data/models`, which is backed by the API PVC, so models do not need to be downloaded again after every pod restart.

## Secret management with Onyxia Vault

### Before catalog integration: manual Helm deployment

For the current manual Helm deployment, only the initial `ROOT_PASSWORD` is managed through Vault.

The current flow is:

```text
Vault -> vault CLI -> Kubernetes Secret -> API env var
```

Install the Vault CLI if needed: <https://developer.hashicorp.com/vault/install>.

Set up the Vault CLI:

```bash
export VAULT_ADDR=<vault-url>
export VAULT_TOKEN=<vault-token>
```

In your Onyxia account, in **My Secrets**, create a KV secret with name `<secret-name>`, then add a variable `ROOT_PASSWORD` with the value `<root-password>`.

Read the secret from Vault:

```bash
vault kv get onyxia-kv/<user-id>/<secret-name>
export ROOT_PASSWORD="$(vault kv get -field=ROOT_PASSWORD onyxia-kv/<user-id>/<secret-name>)"
```

Create the Kubernetes Secret before installing the chart:

```bash
kubectl create secret generic activetigger-secret \
  -n "$NAMESPACE" \
  --from-literal=root-password="$ROOT_PASSWORD"
```

Important: `kubectl create secret` fails if the Secret already exists. For a fresh deployment, this is expected and safe. For an update or redeploy on ENSAE, avoid `kubectl apply` for this Secret: patch/apply requests can be blocked by the gateway/WAF with an HTML “Web Page Blocked” response. Use delete + create instead:

```bash
kubectl delete secret activetigger-secret -n "$NAMESPACE"

kubectl create secret generic activetigger-secret \
  -n "$NAMESPACE" \
  --from-literal=root-password="$ROOT_PASSWORD"
```

Deploy with an existing Kubernetes Secret:

```bash
helm upgrade --install activetigger ./charts/activetigger \
  --namespace "$NAMESPACE" \
  --server-side=false \
  --timeout 25m \
  --set secrets.create=false \
  --set secrets.existingSecret=activetigger-secret
```


When `secrets.create=false`, the chart reads `ROOT_PASSWORD` from this Secret. 

Do not pass real secret values with `helm --set`; they can leak into shell history and Helm release metadata. Keep real values in Vault and only pass the Kubernetes Secret name to Helm.

### After catalog integration: native Onyxia service launch

Once ActiveTigger is added to the Onyxia service catalog, the target flow is:

```text
Onyxia launcher -> chart values -> VAULT_* env vars -> entrypoint reads Vault -> ROOT_PASSWORD env var -> API startup
```

Onyxia can inject Vault connection values into the chart/app, such as:

```text
VAULT_ADDR
VAULT_TOKEN
VAULT_MOUNT
VAULT_TOP_DIR
VAULT_RELATIVE_PATH
```

The chart should then pass these variables to the API pod. A small ActiveTigger entrypoint wrapper can read the Vault KV v2 secret, export `ROOT_PASSWORD`, then execute the normal API entrypoint.

## Useful kubectl commands

Check deployed resources:

```bash
kubectl get pods,svc,ingress,pvc -n "$NAMESPACE"
```

Follow the API rollout and inspect pod events:

```bash
kubectl rollout status deployment/activetigger-api -n "$NAMESPACE"
kubectl describe pod -n "$NAMESPACE" -l app.kubernetes.io/component=api
```

Read logs:

```bash
kubectl logs -n "$NAMESPACE" deploy/activetigger-api --tail=200
kubectl logs -n "$NAMESPACE" deploy/activetigger-api -f
kubectl logs -n "$NAMESPACE" deploy/activetigger-frontend --tail=100
```

Check the internal service and public endpoint:

```bash
kubectl get endpoints activetigger-api -n "$NAMESPACE"
curl https://activetigger.lab.groupe-genes.fr/api/version
curl https://activetigger.lab.groupe-genes.fr/api/server
```


Uninstall the release:

```bash
helm uninstall activetigger -n "$NAMESPACE"
```

PVCs may remain after uninstall, depending on the storage reclaim policy. Delete them only when the project data can be safely removed.


## Known limitations

- `values.yaml` still contains development placeholder secrets for non-Vault local testing.
- The current Vault flow is only a manual bridge. Vault remains the source of truth, but there is no automatic synchronization, rotation, or pod restart when the Vault value changes. More automatic Vault-to-Kubernetes options should be evaluated:
  - **Vault Secrets Operator**: syncs Vault secrets into Kubernetes Secrets, but it is not currently exposed in the ENSAE namespace.
  - **Vault Agent Injector**: injects a Vault Agent init/sidecar and renders secrets as files, but it is not currently available in the ENSAE namespace.
  - **Vault API from Python**: ActiveTigger could read Vault directly at startup
- The API image currently installs Python dependencies dynamically at container startup.
- GPU support is prototype-level and should be validated with explicit CUDA/PyTorch diagnostics.
- The default ingress host is ENSAE-specific.
- The chart currently targets one API replica because of local filesystem persistence.
- Embedded PostgreSQL is useful for testing but should not be the only supported production option.

## Next steps

- Clarify with DSI team whether a supported Vault-to-Kubernetes bridge can be enabled for user namespaces, or whether another secret-management approach without Vault is preferable before catalog integration.
- Validate the Vault mode through the Onyxia service catalog UI.
- Build production images with dependencies preinstalled.
- Add a clean external PostgreSQL configuration path.
- Confirm GPU runtime requirements on the target Kubernetes cluster.
- Package the chart for the Onyxia service catalog once the chart values and image publication workflow are validated.
