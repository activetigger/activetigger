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
- `templates/secret.yaml`: generated or external secret references.

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

Additional Dockerfiles were created for Kubernetes/Helm testing because the existing Docker Compose setup does not map directly to Kubernetes deployment conventions:

- `docker/api/Dockerfile.prototype`: API image used for the CPU prototype;
- `docker/api/Dockerfile.gpu-prototype`: API image used for the GPU prototype;
- `docker/frontend/Dockerfile.prototype`: frontend image used by the chart;
- `docker/frontend/nginx.prototype.conf.template`: nginx template used by the frontend image to route `/api` to the API service.

The chart defaults to these repositories in `charts/activetigger/values.yaml`. The CPU prototype uses the `prototype` API tag, and the GPU prototype uses the `gpu-prototype` API tag.

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

Deploy the GPU prototype:

```bash
helm upgrade --install activetigger ./charts/activetigger \
  --namespace "$NAMESPACE" \
  --server-side=false \
  --timeout 25m \
  --set api.image.tag=gpu-prototype \
  --set api.gpu.enabled=true \
  --set api.env.CPU_ONLY=false \
  --set api.env.GPU=true \
  --set api.env.N_WORKERS_GPU=1 \
  --set api.env.HF_HOME=/data/models/huggingface \
  --set api.env.TRANSFORMERS_CACHE=/data/models/huggingface/transformers \
  --set api.env.SENTENCE_TRANSFORMERS_HOME=/data/models/sentence-transformers
```

On the ENSAE cluster, `--server-side=false` was needed because server-side apply requests for some workloads were blocked by the gateway/WAF.

The HuggingFace/SentenceTransformers cache variables are not required for startup, but they are recommended. They store downloaded embedding models under `/data/models`, which is backed by the API PVC, so models do not need to be downloaded again after every pod restart.

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

- Some secrets are still configurable directly through `values.yaml`.
- The API image currently installs Python dependencies dynamically at container startup.
- GPU support is prototype-level and should be validated with explicit CUDA/PyTorch diagnostics.
- The default ingress host is ENSAE-specific.
- The chart currently targets one API replica because of local filesystem persistence.
- Embedded PostgreSQL is useful for testing but should not be the only supported production option.

## Next steps

- Move secrets to an external Secret or Onyxia-compatible secret mechanism.
- Build production images with dependencies preinstalled.
- Add a clean external PostgreSQL configuration path.
- Confirm GPU runtime requirements on the target Kubernetes cluster.
- Package the chart for the Onyxia service catalog once the chart values and image publication workflow are validated.
