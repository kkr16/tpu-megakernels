#!/usr/bin/env bash
# Zero-Code-Change GKE TPU v7 Launcher for tpu-megakernels (Kimi K3 + DSpark).
#
# Packages the repository + `k8s/shim/sitecustomize.py` into
# `configmap/tpu-megakernels-bundle` and deploys `megakernels-kimi`
# on GKE TPU v7 (`tpu7x`, `2x2x4` topology = 4 hosts = 16 chips = 32 TensorCores).
#
# Environment variables:
#   K3_GCS_BUCKET   Optional GCS bucket name containing `k3/` (Orbax), `safetensors/`, and `dspark/`
#   K3_ORBAX_CKPT   Optional explicit Orbax checkpoint URI (defaults to gs://${K3_GCS_BUCKET}/k3 if K3_GCS_BUCKET is set)
#   K3_WEIGHT_MODE  Weight loading mode: `orbax` (default when K3_ORBAX_CKPT is set) or `synthetic`
#
# Usage:
#   K3_GCS_BUCKET=<bucket> ./k8s/run_on_gke.sh --deploy
#   ./k8s/run_on_gke.sh --logs
#   ./k8s/run_on_gke.sh --bench-dspark
#   ./k8s/run_on_gke.sh --bundle
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

export K3_GCS_BUCKET="${K3_GCS_BUCKET:-}"
if [[ -z "${K3_ORBAX_CKPT:-}" && -n "${K3_GCS_BUCKET}" ]]; then
  export K3_ORBAX_CKPT="gs://${K3_GCS_BUCKET}/k3"
else
  export K3_ORBAX_CKPT="${K3_ORBAX_CKPT:-}"
fi
if [[ -z "${K3_WEIGHT_MODE:-}" ]]; then
  if [[ -n "${K3_ORBAX_CKPT}" ]]; then
    export K3_WEIGHT_MODE="orbax"
  else
    export K3_WEIGHT_MODE="synthetic"
  fi
fi

kc() { env -u GOOGLE_APPLICATION_CREDENTIALS kubectl "$@"; }

bundle_configmap() {
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp}"' RETURN

  mkdir -p "${tmp}/stage/tpu-megakernels" "${tmp}/stage/gke/shim"
  tar cf - \
    --exclude='./.git' \
    --exclude='./assets' \
    --exclude='./uv.lock' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='.jax_cache' \
    . | tar xf - -C "${tmp}/stage/tpu-megakernels"
  cp "${REPO_DIR}/k8s/shim/sitecustomize.py" "${tmp}/stage/gke/shim/sitecustomize.py"

  tar czf "${tmp}/tpu-megakernels-bundle.tgz" -C "${tmp}/stage" tpu-megakernels gke/shim

  local size_kb=$(( $(stat -c%s "${tmp}/tpu-megakernels-bundle.tgz") / 1024 ))
  echo "[run_on_gke] Packaged tpu-megakernels + k8s/shim (${size_kb} KiB)"
  kc create configmap tpu-megakernels-bundle \
    --from-file=tpu-megakernels-bundle.tgz="${tmp}/tpu-megakernels-bundle.tgz" \
    --dry-run=client -o yaml | kc replace --force -f -
}

case "${1:---deploy}" in
  --bundle)
    bundle_configmap
    ;;

  --deploy)
    bundle_configmap
    echo "[run_on_gke] Reclaiming TPU v7 2x2x4 slice..."
    kc delete jobset megakernels-kimi --ignore-not-found --wait=true --timeout=5m || true
    kc wait --for=delete pod -l cloud.google.com/gke-tpu-topology=2x2x4 --timeout=2m >/dev/null 2>&1 || true
    sleep 3
    echo "[run_on_gke] Deploying JobSet megakernels-kimi (K3_WEIGHT_MODE=${K3_WEIGHT_MODE})..."
    envsubst '${K3_WEIGHT_MODE} ${K3_GCS_BUCKET} ${K3_ORBAX_CKPT}' < "${REPO_DIR}/k8s/kimi-k3-tp32-jobset.yaml" | kc apply -f -
    echo "[run_on_gke] Deployed! Follow coordinator logs with:"
    echo "  ./k8s/run_on_gke.sh --logs"
    ;;

  --logs)
    ready_pod="$(kc get pods -l jobset.sigs.k8s.io/jobset-name=megakernels-kimi -o jsonpath='{range .items[?(@.status.containerStatuses[0].ready==true)]}{.metadata.name}{end}' 2>/dev/null || true)"
    if [[ -n "${ready_pod}" ]]; then
      kc logs -f "${ready_pod}"
    else
      kc logs -f -l jobset.sigs.k8s.io/jobset-name=megakernels-kimi --max-log-requests=4
    fi
    ;;

  --bench-dspark)
    port="${PORT:-8000}"
    kc port-forward svc/megakernels-kimi-http "${port}:8000" >/dev/null 2>&1 &
    pf_pid=$!
    trap 'kill "${pf_pid}" 2>/dev/null || true' EXIT
    echo "[run_on_gke] Waiting for http://127.0.0.1:${port}/health ..."
    until curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
      sleep 2
    done
    echo "[run_on_gke] Running streaming benchmark against Kimi K3 + DSpark on GKE..."
    ISL="${ISL:-128}" OSL="${OSL:-256}" N_WARMUP="${N_WARMUP:-1}" N_BENCH="${N_BENCH:-4}" \
      python3 "${REPO_DIR}/bench_openai_stream.py" \
      --base-url "http://127.0.0.1:${port}"
    ;;
esac
