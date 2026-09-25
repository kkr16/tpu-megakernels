# Running `tpu-megakernels` (2.8T Kimi K3 + DSpark) on GKE TPU v7 (`tpu7x`)

This directory (`k8s/`) contains the Kubernetes `JobSet`, `Service`, and zero-code-change runtime bridge (`k8s/shim/sitecustomize.py`) to deploy and serve the **2.8T Kimi K3 + DSpark** TP32 megakernel across a 4-host `2x2x4` GKE TPU v7 Ironwood slice (`16` chips = `32` TensorCores) without modifying a single line of the core megakernel or server code.

---

## 1. Directory Layout (`k8s/`)

```text
k8s/
├── README.md                    # This reproduction guide
├── run_on_gke.sh                # One-command ConfigMap bundler, JobSet deployer & benchmark runner
├── kimi-k3-tp32-jobset.yaml     # GKE 2x2x4 TPU v7 Indexed JobSet + HTTP Service (port 8000)
└── shim/
    └── sitecustomize.py         # Python runtime bridge (GKE->Slurm env + Orbax 2.8T weight loader)
```

---

## 2. Technical Challenges Solved (`k8s/shim/sitecustomize.py`)

| # | GKE TPU v7 Challenge | How `k8s/` Solves It Without Touching Core Code |
|---|---|---|
| 1 | **Multi-Host JAX Coordinator (`distributed_entry.py`)**: Upstream reads `SLURM_PROCID` (`0..3`) and `TP32_COORDINATOR` (`host:port`). | `kimi-k3-tp32-jobset.yaml` and `k8s/shim/sitecustomize.py` map GKE's `TPU_WORKER_ID` $\to$ `SLURM_PROCID` and `TPU_WORKER_HOSTNAMES[0]:8476` $\to$ `TP32_COORDINATOR`, and launch `python3 distributed_entry.py demo_kimi_dspark.py` directly (no Slurm CLI wrappers required). |
| 2 | **Mosaic Compiler Bug on `jax==0.11.0`**: The `2026-08-07` base container ships `jax==0.11.0` (`libtpu==0.0.44`), which fails Mosaic lowering of `_quantized_expert_dot` (`Invalid relayout ... for 'vector<1x24x768xi1>'`). | Pod startup installs the repository's pinned **`jax[tpu]==0.11.1` (`libtpu==0.0.46.1`)**, which compiles all 93 layers of Pallas megakernels cleanly in **59 seconds**. |
| 3 | **Streaming 2.8T Weights Without `gcsfuse`**: When the GKE cluster does not have `gcsfuse.csi.storage.gke.io` enabled or the checkpoint is stored in per-layer Orbax/OCDBT (`gs://<bucket>/k3`), local `emptyDir` cannot hold 1.56 TB of raw `.safetensors`. | `k8s/shim/sitecustomize.py` subclasses `kimi.load.Checkpoint` with `OrbaxCheckpoint` (`K3_WEIGHT_MODE=orbax`), streaming the 2.8T checkpoint directly from `gs://<bucket>/k3` via `TensorStore` (`gcs_request_concurrency=128`). Each of the 4 hosts streams **only its own 224-expert shard** (`[0:224]`, `[224:448]`, `[448:672]`, `[672:896]`) with a 6-layer lookahead pipeline and vectorized `<u4` packing, passing `kimi/load.py`'s lossless on-device FP8 reconstruction check across all `2,944 / 2,944` expert rank-layers in **268 seconds**. |
| 4 | **GKE `TPU_WORKER_ID=0` Physical Host Routing**: On `2x2x4` slices, GKE assigns `TPU_WORKER_ID=0` (`jax.process_index() == 0`, which runs the OpenAI HTTP server on port `8000`) to whichever pod lands on physical coordinate `(0,0,0)`. | `kimi-k3-tp32-jobset.yaml` configures `readinessProbe: httpGet /health:8000` on all pods and omits `job-completion-index` from `service/megakernels-kimi-http`, routing traffic exclusively to the `1/1 Ready` coordinator pod. |

---

## 3. Quick Start (Step-by-Step)

### Step 1: Deploy the 4-Host TPU v7 Megakernel JobSet
From the repository root, deploy either with your GCS checkpoint bucket (`K3_GCS_BUCKET=<your-bucket>`) or in zero-checkpoint synthetic benchmark mode (`K3_WEIGHT_MODE=synthetic`):
```bash
# With trained 2.8T weights in gs://<your-bucket>/{k3,safetensors,dspark}:
K3_GCS_BUCKET=<your-bucket> ./k8s/run_on_gke.sh --deploy

# Or with fast on-device synthetic weights (<2s allocation) for kernel benchmarking:
K3_WEIGHT_MODE=synthetic ./k8s/run_on_gke.sh --deploy
```

### Step 2: Stream Startup & Weight-Loading Logs (~4 min 35 s)
```bash
./k8s/run_on_gke.sh --logs
```
Wait until the coordinator pod logs:
```text
ready: target weights in 268 s on this host, total start-up 275 s (all hosts)
    stages: draft weights resident in 31 s; programs compiled in 59 s; state buffers in 1 s; early collective 0.4 s; pacer: 107 probes, longest 0.2 s; tokenizer 0.2 s; state buffers ready 0.0 s; prefill launch 0.4 s; speculative launch 0.0 s

OpenAI-compatible server listening on http://0.0.0.0:8000/v1 (model 'kimi-k3')
```

### Step 3: Port-Forward the OpenAI-Compatible Server (`:8000`)
```bash
kubectl port-forward svc/megakernels-kimi-http 8000:8000
```

### Step 4: Run the Streaming OpenAI Benchmark
```bash
./k8s/run_on_gke.sh --bench-dspark
```
