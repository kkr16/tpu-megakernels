"""Zero-code-change GKE runtime bridge for Inferact/tpu-megakernels.

Placed on PYTHONPATH (/opt/gke-shim) so Python automatically imports it at startup
without modifying any file inside the `tpu-megakernels` repository.

Responsibilities:
1. Map GKE TPU JobSet env vars (`TPU_WORKER_ID`, `JOB_COMPLETION_INDEX`,
   `TPU_WORKER_HOSTNAMES`, `JOBSET_NAME`) to the Slurm env vars expected by
   `distributed_entry.py` (`SLURM_PROCID`, `TP32_COORDINATOR`).
2. Ensure `runpy.run_path` sets `sys.argv[0]` to the target script path so
   `distributed_entry.py` forwards CLI flags (`--serve`, `--context`, etc.)
   cleanly to `demo_kimi_dspark.py`.
3. Provide `OrbaxCheckpoint` (`K3_WEIGHT_MODE=orbax` when `K3_ORBAX_CKPT` is set)
   so `kimi.load.load_weights` can stream the 2.8T Kimi K3 weights directly from
   a GCS Orbax/OCDBT checkpoint (`gs://<bucket>/k3`) with parallel TensorStore
   prefetching and vectorized TP32 expert slicing, or `K3_WEIGHT_MODE=synthetic`
   for fast zero-checkpoint compilation/throughput testing.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import os
import runpy
import sys
import threading


def _bridge_gke_to_slurm_env() -> None:
    if "SLURM_PROCID" not in os.environ:
        rank = os.environ.get("TPU_WORKER_ID") or os.environ.get("JOB_COMPLETION_INDEX")
        if rank is not None:
            os.environ["SLURM_PROCID"] = str(rank)

    if "TP32_COORDINATOR" not in os.environ:
        hostnames = os.environ.get("TPU_WORKER_HOSTNAMES", "")
        if hostnames:
            coord_host = hostnames.split(",")[0].strip()
            os.environ["TP32_COORDINATOR"] = f"{coord_host}:8476"
        elif "JOBSET_NAME" in os.environ:
            js = os.environ["JOBSET_NAME"]
            rjob = os.environ.get("REPLICATED_JOB_NAME", "slice")
            os.environ["TP32_COORDINATOR"] = f"{js}-{rjob}-0-0.{js}:8476"


_orig_run_path = runpy.run_path


def _patched_run_path(path_name, init_globals=None, run_name=None):
    if run_name == "__main__" and sys.argv and sys.argv[0].endswith("distributed_entry.py"):
        sys.argv[0] = str(path_name)
    return _orig_run_path(path_name, init_globals=init_globals, run_name=run_name)


runpy.run_path = _patched_run_path


def _install_weight_loader() -> None:
    mode = os.environ.get("K3_WEIGHT_MODE", "auto").lower()
    orbax_uri = os.environ.get("K3_ORBAX_CKPT", "").strip().rstrip("/")
    if mode == "auto":
        mode = "orbax" if orbax_uri else "safetensors"
    if mode not in ("orbax", "synthetic"):
        return

    import importlib.abc

    class _KimiLoadPatcher(importlib.abc.MetaPathFinder):
        _patched = False

        def find_spec(self, fullname, path, target=None):
            if fullname == "kimi.load" and not self._patched:
                self._patched = True
                for finder in sys.meta_path:
                    if finder is self:
                        continue
                    spec = finder.find_spec(fullname, path, target)
                    if spec is not None and spec.loader is not None:
                        orig_exec = spec.loader.exec_module

                        def _exec_module(module):
                            orig_exec(module)
                            if mode == "synthetic":
                                _patch_synthetic_loader(module)
                            elif mode == "orbax" and orbax_uri:
                                _patch_orbax_checkpoint(module, orbax_uri)

                        spec.loader.exec_module = _exec_module
                        return spec
            return None

    sys.meta_path.insert(0, _KimiLoadPatcher())


def _patch_orbax_checkpoint(module, orbax_uri: str) -> None:
    import jax
    import numpy as np
    import tensorstore as ts

    OrigCheckpoint = module.Checkpoint
    PREFIX = module.PREFIX
    Config = module.Config

    ts_ctx = ts.Context(
        {
            "gcs_request_concurrency": {"limit": 128},
            "file_io_concurrency": {"limit": 128},
            "data_copy_concurrency": {"limit": 64},
        }
    )

    dense_lock = threading.Lock()
    dense_futures: dict[str, Future] = {}
    dense_pool = ThreadPoolExecutor(max_workers=64)
    prefetch_started = False

    def _parse_key(key: str) -> tuple[str, str]:
        if key.startswith(PREFIX + "layers."):
            rest = key[len(PREFIX + "layers.") :]
            layer_str, rel = rest.split(".", 1)
            return f"layers/{layer_str}", rel
        if key.startswith(PREFIX):
            return "top/model-00094-of-000096", key[len(PREFIX) :]
        if key.startswith("language_model."):
            return "top/model-00094-of-000096", key[len("language_model.") :]
        raise KeyError(f"Unexpected Kimi K3 checkpoint key: {key}")

    def _fetch_dense_tensor(key: str) -> np.ndarray:
        sub, rel = _parse_key(key)
        spec = {
            "driver": "zarr",
            "kvstore": {"driver": "ocdbt", "base": f"{orbax_uri}/{sub}"},
            "path": rel,
        }
        arr = np.asarray(ts.open(spec, context=ts_ctx).result().read().result())
        if rel == "self_attn.A_log":
            return np.pad(arr, (0, 32))
        if rel.endswith("conv1d.weight"):
            return np.ascontiguousarray(arr[:, None, :])
        if arr.ndim == 2 and not rel.endswith("embed_tokens.weight"):
            return np.ascontiguousarray(arr.T)
        return np.ascontiguousarray(arr)

    def _get_dense_future(key: str) -> Future:
        with dense_lock:
            fut = dense_futures.get(key)
            if fut is None:
                fut = dense_pool.submit(_fetch_dense_tensor, key)
                dense_futures[key] = fut
            return fut

    def _start_all_dense_prefetch(config, layers: int = 93) -> None:
        nonlocal prefetch_started
        with dense_lock:
            if prefetch_started:
                return
            prefetch_started = True

        keys = [
            PREFIX + "embed_tokens.weight",
            "language_model.lm_head.weight",
            PREFIX + "norm.weight",
            PREFIX + "output_attn_res_norm.weight",
            PREFIX + "output_attn_res_proj.weight",
        ]
        kda_suffixes = (
            "self_attn.f_a_proj.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.g_proj.weight",
            "self_attn.b_proj.weight",
            "self_attn.f_b_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_conv1d.weight",
            "self_attn.k_conv1d.weight",
            "self_attn.v_conv1d.weight",
            "self_attn.A_log",
            "self_attn.dt_bias",
            "self_attn.o_norm.weight",
        )
        mla_suffixes = (
            "self_attn.q_a_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.g_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.kv_a_layernorm.weight",
        )
        common_layer_suffixes = (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attention_res_norm.weight",
            "self_attention_res_proj.weight",
            "mlp_res_norm.weight",
            "mlp_res_proj.weight",
        )
        moe_dense_suffixes = (
            "block_sparse_moe.gate.weight",
            "block_sparse_moe.gate.e_score_correction_bias",
            "block_sparse_moe.routed_expert_norm.weight",
            "block_sparse_moe.routed_expert_down_proj.weight",
            "block_sparse_moe.routed_expert_up_proj.weight",
            "block_sparse_moe.shared_experts.gate_proj.weight",
            "block_sparse_moe.shared_experts.up_proj.weight",
            "block_sparse_moe.shared_experts.down_proj.weight",
        )
        dense_mlp_suffixes = (
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        )
        for layer in range(layers):
            base = f"{PREFIX}layers.{layer}."
            for s in common_layer_suffixes:
                keys.append(base + s)
            for s in (mla_suffixes if layer in config.full_attention else kda_suffixes):
                keys.append(base + s)
            for s in (dense_mlp_suffixes if layer == 0 else moe_dense_suffixes):
                keys.append(base + s)
        for k in keys:
            _get_dense_future(k)

    moe_lock = threading.Condition()
    moe_started = False
    moe_ready: dict[int, dict[tuple[int, int], tuple]] = {}
    moe_refcounts: dict[int, int] = {}
    MAX_MOE_IN_FLIGHT = 6

    def _fetch_and_slice_moe_layer(layer: int, host_first: int) -> dict[tuple[int, int], tuple]:
        base = f"{orbax_uri}/layers/{layer}"
        names = [
            f"block_sparse_moe.experts.{w}.{k}"
            for w in ("w1", "w2", "w3")
            for k in ("weight_packed", "weight_scale")
        ]
        futs = {
            n: ts.open(
                {"driver": "zarr", "kvstore": {"driver": "ocdbt", "base": base}, "path": n},
                context=ts_ctx,
            )
            for n in names
        }
        stores = {n: f.result() for n, f in futs.items()}
        read_futs = {n: s[host_first : host_first + 224].read() for n, s in stores.items()}
        raw = {n: np.asarray(f.result()) for n, f in read_futs.items()}

        out: dict[tuple[int, int], tuple] = {}
        width = 3072 // 4
        for g in range(2):
            first_exp = host_first + g * 112
            sl = slice(g * 112, (g + 1) * 112)
            w1_pk = raw["block_sparse_moe.experts.w1.weight_packed"][sl]
            w3_pk = raw["block_sparse_moe.experts.w3.weight_packed"][sl]
            w1_sc = raw["block_sparse_moe.experts.w1.weight_scale"][sl]
            w3_sc = raw["block_sparse_moe.experts.w3.weight_scale"][sl]
            w2_pk = raw["block_sparse_moe.experts.w2.weight_packed"][sl]
            w2_sc = raw["block_sparse_moe.experts.w2.weight_scale"][sl]
            for part in range(4):
                w1_p = np.ascontiguousarray(
                    w1_pk[:, :, part * width : (part + 1) * width].transpose(0, 2, 1)
                )
                w3_p = np.ascontiguousarray(
                    w3_pk[:, :, part * width : (part + 1) * width].transpose(0, 2, 1)
                )
                w1_s = w1_sc[:, :, part * width : (part + 1) * width]
                w3_s = w3_sc[:, :, part * width : (part + 1) * width]
                gu = np.concatenate(
                    [w1_p.view("<u4").transpose(0, 2, 1), w3_p.view("<u4").transpose(0, 2, 1)],
                    axis=2,
                ).copy()
                gus = np.concatenate([w1_s, w3_s], axis=2).copy()
                w2_p = np.ascontiguousarray(
                    w2_pk[:, part * (width // 2) : (part + 1) * (width // 2), :].transpose(0, 2, 1)
                )
                downs = w2_p.view("<u4").transpose(0, 2, 1).copy()
                down_s = w2_sc[:, part * (width // 32) : (part + 1) * (width // 32), :].copy()
                out[(first_exp, part)] = (gu, gus, downs, down_s)
        del raw
        return out

    def _ensure_moe_pipeline(host_first: int, total_layers: int = 93) -> None:
        nonlocal moe_started
        with moe_lock:
            if moe_started:
                return
            moe_started = True

        with dense_lock:
            dense_futures.clear()

        def _producer():
            pool = ThreadPoolExecutor(max_workers=MAX_MOE_IN_FLIGHT)
            in_flight: dict[int, Future] = {}
            next_layer = 1
            for _ in range(min(MAX_MOE_IN_FLIGHT, total_layers - 1)):
                in_flight[next_layer] = pool.submit(_fetch_and_slice_moe_layer, next_layer, host_first)
                next_layer += 1

            for layer in range(1, total_layers):
                res = in_flight.pop(layer).result()
                with moe_lock:
                    while len(moe_ready) >= MAX_MOE_IN_FLIGHT:
                        moe_lock.wait()
                    moe_ready[layer] = res
                    moe_refcounts[layer] = 8
                    moe_lock.notify_all()
                if next_layer < total_layers:
                    in_flight[next_layer] = pool.submit(
                        _fetch_and_slice_moe_layer, next_layer, host_first
                    )
                    next_layer += 1
            pool.shutdown(wait=False)

        threading.Thread(target=_producer, daemon=True).start()

    class OrbaxCheckpoint(OrigCheckpoint):
        def __init__(self, path):
            self.path = path
            self.config = Config()
            self._layouts = {}

        def read(self, key, selection=None):
            _start_all_dense_prefetch(self.config, self.config.layers)
            val = _get_dense_future(key).result()
            return val[selection] if selection is not None else val

        def experts(self, layer, first, count, *, part=None, parts=4):
            host_first = 224 * jax.process_index()
            _ensure_moe_pipeline(host_first, self.config.layers)
            with moe_lock:
                while layer not in moe_ready:
                    moe_lock.wait()
                item = moe_ready[layer][(first, part)]
                moe_refcounts[layer] -= 1
                if moe_refcounts[layer] == 0:
                    del moe_ready[layer]
                    del moe_refcounts[layer]
                    moe_lock.notify_all()
                return item

    orig_abstract_weights = module.abstract_weights

    def _fast_real_abstract_weights(mesh, path=None, *, layers: int = 93, vocab: int = 163840):
        out = orig_abstract_weights(mesh, None, layers=layers, vocab=vocab)
        for fp32_name in ("k_norm", "k_conv"):
            if fp32_name in out:
                s = out[fp32_name]
                out[fp32_name] = jax.ShapeDtypeStruct(s.shape, np.dtype(np.float32), sharding=s.sharding)
        return out

    module.abstract_weights = _fast_real_abstract_weights
    module.Checkpoint = OrbaxCheckpoint
    print(
        f"[gke-shim] Installed OrbaxCheckpoint loader against {orbax_uri} "
        f"(host={jax.process_index()}, experts=[{224*jax.process_index()}:{224*(jax.process_index()+1)}])",
        flush=True,
    )


def _patch_synthetic_loader(module) -> None:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    def _fast_load_weights(
        path,
        mesh,
        *,
        layers: int = 93,
        ranks_in_flight: int = 8,
        progress=None,
        log=None,
    ):
        shapes = module.selected_local_shapes(layers, 7168)
        fp8_shapes = module.fp8_expert_shapes(layers)
        dtypes = {
            name: value.dtype
            for name, value in module.synthetic_rank_weights(0, layers=2, include_lm_head=True)
        }
        dtypes["k_gate"] = dtypes["k_projection"]
        sharding = NamedSharding(mesh, P("tp"))
        weights = {}
        for name, shape in shapes.items():
            dtype = dtypes[name]
            if name in fp8_shapes:
                shape, dtype = fp8_shapes[name]
            j_dtype = jnp.dtype(dtype)
            full_shape = (32, *shape)
            if name in ("expert_gus", "expert_ds"):
                fn = jax.jit(lambda s=full_shape, d=j_dtype: jnp.full(s, 120, dtype=d), out_shardings=sharding)
            elif name in ("attn_norm", "ffn_norm", "k_norm", "m_qnorm", "m_knorm", "final_norm"):
                fn = jax.jit(lambda s=full_shape, d=j_dtype: jnp.ones(s, dtype=d), out_shardings=sharding)
            else:
                fn = jax.jit(lambda s=full_shape, d=j_dtype: jnp.zeros(s, dtype=d), out_shardings=sharding)
            weights[name] = fn()
        jax.block_until_ready(weights)
        return weights

    module.load_weights = _fast_load_weights


_bridge_gke_to_slurm_env()
_install_weight_loader()
