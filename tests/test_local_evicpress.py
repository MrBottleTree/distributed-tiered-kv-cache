# SPDX-License-Identifier: Apache-2.0
"""Real core/adapters on CPU; only LMCache's GPU-dependent allocation API is stubbed.

Install evicpress-core before running. These checks do not load models, CUDA
extensions or vLLM, and do not change the official benchmark harness.
"""

import asyncio
from dataclasses import dataclass
import importlib
import importlib.util
import logging
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch
import yaml

from test_head_segments import Pool

ROOT = Path(__file__).parents[1]
BACKENDS = ROOT / "LMCache/lmcache/v1/storage_backend"
PACKAGE = "local_evicpress_test_backend"
package = ModuleType(PACKAGE)
package.__path__ = [str(BACKENDS)]
sys.modules[PACKAGE] = package
selection = importlib.import_module(PACKAGE + ".evicpress_config")
pb = importlib.import_module(PACKAGE + ".evicpress_pb2")
transport = importlib.import_module(PACKAGE + ".evicpress_transport")

try:
    from evicpress.local import LocalEvicPressClient
except ImportError:
    LocalEvicPressClient = None


@dataclass(frozen=True)
class Key:
    model_name: str = "test/model"
    world_size: int = 1
    worker_id: int = 0
    chunk_hash: int = 123
    dtype: object = torch.float16
    tags: object = None


class PluginBase:
    """Allocation-interface stand-in, not a mock of storage/transport behavior."""

    def __init__(self, dst_device, config, metadata, local_cpu_backend, loop):
        self.config = config
        self.metadata = metadata
        self.local_cpu_backend = local_cpu_backend

    def batched_contains(self, keys, pin=False):
        hits = 0
        for key in keys:
            if not self.contains(key, pin):
                break
            hits += 1
        return hits

    def batched_get_blocking(self, keys):
        return [self.get_blocking(key) for key in keys]


class CPUBackend(Pool):
    def __init__(self, capacity=8192):
        super().__init__(capacity)
        self.mirrors = {}

    def allocate(self, shape, dtype, **kwargs):
        return super().allocate(shape, dtype)

    def get_allocator_backend(self):
        return self

    def contains(self, key, pin=False):
        return key in self.mirrors

    def submit_put_task(self, key, obj):
        if key not in self.mirrors:
            obj.ref_count_up()
            self.mirrors[key] = obj

    def remove(self, key, force=True):
        obj = self.mirrors.pop(key, None)
        if obj is not None:
            obj.ref_count_down()
        return obj is not None

    def close(self):
        for key in list(self.mirrors):
            self.remove(key)


def module(name, **attributes):
    value = ModuleType(name)
    value.__dict__.update(attributes)
    return value


GPU_STUBS = {
    "lmcache.utils": module("lmcache.utils", CacheEngineKey=Key),
    "lmcache.v1.memory_management": module(
        "lmcache.v1.memory_management",
        MemoryFormat=SimpleNamespace(KV_2LTD="KV_2LTD"),
        MemoryObj=object,
    ),
    "lmcache.v1.storage_backend.abstract_backend": module(
        "lmcache.v1.storage_backend.abstract_backend", StoragePluginInterface=PluginBase
    ),
    "lmcache.v1.storage_backend": module(
        "lmcache.v1.storage_backend", evicpress_pb2=pb
    ),
}
with mock.patch.dict(sys.modules, GPU_STUBS):
    LocalBackend = importlib.import_module(
        PACKAGE + ".local_evicpress_backend"
    ).LocalEvicPressBackend
    RemoteBackend = importlib.import_module(PACKAGE + ".grpc_backend").GRPCBackend


def settings():
    return SimpleNamespace(
        extra_config={},
        storage_plugins=None,
        enable_pd=False,
        enable_scheduler_bypass_lookup=False,
        enable_async_loading=False,
        enable_p2p=False,
        use_layerwise=False,
        local_disk=None,
        remote_url=None,
        gds_path=None,
        max_local_cpu_size=1e-6,
    )


def metadata():
    return SimpleNamespace(
        model_name="test/model",
        world_size=1,
        role="worker",
        kv_dtype=torch.float16,
        kv_shape=(1, 2, 5, 2, 4),
        use_mla=False,
        get_num_groups=lambda: 1,
    )


def write_core_config(directory, *, level="fp16", mirrors=False, prefetch=False):
    # Keep fixtures independent of another repository's source layout.
    raw = yaml.safe_load((ROOT / "evicpress_local_config.yaml").read_text())
    raw["tier3"]["data_dir"] = str(Path(directory) / "cache")
    raw["tier3"]["capacity_bytes"] = 1024 * 1024
    raw["tier2"]["capacity_bytes"] = 0
    raw["tier1"]["capacity_bytes"] = 1024 * 1024 if mirrors else 0
    raw["prefetch"]["enabled"] = prefetch
    raw["quantization"]["enabled"] = level != "fp16"
    raw["placement"]["bands"] = [
        {
            "name": "test",
            "min_utility": 0,
            "tiers": [1, 3] if mirrors else [3],
            "quant": level,
        }
    ]
    path = Path(directory) / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def backend_config(path, granularity="chunk"):
    cfg = settings()
    cfg.extra_config = {
        "evicpress_backend": "local",
        "evicpress_config_path": str(path),
        "evicpress_model_revision": "pinned",
        "evicpress_granularity": granularity,
    }
    return cfg


def allocation(pool):
    obj = pool.allocate((2, 1, 5, 8), torch.float16)
    obj.tensor.copy_(torch.linspace(-1, 1, 80).reshape(2, 1, 5, 8).half())
    return obj


def load_launcher():
    """Load the real plugin launcher with GPU-only dependencies substituted."""
    dependencies = dict(GPU_STUBS)
    dependencies["lmcache.logging"] = module(
        "lmcache.logging", init_logger=logging.getLogger
    )
    dependencies["lmcache.v1.config"] = module(
        "lmcache.v1.config", LMCacheEngineConfig=object
    )
    dependencies["lmcache.v1.metadata"] = module(
        "lmcache.v1.metadata", LMCacheMetadata=object
    )
    dependencies["lmcache.v1.storage_backend.abstract_backend"] = module(
        "lmcache.v1.storage_backend.abstract_backend", StorageBackendInterface=object
    )
    for filename, name in (
        ("gds_backend", "GdsBackend"),
        ("local_cpu_backend", "LocalCPUBackend"),
        ("local_disk_backend", "LocalDiskBackend"),
        ("p2p_backend", "P2PBackend"),
        ("remote_backend", "RemoteBackend"),
    ):
        full = "lmcache.v1.storage_backend." + filename
        dependencies[full] = module(full, **{name: object})
    spec = importlib.util.spec_from_file_location(
        "local_launcher_test", BACKENDS / "__init__.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(loaded)
    return loaded.storage_plugin_launcher


class SelectionTest(unittest.TestCase):
    def test_remote_default_explicit_disable_and_custom_plugins(self):
        cfg = settings()
        selection.configure_evicpress_plugin(cfg)
        self.assertEqual(cfg.storage_plugins, ["grpc"])
        for plugins in ([], ["custom"]):
            cfg = settings()
            cfg.storage_plugins = plugins
            selection.configure_evicpress_plugin(cfg)
            self.assertEqual(cfg.storage_plugins, plugins)

    def test_local_selection_is_exclusive_and_required(self):
        cfg = settings()
        cfg.extra_config["evicpress_backend"] = "local"
        selection.configure_evicpress_plugin(cfg)
        self.assertEqual(cfg.storage_plugins, ["evicpress_local"])
        self.assertTrue(cfg.extra_config["storage_plugin.evicpress_local.required"])
        cfg.storage_plugins = ["grpc"]
        with self.assertRaises(ValueError):
            selection.configure_evicpress_plugin(cfg)
        cfg.storage_plugins = ["evicpress_local"]
        cfg.enable_scheduler_bypass_lookup = True
        with self.assertRaises(ValueError):
            selection.configure_evicpress_plugin(cfg)

    def test_required_launcher_failures_cannot_silently_disable_backend(self):
        launch = load_launcher()
        cfg = settings()
        cfg.storage_plugins = ["evicpress_local"]
        with self.assertRaises(ValueError):
            launch(cfg, metadata(), None, None, "cpu", {})
        cfg.extra_config = {
            "storage_plugin.evicpress_local.module_path": "missing_local_package",
            "storage_plugin.evicpress_local.class_name": "LocalEvicPressBackend",
        }
        with self.assertRaises(RuntimeError):
            launch(cfg, metadata(), None, None, "cpu", {})


@unittest.skipIf(
    LocalEvicPressClient is None, "Install evicpress-core to test real local storage"
)
class LocalAdapterTest(unittest.TestCase):
    def test_launcher_creates_one_runtime_and_skips_existing_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_core_config(directory)
            cfg = backend_config(path)
            selection.configure_evicpress_plugin(cfg)
            cfg.extra_config["storage_plugin.evicpress_local.module_path"] = (
                PACKAGE + ".local_evicpress_backend"
            )
            launch = load_launcher()
            backends = {}
            with mock.patch.dict(sys.modules, GPU_STUBS):
                launch(cfg, metadata(), None, CPUBackend(), "cpu", backends)
            first = backends["evicpress_local"]
            try:
                launch(
                    cfg,
                    metadata(),
                    None,
                    CPUBackend(),
                    "cpu",
                    backends,
                    skip_plugins=set(backends),
                )
                self.assertIs(backends["evicpress_local"], first)
            finally:
                first.close()

    def test_chunk_restore_precision_tags_and_no_channel(self):
        for level, tolerance in (("fp16", 0), ("int8", 0.01), ("int4", 0.08)):
            with self.subTest(level=level), tempfile.TemporaryDirectory() as directory:
                path = write_core_config(directory, level=level)
                pool = CPUBackend()
                source = allocation(pool)
                first = Key(tags=(("sample", "a"),))
                second = Key(tags=(("sample", "b"),))
                reordered = Key(tags=(("z", "1"), ("a", "2")))
                reordered_again = Key(tags=(("a", "2"), ("z", "1")))
                with mock.patch(
                    "grpc.insecure_channel",
                    side_effect=AssertionError("local mode opened gRPC"),
                ):
                    backend = LocalBackend(
                        config=backend_config(path),
                        metadata=metadata(),
                        local_cpu_backend=pool,
                    )
                    try:
                        backend.batched_submit_put_task(
                            [first, reordered], [source, source]
                        )
                        self.assertTrue(backend.contains(first))
                        self.assertTrue(backend.contains(reordered_again))
                        self.assertFalse(backend.contains(second))
                        restored = backend.get_blocking(first)
                        self.assertIsNotNone(restored)
                        self.assertLessEqual(
                            (restored.tensor - source.tensor).abs().max().item(),
                            tolerance,
                        )
                        restored.ref_count_down()
                        self.assertIsNone(backend.get_blocking(second))
                        state = backend.get_state()
                        self.assertEqual(state["transport_metrics"]["rpc_calls"], 0)
                        self.assertEqual(state["transport_metrics"]["wire_bytes"], 0)
                        self.assertGreater(
                            state["transport_metrics"]["payload_bytes"], 0
                        )
                        self.assertTrue(backend.remove(first))
                        self.assertFalse(backend.contains(first))
                    finally:
                        backend.close()
                        source.ref_count_down()
                        pool.close()
                self.assertEqual(pool.used, 0)

    def test_chunk_tier1_mirror_and_legacy_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_core_config(directory, mirrors=True)
            pool = CPUBackend()
            source = allocation(pool)
            backend = LocalBackend(
                config=backend_config(path), metadata=metadata(), local_cpu_backend=pool
            )
            try:
                key = Key()
                backend.batched_submit_put_task([key], [source])
                self.assertTrue(pool.contains(key))
                calls = backend.get_state()["transport_metrics"]["calls"]
                self.assertTrue(backend.contains(key))
                self.assertEqual(
                    backend.get_state()["transport_metrics"]["calls"], calls
                )
                legacy = "test/model|1|0|123|torch.float16"
                self.assertTrue(backend.transport.client.lookup(legacy).hit)
                self.assertTrue(backend.remove(key))
                self.assertFalse(backend.contains(key))
            finally:
                backend.close()
                source.ref_count_down()
                pool.close()
            self.assertEqual(pool.used, 0)

    def test_head_warm_and_cold_restore_leases_and_pool_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_core_config(directory, mirrors=True, prefetch=True)
            pool = CPUBackend()
            source = allocation(pool)
            cfg = backend_config(path, "head")
            with mock.patch(
                "grpc.insecure_channel",
                side_effect=AssertionError("local mode opened gRPC"),
            ):
                backend = LocalBackend(
                    config=cfg, metadata=metadata(), local_cpu_backend=pool
                )
                try:
                    key = Key(tags=(("sample", "head"),))
                    backend.batched_submit_put_task([key], [source])
                    self.assertEqual(backend.batched_contains([key], pin=True), 1)
                    result = backend.batched_get_blocking([key])[0]
                    self.assertTrue(torch.equal(result.tensor, source.tensor))
                    result.ref_count_down()
                    self.assertEqual(backend.head_client.metrics["rpc_calls"], 0)
                    self.assertEqual(backend.head_client.metrics["wire_bytes"], 0)
                finally:
                    backend.close()
                self.assertEqual(pool.used, source.tensor.nbytes)
                # Reopen with no head mirrors: all heads must come from disk.
                cfg.max_local_cpu_size = 0
                backend = LocalBackend(
                    config=cfg, metadata=metadata(), local_cpu_backend=pool
                )
                try:
                    self.assertEqual(
                        backend.batched_contains([key, Key(chunk_hash=999)], pin=True),
                        1,
                    )
                    self.assertFalse(backend.remove(key, force=False))
                    result = backend.get_blocking(key)
                    self.assertTrue(torch.equal(result.tensor, source.tensor))
                    result.ref_count_down()
                    self.assertTrue(backend.remove(key))
                    self.assertEqual(backend.batched_contains([key]), 0)
                    self.assertEqual(backend.get_state()["stats"]["head_rpc_calls"], 0)
                finally:
                    backend.close()
                    source.ref_count_down()
            self.assertEqual(pool.used, 0)

    def test_ownership_missing_config_and_unsupported_worker_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_core_config(directory)
            backend = LocalBackend(config=backend_config(path), metadata=metadata())
            try:
                with self.assertRaises(RuntimeError):
                    LocalBackend(config=backend_config(path), metadata=metadata())
            finally:
                backend.close()
            cfg = backend_config(path)
            cfg.extra_config["evicpress_config_path"] = str(
                path.with_name("missing.yaml")
            )
            with self.assertRaises(FileNotFoundError):
                LocalBackend(config=cfg, metadata=metadata())
            for attribute, value in (("role", "scheduler"), ("world_size", 2)):
                meta = metadata()
                setattr(meta, attribute, value)
                with self.assertRaises(ValueError):
                    LocalBackend(config=backend_config(path), metadata=meta)
            cfg = backend_config(path)
            cfg.enable_scheduler_bypass_lookup = True
            with self.assertRaises(ValueError):
                LocalBackend(config=cfg, metadata=metadata())

    def test_corrupt_chunk_and_allocator_exhaustion_fail_as_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_core_config(directory)
            pool = CPUBackend(capacity=160)
            source = allocation(pool)
            backend = LocalBackend(
                config=backend_config(path), metadata=metadata(), local_cpu_backend=pool
            )
            try:
                key = Key()
                backend.batched_submit_put_task([key], [source])
                self.assertIsNone(backend.get_blocking(key))  # No staging space.
                source.ref_count_down()
                # Keep the actual encoded identity, but replace its bytes with corruption.
                backend.transport.client.store(
                    "test/model|1|0|123|torch.float16", b"bad", 0.5
                )
                self.assertIsNone(backend.get_blocking(key))
                self.assertFalse(backend.contains(key))
            finally:
                backend.close()
            self.assertEqual(pool.used, 0)

    def test_real_loopback_chunk_transport_matches_local(self):
        try:
            import grpc
            from evicpress.config import load_config
            from evicpress.manager import EvicPressManager
            from generated import evicpress_pb2_grpc as rpc
            from server.grpc_server import EvicPressServicer
        except ImportError:
            self.skipTest(
                "Set PYTHONPATH to companion machine_b for optional loopback control"
            )

        async def check():
            with (
                tempfile.TemporaryDirectory() as local_dir,
                tempfile.TemporaryDirectory() as remote_dir,
            ):
                local_path = write_core_config(local_dir, level="int4")
                remote_path = write_core_config(remote_dir, level="int4")
                server = grpc.aio.server()
                rpc.add_EvicPressServiceServicer_to_server(
                    EvicPressServicer(EvicPressManager(load_config(str(remote_path)))),
                    server,
                )
                port = server.add_insecure_port("127.0.0.1:0")
                await server.start()

                def exercise():
                    cfg = backend_config(local_path)
                    remote_cfg = settings()
                    remote_cfg.extra_config = {"grpc_server": f"127.0.0.1:{port}"}
                    local_pool, remote_pool = CPUBackend(), CPUBackend()
                    source = allocation(local_pool)
                    local = LocalBackend(
                        config=cfg, metadata=metadata(), local_cpu_backend=local_pool
                    )
                    remote = RemoteBackend(
                        config=remote_cfg,
                        metadata=metadata(),
                        local_cpu_backend=remote_pool,
                    )
                    try:
                        key = Key(tags=(("sample", "parity"),))
                        for backend in (local, remote):
                            backend.batched_submit_put_task([key], [source])
                            self.assertTrue(backend.contains(key))
                        first, second = (
                            local.get_blocking(key),
                            remote.get_blocking(key),
                        )
                        self.assertTrue(torch.equal(first.tensor, second.tensor))
                        first.ref_count_down()
                        second.ref_count_down()
                        self.assertTrue(local.remove(key))
                        self.assertTrue(remote.remove(key))
                    finally:
                        local.close()
                        remote.close()
                        source.ref_count_down()
                    self.assertEqual((local_pool.used, remote_pool.used), (0, 0))

                try:
                    await asyncio.to_thread(exercise)
                finally:
                    await server.stop(0)

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
