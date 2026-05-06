import ast
from contextlib import nullcontext
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="stage-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]


def _extract_class_method(rel_path: str, class_name: str, method_name: str, globals_dict):
    source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    module_ast = ast.parse(source)
    for node in module_ast.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    cloned = ast.FunctionDef(
                        name=item.name,
                        args=item.args,
                        body=item.body,
                        decorator_list=[],
                        returns=item.returns,
                        type_comment=item.type_comment,
                    )
                    ast.fix_missing_locations(cloned)
                    namespace = dict(globals_dict)
                    exec(
                        "from __future__ import annotations\n"
                        + textwrap.dedent(ast.unparse(cloned)),
                        namespace,
                    )
                    return namespace[method_name]
    raise ValueError(f"Failed to find {class_name}.{method_name} in {rel_path}")


def _extract_class_with_method(
    rel_path: str,
    class_name: str,
    method_name: str,
    globals_dict,
    base_class,
    extracted_class_name: str,
):
    source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    module_ast = ast.parse(source)
    for node in module_ast.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    cloned = ast.FunctionDef(
                        name=item.name,
                        args=item.args,
                        body=item.body,
                        decorator_list=[],
                        returns=item.returns,
                        type_comment=item.type_comment,
                    )
                    class_def = ast.ClassDef(
                        name=extracted_class_name,
                        bases=[ast.Name(id="_Base", ctx=ast.Load())],
                        keywords=[],
                        body=[cloned],
                        decorator_list=[],
                    )
                    module = ast.Module(body=[class_def], type_ignores=[])
                    ast.fix_missing_locations(module)
                    namespace = dict(globals_dict)
                    namespace["_Base"] = base_class
                    exec(
                        "from __future__ import annotations\n"
                        + textwrap.dedent(ast.unparse(module)),
                        namespace,
                    )
                    return namespace[extracted_class_name]
    raise ValueError(f"Failed to find {class_name}.{method_name} in {rel_path}")


def _fake_empty_context(*_args, **_kwargs):
    return nullcontext()


class _FakeSpeculativeAlgorithm:
    class _Value:
        def __init__(self, name):
            self.name = name

        def is_eagle3(self):
            return self.name == "EAGLE3"

    @staticmethod
    def from_string(name):
        return _FakeSpeculativeAlgorithm._Value(name)


class _FakeTpModelWorker:
    def __init__(self, **_kwargs):
        pass


_STANDALONE_INIT = _extract_class_method(
    "python/sglang/srt/speculative/standalone_worker.py",
    "StandaloneWorker",
    "__init__",
    {
        "torch": torch,
        "TpModelWorker": _FakeTpModelWorker,
        "SpeculativeAlgorithm": _FakeSpeculativeAlgorithm,
        "draft_tp_context": _fake_empty_context,
        "load_token_map": lambda path: [path],
        "empty_context": _fake_empty_context,
        "speculative_moe_backend_context": _fake_empty_context,
        "speculative_moe_a2a_backend_context": _fake_empty_context,
    },
)

_RUN_DRAFT_EXTEND_FOR_PP_PREFILL = _extract_class_method(
    "python/sglang/srt/speculative/eagle_worker.py",
    "EAGLEWorker",
    "run_draft_extend_for_pp_prefill",
    {
        "speculative_moe_backend_context": _fake_empty_context,
        "speculative_moe_a2a_backend_context": _fake_empty_context,
    },
)

_ExtractedMultiLayerWorker = _extract_class_with_method(
    "python/sglang/srt/speculative/multi_layer_eagle_worker.py",
    "MultiLayerEagleWorker",
    "__init__",
    {
        "torch": torch,
        "TpModelWorker": _FakeTpModelWorker,
        "SpeculativeAlgorithm": _FakeSpeculativeAlgorithm,
        "draft_tp_context": _fake_empty_context,
        "get_attention_tp_group": lambda: "attn-tp-group",
        "load_token_map": lambda path: [path],
        "empty_context": _fake_empty_context,
        "speculative_moe_backend_context": _fake_empty_context,
        "speculative_moe_a2a_backend_context": _fake_empty_context,
        "logger": MagicMock(),
    },
    base_class=_FakeTpModelWorker,
    extracted_class_name="_ExtractedMultiLayerWorker",
)

_MULTI_LAYER_RUN_DRAFT_EXTEND_FOR_PP_PREFILL = _extract_class_method(
    "python/sglang/srt/speculative/multi_layer_eagle_worker.py",
    "MultiLayerEagleWorker",
    "run_draft_extend_for_pp_prefill",
    {
        "speculative_moe_backend_context": _fake_empty_context,
        "speculative_moe_a2a_backend_context": _fake_empty_context,
    },
)


class _StandaloneWorkerUnderTest:
    __init__ = _STANDALONE_INIT
    run_draft_extend_for_pp_prefill = _RUN_DRAFT_EXTEND_FOR_PP_PREFILL

    def init_attention_backend(self):
        raise NotImplementedError()

    def init_cuda_graphs(self):
        raise NotImplementedError()


class _MultiLayerWorkerUnderTest(_ExtractedMultiLayerWorker):
    run_draft_extend_for_pp_prefill = _MULTI_LAYER_RUN_DRAFT_EXTEND_FOR_PP_PREFILL

    def init_attention_backend(self):
        raise NotImplementedError()

    def init_cuda_graphs(self):
        raise NotImplementedError()


class TestSpecWorkerPPPrefill(CustomTestCase):
    def _new_server_args(self):
        return SimpleNamespace(
            speculative_eagle_topk=4,
            speculative_num_steps=3,
            speculative_num_draft_tokens=5,
            gpu_id=0,
            device="cpu",
            page_size=16,
            speculative_algorithm="STANDALONE",
            disable_cuda_graph=False,
            speculative_token_map=None,
            json_model_override_args=None,
            pp_size=4,
            enable_dp_attention=False,
            context_length=None,
        )

    def _new_target_worker(self):
        return SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(context_len=8192),
                memory_pool_config="pool-config",
            ),
            get_memory_pool=MagicMock(return_value=("req-pool", "kv-pool")),
        )

    def _new_multi_layer_target_worker(self):
        return SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(context_len=8192),
                memory_pool_config="pool-config",
                model=SimpleNamespace(get_embed_and_head=MagicMock(return_value=("embed", "head"))),
            ),
            get_memory_pool=MagicMock(return_value=("req-pool", "kv-pool")),
        )

    def test_standalone_worker_init_keeps_draft_out_of_pp(self):
        server_args = self._new_server_args()
        target_worker = self._new_target_worker()
        init_calls = []

        def fake_tp_init(self, **kwargs):
            init_calls.append(
                {
                    "pp_size": kwargs["server_args"].pp_size,
                    "pp_rank": kwargs["pp_rank"],
                    "is_draft_worker": kwargs["is_draft_worker"],
                    "req_pool": kwargs["req_to_token_pool"],
                    "kv_pool": kwargs["token_to_kv_pool_allocator"],
                    "memory_pool_config": kwargs["memory_pool_config"],
                }
            )
            self.draft_model_runner = SimpleNamespace(
                server_args=SimpleNamespace(disable_cuda_graph=None),
                tp_group="draft-tp-group",
            )

        with (
            patch.object(_FakeTpModelWorker, "__init__", new=fake_tp_init),
            patch.object(_StandaloneWorkerUnderTest, "init_attention_backend", autospec=True),
            patch.object(_StandaloneWorkerUnderTest, "init_cuda_graphs", autospec=True),
        ):
            worker = _StandaloneWorkerUnderTest(
                server_args=server_args,
                gpu_id=0,
                tp_rank=1,
                dp_rank=None,
                moe_ep_rank=0,
                attn_cp_rank=0,
                moe_dp_rank=0,
                nccl_port=12345,
                target_worker=target_worker,
            )

        self.assertEqual(len(init_calls), 1)
        self.assertEqual(init_calls[0]["pp_size"], 1)
        self.assertEqual(init_calls[0]["pp_rank"], 0)
        self.assertTrue(init_calls[0]["is_draft_worker"])
        self.assertEqual(init_calls[0]["req_pool"], "req-pool")
        self.assertEqual(init_calls[0]["kv_pool"], "kv-pool")
        self.assertEqual(init_calls[0]["memory_pool_config"], "pool-config")
        self.assertEqual(server_args.pp_size, 4)
        self.assertEqual(server_args.context_length, 8192)
        self.assertFalse(worker.draft_model_runner.server_args.disable_cuda_graph)

    def test_pp_prefill_path_only_runs_forward_draft_extend(self):
        worker = _StandaloneWorkerUnderTest.__new__(_StandaloneWorkerUnderTest)
        worker.draft_model_runner = SimpleNamespace(tp_group="draft-tp-group")
        worker.draft_tp_context = lambda *_args, **_kwargs: nullcontext()
        worker.forward_draft_extend = MagicMock()
        worker.forward_draft_extend_after_decode = MagicMock()

        batch = SimpleNamespace(name="prefill-batch")
        hidden_states = MagicMock(name="hidden_states")
        next_token_ids = MagicMock(name="next_token_ids")
        mm_input_embeds = MagicMock(name="mm_input_embeds")

        worker.run_draft_extend_for_pp_prefill(
            batch,
            hidden_states,
            next_token_ids,
            mm_input_embeds,
        )

        worker.forward_draft_extend.assert_called_once_with(
            batch,
            hidden_states,
            next_token_ids,
            seq_lens_cpu=None,
            mm_input_embeds=mm_input_embeds,
        )
        worker.forward_draft_extend_after_decode.assert_not_called()

    def test_multi_layer_worker_init_keeps_draft_out_of_pp(self):
        server_args = self._new_server_args()
        server_args.speculative_algorithm = "EAGLE"
        target_worker = self._new_multi_layer_target_worker()
        init_calls = []

        def fake_tp_init(self, **kwargs):
            init_calls.append(
                {
                    "pp_size": kwargs["server_args"].pp_size,
                    "pp_rank": kwargs["pp_rank"],
                    "is_draft_worker": kwargs["is_draft_worker"],
                    "is_multi_layer_eagle": kwargs["is_multi_layer_eagle"],
                    "req_pool": kwargs["req_to_token_pool"],
                    "kv_pool": kwargs["token_to_kv_pool_allocator"],
                    "memory_pool_config": kwargs["memory_pool_config"],
                }
            )
            self._mtp_runners = [
                SimpleNamespace(
                    server_args=SimpleNamespace(disable_cuda_graph=None),
                    model=SimpleNamespace(set_embed_and_head=MagicMock()),
                    tp_group=f"mtp-group-{i}",
                )
                for i in range(kwargs["server_args"].speculative_num_steps)
            ]
            self.mtp_model_runner = lambda idx: self._mtp_runners[idx]

        with (
            patch.object(_FakeTpModelWorker, "__init__", new=fake_tp_init),
            patch.object(_MultiLayerWorkerUnderTest, "init_attention_backend", autospec=True),
            patch.object(_MultiLayerWorkerUnderTest, "init_cuda_graphs", autospec=True),
        ):
            worker = _MultiLayerWorkerUnderTest(
                server_args=server_args,
                gpu_id=0,
                tp_rank=1,
                dp_rank=None,
                moe_ep_rank=0,
                attn_cp_rank=0,
                moe_dp_rank=0,
                nccl_port=12345,
                target_worker=target_worker,
            )

        self.assertEqual(len(init_calls), 1)
        self.assertEqual(init_calls[0]["pp_size"], 1)
        self.assertEqual(init_calls[0]["pp_rank"], 0)
        self.assertTrue(init_calls[0]["is_draft_worker"])
        self.assertTrue(init_calls[0]["is_multi_layer_eagle"])
        self.assertEqual(init_calls[0]["req_pool"], "req-pool")
        self.assertEqual(init_calls[0]["kv_pool"], "kv-pool")
        self.assertEqual(init_calls[0]["memory_pool_config"], "pool-config")
        self.assertEqual(server_args.pp_size, 4)
        self.assertEqual(server_args.context_length, 8192)
        for mtp_runner in worker._mtp_runners:
            mtp_runner.model.set_embed_and_head.assert_called_once_with("embed", "head")
            self.assertFalse(mtp_runner.server_args.disable_cuda_graph)

    def test_multi_layer_pp_prefill_path_runs_forward_draft_extend(self):
        worker = _MultiLayerWorkerUnderTest.__new__(_MultiLayerWorkerUnderTest)
        worker.mtp_model_runner = lambda _idx: SimpleNamespace(tp_group="mtp-group-0")
        worker.draft_tp_context = lambda *_args, **_kwargs: nullcontext()
        worker.forward_draft_extend = MagicMock()

        batch = SimpleNamespace(name="prefill-batch")
        hidden_states = MagicMock(name="hidden_states")
        next_token_ids = MagicMock(name="next_token_ids")
        mm_input_embeds = MagicMock(name="mm_input_embeds")

        worker.run_draft_extend_for_pp_prefill(
            batch,
            hidden_states,
            next_token_ids,
            mm_input_embeds,
        )

        worker.forward_draft_extend.assert_called_once_with(
            batch,
            hidden_states,
            next_token_ids,
            seq_lens_cpu=None,
        )


if __name__ == "__main__":
    unittest.main()