import ast
import sys
import textwrap
import unittest
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

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


@dataclass
class PPBatchMetadata:
    can_run_cuda_graph: bool


class PPProxyTensors:
    def __init__(self, tensors):
        self.tensors = tensors


class _FakeForwardModeValue:
    def __init__(self, name: str):
        self.name = name

    def is_prebuilt(self):
        return self.name == "prebuilt"


class _FakeForwardMode:
    EXTEND = _FakeForwardModeValue("extend")


class _FakeCaptureHiddenMode:
    FULL = "full"


class _FakeForwardBatch:
    last_init_args = None

    @staticmethod
    def init_new(model_worker_batch, model_runner):
        _FakeForwardBatch.last_init_args = (model_worker_batch, model_runner)
        return "forward-batch"


class _FakeDisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


class _FakeTransferBackend(Enum):
    ASCEND = "ascend"


class _FakeReqToMetadataIdxAllocator:
    def __init__(self, size):
        self.size = size


class _FakeMetadataBuffers:
    last_init = None

    def __init__(
        self,
        size,
        hidden_size,
        hidden_states_dtype,
        max_spec_topk_num=16,
        max_top_logprobs_num=128,
        custom_mem_pool=None,
    ):
        _FakeMetadataBuffers.last_init = SimpleNamespace(
            size=size,
            hidden_size=hidden_size,
            hidden_states_dtype=hidden_states_dtype,
            max_spec_topk_num=max_spec_topk_num,
            max_top_logprobs_num=max_top_logprobs_num,
            custom_mem_pool=custom_mem_pool,
        )


class _FakePrefillBootstrapQueue:
    last_init_kwargs = None

    def __init__(self, **kwargs):
        _FakePrefillBootstrapQueue.last_init_kwargs = kwargs


_GLOBAL_SERVER_ARGS = SimpleNamespace(pp_max_micro_batch_size=None)
_GROUP = SimpleNamespace(cpu_group=MagicMock())
_LOGGER = MagicMock()

_CHECK_SERVER_ARGS = _extract_class_method(
    "python/sglang/srt/server_args.py",
    "ServerArgs",
    "check_server_args",
    {
        "is_runai_obj_uri": lambda _name: False,
        "logger": _LOGGER,
        "torch_release": (0, 0),
        "is_npu": lambda: False,
    },
)

_MAYBE_INIT_DRAFT_WORKER = _extract_class_method(
    "python/sglang/srt/managers/scheduler.py",
    "Scheduler",
    "maybe_init_draft_worker",
    {
        "logger": _LOGGER,
    },
)

_INIT_MODEL_WORKER = _extract_class_method(
    "python/sglang/srt/managers/scheduler.py",
    "Scheduler",
    "init_model_worker",
    {
        "get_global_server_args": lambda: _GLOBAL_SERVER_ARGS,
        "get_tp_group": lambda: _GROUP,
        "get_attention_tp_group": lambda: _GROUP,
        "get_attention_cp_group": lambda: _GROUP,
        "get_pp_group": lambda: _GROUP,
        "get_world_group": lambda: _GROUP,
        "set_random_seed": lambda _seed: None,
        "get_available_gpu_memory": lambda *_args, **_kwargs: 0.0,
        "logger": _LOGGER,
    },
)

_PP_LAUNCH_BATCH = _extract_class_method(
    "python/sglang/srt/managers/scheduler_pp_mixin.py",
    "SchedulerPPMixin",
    "_pp_launch_batch",
    {
        "torch": SimpleNamespace(
            profiler=SimpleNamespace(record_function=lambda _name: nullcontext())
        ),
        "PPBatchMetadata": PPBatchMetadata,
        "PPProxyTensors": PPProxyTensors,
    },
)

_RUN_BATCH = _extract_class_method(
    "python/sglang/srt/managers/scheduler.py",
    "Scheduler",
    "run_batch",
    {
        "logger": _LOGGER,
        "time": SimpleNamespace(sleep=lambda _seconds: None),
        "set_time_batch": lambda *_args, **_kwargs: None,
        "ForwardMode": _FakeForwardMode,
    },
)

_FORWARD_BATCH_GENERATION = _extract_class_method(
    "python/sglang/srt/managers/tp_worker.py",
    "TpModelWorker",
    "forward_batch_generation",
    {
        "ForwardBatch": _FakeForwardBatch,
        "CaptureHiddenMode": _FakeCaptureHiddenMode,
    },
)

_VALIDATE_MTP_PP_COMPATIBILITY = _extract_class_method(
    "python/sglang/srt/model_executor/model_runner.py",
    "ModelRunner",
    "_validate_mtp_pp_compatibility",
    {},
)

_INIT_DISAGGREGATION = _extract_class_method(
    "python/sglang/srt/managers/scheduler.py",
    "Scheduler",
    "init_disaggregation",
    {
        "DisaggregationMode": _FakeDisaggregationMode,
        "TransferBackend": _FakeTransferBackend,
        "ReqToMetadataIdxAllocator": _FakeReqToMetadataIdxAllocator,
        "MetadataBuffers": _FakeMetadataBuffers,
        "PrefillBootstrapQueue": _FakePrefillBootstrapQueue,
        "torch": SimpleNamespace(float32="float32"),
    },
)


class TestPDPrefillPPMTPSupport(CustomTestCase):
    def _new_scheduler_like(self, pp_rank: int, spec_algorithm: SpeculativeAlgorithm):
        return SimpleNamespace(
            spec_algorithm=spec_algorithm,
            pp_size=2,
            pp_rank=pp_rank,
            gpu_id=0,
            tp_rank=0,
            moe_ep_rank=0,
            nccl_port=1234,
            dp_rank=0,
            attn_cp_rank=0,
            moe_dp_rank=0,
            tp_worker=MagicMock(),
            send_to_tokenizer=SimpleNamespace(send_output=MagicMock()),
            server_args=SimpleNamespace(
                speculative_draft_load_format=None,
                disaggregation_mode="prefill",
                enable_dp_attention=False,
                chunked_prefill_size=-1,
            ),
        )

    def _new_server_args_for_check(self, disaggregation_mode: str, speculative_algorithm):
        args = SimpleNamespace(
            tp_size=1,
            pp_size=2,
            nnodes=1,
            pp_max_micro_batch_size=None,
            disable_overlap_schedule=True,
            speculative_algorithm=speculative_algorithm,
            disaggregation_mode=disaggregation_mode,
            dp_size=1,
            enable_dp_attention=False,
            base_gpu_id=0,
            gpu_id_step=1,
            moe_dense_tp_size=1,
            served_model_name="dummy",
            enable_mixed_chunk=False,
            chunked_prefill_size=-1,
            page_size=1,
            enable_pdmux=False,
            tokenizer_worker_num=1,
            prompt_tokens_buckets=[],
            generation_tokens_buckets=[],
            enable_priority_scheduling=False,
            disable_priority_preemption=False,
            default_priority_value=None,
            enable_hisparse=False,
            schedule_conservativeness=0,
            model_impl="sglang",
            tokenizer_metrics_custom_labels_header=None,
            tokenizer_metrics_allowed_custom_labels=[],
            export_metrics_to_file=False,
            export_metrics_to_file_dir=None,
            enable_two_batch_overlap=False,
            moe_a2a_backend="flashinfer",
            enable_grpc=False,
            grpc_port=None,
            port=30000,
            gc_threshold=None,
        )
        args.check_lora_server_args = MagicMock()
        args.validate_buckets_rule = MagicMock()
        return args

    def test_pp_spec_requires_disaggregated_prefill(self):
        valid_args = self._new_server_args_for_check("prefill", "EAGLE")
        _CHECK_SERVER_ARGS(valid_args)

        invalid_args = self._new_server_args_for_check("null", "STANDALONE")
        with self.assertRaisesRegex(
            AssertionError,
            "PP \\+ speculative decoding is only supported in disaggregated prefill mode",
        ):
            _CHECK_SERVER_ARGS(invalid_args)

    def test_create_worker_supports_multi_layer_and_standalone(self):
        class FakeMultiLayerWorker:
            pass

        class FakeStandaloneWorker:
            pass

        multi_layer_module = ModuleType(
            "sglang.srt.speculative.multi_layer_eagle_worker"
        )
        multi_layer_module.MultiLayerEagleWorker = FakeMultiLayerWorker

        standalone_module = ModuleType("sglang.srt.speculative.standalone_worker")
        standalone_module.StandaloneWorker = FakeStandaloneWorker

        with patch.dict(
            sys.modules,
            {
                "sglang.srt.speculative.multi_layer_eagle_worker": multi_layer_module,
                "sglang.srt.speculative.standalone_worker": standalone_module,
            },
        ):
            multi_layer_args = SimpleNamespace(
                disable_overlap_schedule=True,
                enable_multi_layer_eagle=True,
            )
            standalone_args = SimpleNamespace(
                disable_overlap_schedule=True,
                enable_multi_layer_eagle=False,
            )

            self.assertIs(
                SpeculativeAlgorithm.EAGLE.create_worker(multi_layer_args),
                FakeMultiLayerWorker,
            )
            self.assertIs(
                SpeculativeAlgorithm.STANDALONE.create_worker(standalone_args),
                FakeStandaloneWorker,
            )

    @patch.object(SpeculativeAlgorithm, "create_worker")
    def test_only_last_pp_rank_holds_draft_worker(self, mock_create_worker):
        class FakeDraftWorker:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        mock_create_worker.return_value = FakeDraftWorker

        for spec_algorithm in (
            SpeculativeAlgorithm.EAGLE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.STANDALONE,
        ):
            non_last_scheduler = self._new_scheduler_like(0, spec_algorithm)
            _MAYBE_INIT_DRAFT_WORKER(non_last_scheduler)
            self.assertIsNone(non_last_scheduler.draft_worker)

            last_scheduler = self._new_scheduler_like(1, spec_algorithm)
            _MAYBE_INIT_DRAFT_WORKER(last_scheduler)
            self.assertIsInstance(last_scheduler.draft_worker, FakeDraftWorker)
            self.assertIsNone(last_scheduler.external_corpus_manager)

        self.assertEqual(mock_create_worker.call_count, 3)

    def test_non_last_pp_rank_dispatches_tp_worker(self):
        _GLOBAL_SERVER_ARGS.pp_max_micro_batch_size = None

        tp_worker = MagicMock(name="tp_worker")
        tp_worker.get_worker_info.return_value = (
            128,
            64,
            32,
            16,
            4096,
            4096,
            1,
            "cpu",
            MagicMock(name="forward_stream"),
            None,
            None,
            None,
        )
        tp_worker.get_pad_input_ids_func.return_value = MagicMock()

        scheduler = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            tp_worker=tp_worker,
            draft_worker=None,
            init_tp_model_worker=MagicMock(),
            maybe_init_draft_worker=MagicMock(),
            pp_size=2,
            server_args=SimpleNamespace(enable_dp_attention=False, chunked_prefill_size=-1),
            enable_metrics=False,
            tp_rank=1,
            model_config=SimpleNamespace(context_len=4096),
            page_size=1,
            gpu_id=0,
        )

        _INIT_MODEL_WORKER(scheduler)

        self.assertIs(scheduler.model_worker, tp_worker)
        scheduler.init_tp_model_worker.assert_called_once_with()
        scheduler.maybe_init_draft_worker.assert_called_once_with()

    def test_run_batch_uses_tp_worker_for_pp_prefill_spec(self):
        batch_result = SimpleNamespace(
            next_token_ids=[7, 8],
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )
        batch = SimpleNamespace(
            forward_mode=_FakeForwardMode.EXTEND,
            reqs=[],
            return_logprob=False,
            get_model_worker_batch=MagicMock(return_value="worker-batch"),
            output_ids=None,
        )
        scheduler = SimpleNamespace(
            forward_ct=0,
            _profile_batch_predicate=MagicMock(),
            forward_sleep_time=None,
            is_generation=True,
            spec_algorithm=SimpleNamespace(is_none=lambda: False),
            enable_overlap=False,
            pp_size=2,
            enable_pdmux=False,
            server_args=SimpleNamespace(
                disaggregation_mode="prefill",
                enable_dp_attention=False,
                elastic_ep_backend=None,
            ),
            tp_worker=MagicMock(),
            model_worker=MagicMock(),
            record_forward_metrics=lambda _batch: nullcontext(),
            update_cache_from_scheduler=MagicMock(),
        )
        scheduler.tp_worker.forward_batch_generation.return_value = batch_result

        result = _RUN_BATCH(scheduler, batch, pp_proxy_tensors="pp-proxy")

        self.assertIs(result, batch_result)
        batch.get_model_worker_batch.assert_called_once_with()
        scheduler.tp_worker.forward_batch_generation.assert_called_once_with(
            "worker-batch",
            pp_proxy_tensors="pp-proxy",
        )
        scheduler.model_worker.forward_batch_generation.assert_not_called()
        scheduler.update_cache_from_scheduler.assert_called_once_with(batch, batch_result)
        self.assertEqual(batch.output_ids, [7, 8])

    def test_tp_worker_capture_full_hidden_states_for_pp_prefill_spec(self):
        model_worker_batch = SimpleNamespace(
            hicache_consumer_index=5,
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            capture_hidden_mode=None,
        )
        worker = SimpleNamespace(
            enable_spec=True,
            set_hicache_consumer=MagicMock(),
            model_runner="target-model-runner",
            is_dllm=lambda: True,
            _forward_batch_generation_dllm=MagicMock(return_value="dllm-result"),
        )

        result = _FORWARD_BATCH_GENERATION(worker, model_worker_batch)

        self.assertEqual(result, "dllm-result")
        worker.set_hicache_consumer.assert_called_once_with(5)
        self.assertEqual(model_worker_batch.capture_hidden_mode, _FakeCaptureHiddenMode.FULL)
        self.assertEqual(
            _FakeForwardBatch.last_init_args,
            (model_worker_batch, "target-model-runner"),
        )

    def test_model_runner_allows_pp_for_prefill_target_with_mtp(self):
        model_runner = SimpleNamespace(
            is_draft_worker=False,
            pp_size=2,
            server_args=SimpleNamespace(disaggregation_mode="prefill"),
            spec_algorithm=SimpleNamespace(is_none=lambda: False),
            num_effective_layers=30,
        )

        _VALIDATE_MTP_PP_COMPATIBILITY(
            model_runner,
            model_has_mtp_layers=True,
            model_num_layers=61,
        )

    def test_model_runner_still_rejects_partitioned_mtp_draft_worker(self):
        model_runner = SimpleNamespace(
            is_draft_worker=True,
            pp_size=1,
            server_args=SimpleNamespace(disaggregation_mode="prefill"),
            spec_algorithm=SimpleNamespace(is_none=lambda: False),
            num_effective_layers=1,
        )

        with self.assertRaisesRegex(
            AssertionError,
            "PP is not compatible with MTP models.",
        ):
            _VALIDATE_MTP_PP_COMPATIBILITY(
                model_runner,
                model_has_mtp_layers=True,
                model_num_layers=2,
            )

    def test_model_runner_still_rejects_non_prefill_partitioned_mtp_target(self):
        model_runner = SimpleNamespace(
            is_draft_worker=False,
            pp_size=2,
            server_args=SimpleNamespace(disaggregation_mode="decode"),
            spec_algorithm=SimpleNamespace(is_none=lambda: False),
            num_effective_layers=30,
        )

        with self.assertRaisesRegex(
            AssertionError,
            "PP is not compatible with MTP models.",
        ):
            _VALIDATE_MTP_PP_COMPATIBILITY(
                model_runner,
                model_has_mtp_layers=True,
                model_num_layers=61,
            )

    def test_init_disaggregation_prefill_pp_spec_without_local_draft_worker(self):
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                disaggregation_mode="prefill",
                disaggregation_transfer_backend="ascend",
                enable_multi_layer_eagle=False,
                speculative_eagle_topk=4,
                speculative_num_steps=3,
                disaggregation_bootstrap_port=8995,
                dp_size=1,
                language_only=False,
            ),
            draft_worker=None,
            enable_overlap=False,
            spec_algorithm=SimpleNamespace(
                is_ngram=lambda: False,
                supports_spec_v2=lambda: True,
                is_eagle=lambda: True,
                is_standalone=lambda: False,
            ),
            max_running_requests=1,
            model_config=SimpleNamespace(hidden_size=7168, dtype="bf16"),
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: SimpleNamespace(
                    maybe_get_custom_mem_pool=lambda: "custom-pool"
                )
            ),
            tp_rank=0,
            tp_size=8,
            gpu_id=0,
            attn_tp_cpu_group="gloo-group",
            max_total_num_tokens=257408,
            pp_rank=0,
            pp_size=2,
        )

        _INIT_DISAGGREGATION(scheduler)

        self.assertEqual(
            _FakeMetadataBuffers.last_init.hidden_size,
            scheduler.model_config.hidden_size,
        )
        self.assertEqual(
            _FakeMetadataBuffers.last_init.hidden_states_dtype,
            scheduler.model_config.dtype,
        )
        self.assertEqual(
            _FakePrefillBootstrapQueue.last_init_kwargs["pp_size"],
            scheduler.pp_size,
        )
        self.assertEqual(_FakeMetadataBuffers.last_init.max_spec_topk_num, 4)

    def test_init_disaggregation_prefill_multilayer_expands_metadata_topk_width(self):
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                disaggregation_mode="prefill",
                disaggregation_transfer_backend="ascend",
                enable_multi_layer_eagle=True,
                speculative_eagle_topk=4,
                speculative_num_steps=5,
                disaggregation_bootstrap_port=8995,
                dp_size=1,
                language_only=False,
            ),
            draft_worker=None,
            enable_overlap=False,
            spec_algorithm=SimpleNamespace(
                is_ngram=lambda: False,
                supports_spec_v2=lambda: True,
                is_eagle=lambda: True,
                is_standalone=lambda: False,
            ),
            max_running_requests=1,
            model_config=SimpleNamespace(hidden_size=7168, dtype="bf16"),
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: SimpleNamespace(
                    maybe_get_custom_mem_pool=lambda: "custom-pool"
                )
            ),
            tp_rank=0,
            tp_size=8,
            gpu_id=0,
            attn_tp_cpu_group="gloo-group",
            max_total_num_tokens=257408,
            pp_rank=0,
            pp_size=2,
        )

        _INIT_DISAGGREGATION(scheduler)

        self.assertEqual(_FakeMetadataBuffers.last_init.max_spec_topk_num, 20)

    def test_last_pp_rank_runs_draft_extend_after_target_forward(self):
        scheduler = SimpleNamespace(
            forward_stream_ctx=nullcontext(),
            forward_stream=MagicMock(),
            schedule_stream=MagicMock(),
            run_batch=MagicMock(),
            pp_group=SimpleNamespace(is_last_rank=True),
            spec_algorithm=SpeculativeAlgorithm.STANDALONE,
            server_args=SimpleNamespace(disaggregation_mode="prefill"),
            draft_worker=MagicMock(),
            device_module=MagicMock(),
            _pp_prepare_tensor_dict=MagicMock(return_value={"hidden": 1}),
        )
        scheduler.device_module.Event.return_value = MagicMock()
        scheduler.device_module.current_stream.return_value = MagicMock()
        scheduler.cur_batch = MagicMock()
        scheduler.cur_batch.forward_mode.is_extend.return_value = True
        scheduler.cur_batch.is_extend_in_batch = False

        result = MagicMock()
        result.can_run_cuda_graph = False
        result.next_token_ids = [1, 2, 3]
        result.logits_output = SimpleNamespace(
            hidden_states="hidden_states",
            mm_input_embeds="mm_input_embeds",
        )
        scheduler.run_batch.return_value = result

        last_rank_comm_queue = deque()
        mb_metadata = [None]

        _PP_LAUNCH_BATCH(
            scheduler,
            mb_id=0,
            pp_proxy_tensors=MagicMock(),
            mb_metadata=mb_metadata,
            last_rank_comm_queue=last_rank_comm_queue,
        )

        scheduler.draft_worker.run_draft_extend_for_pp_prefill.assert_called_once_with(
            scheduler.cur_batch,
            "hidden_states",
            [1, 2, 3],
            "mm_input_embeds",
        )
        self.assertEqual(len(last_rank_comm_queue), 1)

    def test_last_pp_rank_skips_draft_extend_without_hidden_states(self):
        scheduler = SimpleNamespace(
            forward_stream_ctx=nullcontext(),
            forward_stream=MagicMock(),
            schedule_stream=MagicMock(),
            run_batch=MagicMock(),
            pp_group=SimpleNamespace(is_last_rank=True),
            spec_algorithm=SpeculativeAlgorithm.STANDALONE,
            server_args=SimpleNamespace(disaggregation_mode="prefill"),
            draft_worker=MagicMock(),
            device_module=MagicMock(),
            _pp_prepare_tensor_dict=MagicMock(return_value={"hidden": 1}),
        )
        scheduler.device_module.Event.return_value = MagicMock()
        scheduler.device_module.current_stream.return_value = MagicMock()
        scheduler.cur_batch = MagicMock()
        scheduler.cur_batch.forward_mode.is_extend.return_value = True
        scheduler.cur_batch.is_extend_in_batch = False

        result = MagicMock()
        result.can_run_cuda_graph = False
        result.next_token_ids = [1, 2, 3]
        result.logits_output = SimpleNamespace(
            hidden_states=None,
            mm_input_embeds="mm_input_embeds",
        )
        scheduler.run_batch.return_value = result

        _PP_LAUNCH_BATCH(
            scheduler,
            mb_id=0,
            pp_proxy_tensors=MagicMock(),
            mb_metadata=[None],
            last_rank_comm_queue=deque(),
        )

        scheduler.draft_worker.run_draft_extend_for_pp_prefill.assert_not_called()


if __name__ == "__main__":
    unittest.main()