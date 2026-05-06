import ast
import textwrap
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
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


class _FakeCaptureHiddenMode:
    LAST = "last"


class _FakeSpecAlgorithm:
    def supports_spec_v2(self):
        return True


class _FakeEagleDraftInput:
    def __init__(
        self,
        topk_p,
        topk_index,
        hidden_states,
        verified_id,
        new_seq_lens,
    ):
        self.topk_p = topk_p
        self.topk_index = topk_index
        self.hidden_states = hidden_states
        self.verified_id = verified_id
        self.new_seq_lens = new_seq_lens
        self.capture_hidden_mode = None
        self.future_indices = None

    def prepare_for_extend(self, batch):
        pt = 0
        for i, extend_len in enumerate(batch.extend_lens):
            input_ids = batch.input_ids[pt : pt + extend_len]
            batch.input_ids[pt : pt + extend_len] = torch.cat(
                (input_ids[1:], self.verified_id[i].reshape(1))
            )
            pt += extend_len


_PROCESS_BATCH_RESULT_DISAGG_PREFILL = _extract_class_method(
    "python/sglang/srt/disaggregation/prefill.py",
    "SchedulerDisaggregationPrefillMixin",
    "process_batch_result_disagg_prefill",
    {
        "release_kv_cache": lambda *_args, **_kwargs: None,
        "prepare_abort": lambda *_args, **_kwargs: None,
        "HTTPStatus": SimpleNamespace(INTERNAL_SERVER_ERROR=500),
    },
)

_PROCESS_PREBUILT = _extract_class_method(
    "python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py",
    "ScheduleBatchDisaggregationDecodeMixin",
    "process_prebuilt",
    {
        "torch": torch,
        "CaptureHiddenMode": _FakeCaptureHiddenMode,
    },
)


class TestSpecMetadataChain(CustomTestCase):
    def _new_req(self):
        return SimpleNamespace(
            output_ids=[],
            is_chunked=0,
            return_logprob=False,
            time_stats=SimpleNamespace(
                set_prefill_finished_time=MagicMock(),
                set_prefill_transfer_queue_entry_time=MagicMock(),
            ),
            grammar=None,
            output_topk_p=None,
            output_topk_index=None,
            hidden_states_tensor=None,
            finished=MagicMock(return_value=False),
        )

    def test_prefill_metadata_flows_into_decode_spec_info(self):
        reqs = [self._new_req(), self._new_req()]
        prefill_scheduler = SimpleNamespace(
            spec_algorithm=_FakeSpecAlgorithm(),
            tree_cache=SimpleNamespace(cache_unfinished_req=MagicMock()),
            disagg_prefill_inflight_queue=[],
            send_kv_chunk=MagicMock(),
            add_logprob_return_values=MagicMock(),
            report_prefill_stats=MagicMock(),
        )
        batch_spec_info = SimpleNamespace(
            topk_p=torch.tensor([[0.7, 0.2], [0.6, 0.3]], dtype=torch.float32),
            topk_index=torch.tensor([[101, 102], [201, 202]], dtype=torch.int64),
            hidden_states=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        )
        prefill_batch = SimpleNamespace(
            reqs=reqs,
            return_logprob=False,
            spec_info=batch_spec_info,
            prefill_stats="prefill-stats",
            dp_cooperation_info="dp-info",
        )
        result = SimpleNamespace(
            logits_output=SimpleNamespace(
                next_token_logprobs=None,
                input_token_logprobs=None,
            ),
            next_token_ids=torch.tensor([11, 22], dtype=torch.int64),
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            copy_done=None,
            routed_experts_output=None,
        )

        _PROCESS_BATCH_RESULT_DISAGG_PREFILL(prefill_scheduler, prefill_batch, result)

        self.assertEqual(reqs[0].output_ids, [11])
        self.assertEqual(reqs[1].output_ids, [22])
        self.assertEqual(len(prefill_scheduler.disagg_prefill_inflight_queue), 2)
        self.assertTrue(torch.equal(reqs[0].output_topk_p, batch_spec_info.topk_p[0]))
        self.assertTrue(torch.equal(reqs[1].output_topk_index, batch_spec_info.topk_index[1]))
        self.assertTrue(
            torch.equal(reqs[0].hidden_states_tensor, batch_spec_info.hidden_states[0])
        )
        prefill_scheduler.send_kv_chunk.assert_any_call(reqs[0], last_chunk=True)
        prefill_scheduler.send_kv_chunk.assert_any_call(reqs[1], last_chunk=True)
        prefill_scheduler.report_prefill_stats.assert_called_once_with(
            prefill_stats="prefill-stats",
            can_run_cuda_graph=False,
            dp_cooperation_info="dp-info",
        )

        decode_batch = SimpleNamespace(
            reqs=reqs,
            tree_cache=SimpleNamespace(cache_unfinished_req=MagicMock()),
            device="cpu",
            spec_algorithm=_FakeSpecAlgorithm(),
            seq_lens=torch.tensor([5, 7], dtype=torch.int32),
            extend_lens=[2, 3],
            input_ids=torch.tensor([100, 101, 200, 201, 202], dtype=torch.int64),
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            enable_overlap=False,
        )
        server_args = SimpleNamespace(
            speculative_eagle_topk=2,
            enable_multi_layer_eagle=False,
            speculative_num_steps=3,
        )
        future_map = MagicMock()

        fake_eagle_info = ModuleType("sglang.srt.speculative.eagle_info")
        fake_eagle_info.EagleDraftInput = _FakeEagleDraftInput

        with patch.dict(
            "sys.modules",
            {"sglang.srt.speculative.eagle_info": fake_eagle_info},
        ):
            _PROCESS_PREBUILT(decode_batch, server_args, future_map)

        self.assertTrue(torch.equal(decode_batch.output_ids, torch.tensor([11, 22])))
        self.assertIsInstance(decode_batch.spec_info, _FakeEagleDraftInput)
        self.assertTrue(
            torch.equal(decode_batch.spec_info.topk_p, batch_spec_info.topk_p)
        )
        self.assertTrue(
            torch.equal(decode_batch.spec_info.topk_index, batch_spec_info.topk_index)
        )
        self.assertTrue(
            torch.equal(
                decode_batch.spec_info.hidden_states,
                torch.stack([req.hidden_states_tensor for req in reqs], dim=0),
            )
        )
        self.assertTrue(
            torch.equal(decode_batch.spec_info.verified_id, torch.tensor([11, 22]))
        )
        self.assertEqual(decode_batch.spec_info.capture_hidden_mode, _FakeCaptureHiddenMode.LAST)
        self.assertTrue(
            torch.equal(
                decode_batch.input_ids,
                torch.tensor([101, 11, 201, 202, 22], dtype=torch.int64),
            )
        )
        future_map.alloc_future_indices.assert_not_called()
        future_map.store_to_map_for_new_batch.assert_not_called()

    def test_process_prebuilt_overlap_allocates_future_indices(self):
        req = self._new_req()
        req.output_ids = [33]
        req.output_topk_p = torch.tensor([0.9, 0.1], dtype=torch.float32)
        req.output_topk_index = torch.tensor([301, 302], dtype=torch.int64)
        req.hidden_states_tensor = torch.tensor([5.0, 6.0], dtype=torch.float32)

        decode_batch = SimpleNamespace(
            reqs=[req],
            tree_cache=SimpleNamespace(cache_unfinished_req=MagicMock()),
            device="cpu",
            spec_algorithm=_FakeSpecAlgorithm(),
            seq_lens=torch.tensor([9], dtype=torch.int32),
            extend_lens=[2],
            input_ids=torch.tensor([300, 301], dtype=torch.int64),
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            enable_overlap=True,
        )
        server_args = SimpleNamespace(
            speculative_eagle_topk=2,
            enable_multi_layer_eagle=False,
            speculative_num_steps=4,
        )
        future_map = MagicMock()
        future_map.alloc_future_indices.return_value = "future-slot"

        fake_eagle_info = ModuleType("sglang.srt.speculative.eagle_info")
        fake_eagle_info.EagleDraftInput = _FakeEagleDraftInput

        with patch.dict(
            "sys.modules",
            {"sglang.srt.speculative.eagle_info": fake_eagle_info},
        ):
            _PROCESS_PREBUILT(decode_batch, server_args, future_map)

        self.assertEqual(decode_batch.spec_info.future_indices, "future-slot")
        future_map.alloc_future_indices.assert_called_once_with(1)
        future_map.store_to_map_for_new_batch.assert_called_once_with(
            "future-slot", decode_batch.spec_info
        )


if __name__ == "__main__":
    unittest.main()