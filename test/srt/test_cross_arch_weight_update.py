"""
Test cross-architecture weight update behavior.

This test documents that update_weights_from_disk only supports same-architecture
model switching. When attempting to switch between models with different architectures
(e.g., different hidden_size, num_layers), the current implementation crashes the
scheduler process instead of returning a graceful error.

A proper cross-architecture reload API (reload_model) is tracked in RFC #29363.
"""

import unittest

import requests

import sglang as sgl
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Llama-3.2-1B and Qwen2.5-1.5B have different architectures:
# - Different hidden_size (2048 vs 1536)
# - Different num_attention_heads (32 vs 12)
# - Different num_layers (16 vs 28)
MODEL_A = DEFAULT_SMALL_MODEL_NAME_FOR_TEST  # meta-llama/Llama-3.2-1B-Instruct
MODEL_B = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN  # Qwen/Qwen2.5-1.5B-Instruct


class TestCrossArchWeightUpdateServer(CustomTestCase):
    """Test that cross-architecture update_weights_from_disk is not silently accepted.

    Currently, cross-arch updates crash the scheduler process because
    load_weights hits a shape mismatch (e.g., hidden_size 2048 vs 1536) and
    the rollback path also fails. This test documents that behavior.

    Once reload_model (RFC #29363) is implemented, this test should be updated
    to verify that cross-arch switching works correctly via the new API.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_A
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        return response.json()["text"]

    def test_cross_arch_update_does_not_succeed(self):
        """Cross-arch update should not silently succeed with corrupted weights.

        The acceptable outcomes are:
        1. Return {"success": false} with an error message (ideal, not yet implemented)
        2. Crash/disconnect (current behavior — documents the limitation)

        The unacceptable outcome is returning {"success": true} because the
        loaded weights would have wrong shapes, causing silent corruption.
        """
        origin_response = self.run_decode()
        self.assertTrue(len(origin_response) > 0, "Initial generation should work")

        try:
            response = requests.post(
                self.base_url + "/update_weights_from_disk",
                json={"model_path": MODEL_B},
                timeout=60,
            )
            ret = response.json()
            # If we get here, the server returned a response instead of crashing.
            # The only acceptable outcome is success=false.
            self.assertFalse(
                ret["success"],
                f"Cross-architecture weight update must not return success=True. "
                f"Attempted: {MODEL_A} -> {MODEL_B}. "
                f"This would mean weights with mismatched shapes were loaded.",
            )
            print(f"[Expected] Cross-arch update failed gracefully: {ret}")
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ):
            # Server crashed — this is the current expected behavior.
            # The scheduler process dies on shape mismatch during load_weights.
            print(
                "[Expected] Server crashed on cross-arch update "
                "(shape mismatch in load_weights). "
                "This documents the need for reload_model API (RFC #29363)."
            )


if __name__ == "__main__":
    unittest.main()
