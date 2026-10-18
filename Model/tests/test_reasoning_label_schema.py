"""Tests for the reasoning label schema.

Pure-Python, no torch/GPU/network. Covers:
    * abstained records carry the abstain marking and no horizons.
"""

from __future__ import annotations

from data_processing.reasoning_label_generation.schema import ReasoningLabelRecord


def test_abstained_record_carries_no_horizons():
    rec = ReasoningLabelRecord.abstain(
        sample_id="scene_x_t0",
        dataset_name="l2d",
        teacher_provider="openai_compatible",
        teacher_model="cosmos3-nano",
        prompt_version="action_relevant_reasoning_v2",
        request_mode="clip_horizons",
        teacher_error="HTTP 503 from endpoint",
    )
    assert rec.abstained is True
    assert rec.teacher_error == "HTTP 503 from endpoint"
    assert rec.horizons == []
