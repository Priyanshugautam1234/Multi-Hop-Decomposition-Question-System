"""
Unit tests for end-to-end MultiHopPipeline execution and abstention.
"""

import pytest
from src.pipeline import MultiHopPipeline


def test_pipeline_bridge_question_resolution():
    pipeline = MultiHopPipeline()
    result = pipeline.query("Which city was the director of Inception born in?")
    
    assert result["abstained"] is False
    assert result["total_hops"] == 2
    assert "London" in result["final_answer"]
    
    # Verify citations are present
    assert "[Doc" in result["final_answer"]
    
    # Verify variable binding in trace
    hop1 = result["reasoning_trace"][0]
    hop2 = result["reasoning_trace"][1]
    
    assert "Christopher Nolan" in hop1["resolved_entity"]
    assert "Christopher Nolan" in hop2["sub_query"]
    assert hop2["depends_on"] == "hop_1"


def test_pipeline_unanswerable_abstain_fallback():
    pipeline = MultiHopPipeline()
    # Query with no supporting facts in corpus
    result = pipeline.query("What was the birth date of the astronaut who landed on Mars in 1820?")
    
    assert result["abstained"] is True
    assert "Insufficient Evidence / Abstain" in result["final_answer"]
