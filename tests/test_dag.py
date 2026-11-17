"""
Unit tests for StatefulDAG, dynamic variable binding, and guardrails.
"""

import pytest
from src.dag import StatefulDAG, QueryGuardrails, ReasoningTraceHop, DocumentChunk


def test_stateful_dag_variable_binding():
    dag = StatefulDAG(root_question="Which city was the director of Inception born in?")
    
    # Hop 1: Who is the director of Inception?
    dag.add_subquestion(
        hop_id="hop_1",
        sub_question="Who is the director of Inception?",
        depends_on=None,
    )
    
    # Hop 2: Which city was #1 born in?
    dag.add_subquestion(
        hop_id="hop_2",
        sub_question="Which city was #1 born in?",
        depends_on="hop_1",
        variable_placeholder="#1",
    )
    
    # Before binding: Hop 2 has raw template with #1
    assert dag.nodes["hop_2"].sub_question == "Which city was #1 born in?"
    
    # Bind entity discovered in Hop 1
    dag.bind_variable("hop_1", "Christopher Nolan")
    
    # After binding: Hop 2 dynamically injected Christopher Nolan
    assert dag.nodes["hop_2"].sub_question == "Which city was Christopher Nolan born in?"
    assert dag.nodes["hop_2"].resolved_entity == "Christopher Nolan"
    assert dag.bound_variables["hop_1"] == "Christopher Nolan"
    assert dag.bound_variables["#1"] == "Christopher Nolan"


def test_dag_cycle_and_duplicate_detection():
    dag = StatefulDAG(root_question="Who is the director of Inception?")
    
    trace1 = ReasoningTraceHop(
        hop_number=1,
        sub_query="Who directed Inception?",
        intermediate_answer="Christopher Nolan",
    )
    dag.record_hop_execution(trace1)
    
    # Exact duplicate
    assert dag.detect_cycle_or_duplicate("Who directed Inception?") is True
    # Semantically near-identical query
    assert dag.detect_cycle_or_duplicate("who directed inception") is True
    # Distinct query
    assert dag.detect_cycle_or_duplicate("Where was Christopher Nolan born?") is False


def test_dag_hop_budget_guardrails():
    guardrails = QueryGuardrails(max_hops=2)
    dag = StatefulDAG(root_question="Root Q", guardrails=guardrails)
    
    assert dag.can_continue() is True
    
    dag.record_hop_execution(ReasoningTraceHop(hop_number=1, sub_query="Q1"))
    assert dag.can_continue() is True
    
    dag.record_hop_execution(ReasoningTraceHop(hop_number=2, sub_query="Q2"))
    # Hit max_hops = 2 ceiling
    assert dag.can_continue() is False
