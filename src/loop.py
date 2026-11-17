"""
Adaptive Iterative Loop (Retrieve-Reason-Retrieve Control Loop).
Replaces static one-shot splitting with a dynamic, stateful loop:
Executes Hop 1 -> Retrieves evidence -> Extracts intermediate entity ->
Binds variable dynamically into Hop 2 -> Reranks -> Synthesizes final cited answer.
Enforces hard guardrails: max_hops = 4, cycle detection, and early stopping.
"""

from typing import Dict, Any, List, Optional
import time
from src.dag import StatefulDAG, QueryGuardrails, ReasoningTraceHop, DocumentChunk
from src.decomposer import AdaptiveDecomposer
from src.retriever import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.synthesizer import EvidenceSynthesizer


class AdaptiveIterativeLoop:
    """
    Iterative Retrieve-Reason-Retrieve controller.
    Dynamically binds discovered entities across hops and enforces execution guardrails.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
        synthesizer: Optional[EvidenceSynthesizer] = None,
        decomposer: Optional[AdaptiveDecomposer] = None,
        guardrails: Optional[QueryGuardrails] = None,
    ):
        self.retriever = retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.synthesizer = synthesizer or EvidenceSynthesizer()
        self.decomposer = decomposer or AdaptiveDecomposer()
        self.guardrails = guardrails or QueryGuardrails()

    def run(self, root_question: str) -> Dict[str, Any]:
        """
        Executes the adaptive multi-hop reasoning loop on the root question.
        Returns the final answer with citations and the complete execution trace.
        """
        start_total_time = time.perf_counter()
        dag = StatefulDAG(root_question, guardrails=self.guardrails)

        # Step 1: Initial Structured Decomposition
        sub_questions = self.decomposer.decompose(root_question)
        for sq in sub_questions:
            dag.add_subquestion(
                hop_id=sq.hop_id,
                sub_question=sq.sub_question,
                depends_on=sq.depends_on,
                variable_placeholder=sq.variable_placeholder,
            )

        hop_count = 0
        current_node_index = 0

        while dag.can_continue() and current_node_index < len(dag.node_order):
            hop_count += 1
            node_id = dag.node_order[current_node_index]
            node = dag.nodes[node_id]

            hop_start_time = time.perf_counter()

            # Dynamic Variable Binding: Check if this hop depends on an earlier hop
            if node.depends_on and node.depends_on in dag.bound_variables:
                bound_entity = dag.bound_variables[node.depends_on]
                dag.bind_variable(node.depends_on, bound_entity)

            # Concrete query for this hop
            current_sub_query = node.sub_question

            # Guardrail 1: Cycle / Duplicate Query Detection
            if dag.detect_cycle_or_duplicate(current_sub_query):
                # Cycle or identical query detected - early break to prevent infinite loops
                break

            # Guardrail 2: Hard Hop Budget Ceiling (max_hops = 4)
            if hop_count > self.guardrails.max_hops:
                break

            # Step 2: Hybrid Retrieval (Dense FAISS + BM25 Lexical + RRF)
            retrieved_chunks = self.retriever.search(current_sub_query, top_k=5)

            # Step 3: Cross-Encoder Reranking
            reranked_chunks = self.reranker.rerank(current_sub_query, retrieved_chunks, top_k=3)

            # Step 4: Intermediate Extraction & Variable Resolution
            inter_ans, entity, conf, has_enough = self.synthesizer.extract_intermediate_entity(
                current_sub_query, reranked_chunks
            )

            # Bind extracted entity to DAG state for future hops
            if entity:
                dag.bind_variable(node.hop_id, entity)

            hop_latency_ms = round((time.perf_counter() - hop_start_time) * 1000, 2)
            # Estimate tokens used for this hop
            tokens_used = len(current_sub_query.split()) + sum(len(c.text.split()) for c in reranked_chunks) + len(inter_ans.split())

            # Record Hop Trace
            trace_hop = ReasoningTraceHop(
                hop_number=hop_count,
                sub_query=current_sub_query,
                raw_template=node.raw_template,
                depends_on=node.depends_on,
                resolved_entity=entity or None,
                retrieved_chunks=reranked_chunks,
                intermediate_answer=inter_ans,
                has_enough_evidence=has_enough,
                latency_ms=hop_latency_ms,
                tokens_used=tokens_used,
            )
            dag.record_hop_execution(trace_hop)

            # Guardrail 3: Early-Stopping Sufficiency Check
            # If evidence is fully conclusive or if no further nodes remain, break early
            if current_node_index == len(dag.node_order) - 1 and has_enough:
                break

            current_node_index += 1

        # Step 5: Final Synthesis with Inline [Doc X] Citations and Abstain Guard
        final_answer, abstained = self.synthesizer.synthesize_final_answer(
            root_question=root_question,
            reasoning_traces=dag.history_traces,
        )

        total_latency_ms = round((time.perf_counter() - start_total_time) * 1000, 2)
        total_tokens = sum(t.tokens_used for t in dag.history_traces)

        return {
            "query": root_question,
            "final_answer": final_answer,
            "abstained": abstained,
            "total_hops": len(dag.history_traces),
            "total_latency_ms": total_latency_ms,
            "total_tokens_used": total_tokens,
            "bound_variables": dag.bound_variables,
            "reasoning_trace": [t.model_dump() for t in dag.history_traces],
        }
