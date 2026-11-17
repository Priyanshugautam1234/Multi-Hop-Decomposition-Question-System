"""
Stateful Directed Acyclic Graph (DAG) & Dynamic Variable Binding Engine.
Manages sub-question nodes, dependency propagation, cycle detection, and guardrails.
"""

from typing import List, Dict, Optional, Any, Set
import re
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


class SubQuestion(BaseModel):
    hop_id: str = Field(description="Unique identifier for the hop (e.g., hop_1, hop_2)")
    sub_question: str = Field(description="The concrete natural language sub-question")
    raw_template: Optional[str] = Field(default=None, description="Original template before variable resolution")
    depends_on: Optional[str] = Field(default=None, description="ID of preceding hop this question depends on (e.g., hop_1)")
    variable_placeholder: Optional[str] = Field(default=None, description="Placeholder token to replace (e.g., #1, $entity)")
    resolved_entity: Optional[str] = Field(default=None, description="Entity extracted from dependent hop to bind into this query")
    status: str = Field(default="pending", description="pending | running | resolved | skipped | abstained")


class ReasoningTraceHop(BaseModel):
    hop_number: int
    sub_query: str
    raw_template: Optional[str] = None
    depends_on: Optional[str] = None
    resolved_entity: Optional[str] = None
    retrieved_chunks: List[DocumentChunk] = Field(default_factory=list)
    intermediate_answer: str = ""
    has_enough_evidence: bool = False
    latency_ms: float = 0.0
    tokens_used: int = 0


class QueryGuardrails(BaseModel):
    max_hops: int = 4
    similarity_threshold: float = 0.85
    min_confidence_score: float = 0.15


class StatefulDAG:
    """
    Stateful execution graph tracking sub-questions, intermediate variable bindings,
    visited queries for cycle detection, and early-stopping criteria.
    """

    def __init__(self, root_question: str, guardrails: Optional[QueryGuardrails] = None):
        self.root_question = root_question
        self.guardrails = guardrails or QueryGuardrails()
        self.nodes: Dict[str, SubQuestion] = {}
        self.node_order: List[str] = []
        self.history_traces: List[ReasoningTraceHop] = []
        self.visited_queries: Set[str] = set()
        self.bound_variables: Dict[str, str] = {}  # e.g., {"hop_1": "Christopher Nolan", "#1": "Christopher Nolan"}

    @staticmethod
    def normalize_query(text: str) -> str:
        """Normalize query string for cycle & duplicate detection."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        return " ".join(text.split())

    def detect_cycle_or_duplicate(self, candidate_query: str) -> bool:
        """
        Detects if candidate query has already been visited or is semantically duplicate.
        Prevents endless retrieval loops and repetitive questions.
        """
        norm = self.normalize_query(candidate_query)
        if norm in self.visited_queries:
            return True

        # Token Jaccard check against visited queries
        tokens_cand = set(norm.split())
        if not tokens_cand:
            return True

        for v in self.visited_queries:
            tokens_v = set(v.split())
            if not tokens_v:
                continue
            intersection = len(tokens_cand & tokens_v)
            union = len(tokens_cand | tokens_v)
            jaccard = intersection / union if union > 0 else 0.0
            if jaccard >= self.guardrails.similarity_threshold:
                return True

        return False

    def add_subquestion(
        self,
        hop_id: str,
        sub_question: str,
        depends_on: Optional[str] = None,
        variable_placeholder: Optional[str] = None,
    ) -> SubQuestion:
        """Registers a sub-question into the DAG."""
        node = SubQuestion(
            hop_id=hop_id,
            sub_question=sub_question,
            raw_template=sub_question,
            depends_on=depends_on,
            variable_placeholder=variable_placeholder or (f"#{depends_on.split('_')[-1]}" if depends_on else None),
        )
        self.nodes[hop_id] = node
        if hop_id not in self.node_order:
            self.node_order.append(hop_id)
        return node

    def bind_variable(self, hop_id: str, entity_value: str):
        """
        Stores the resolved entity for hop_id and propagates it to dependent nodes.
        Supports structured JSON binding:
        {"sub_question": "...", "depends_on": "hop_1", "resolved_entity": "Christopher Nolan"}
        """
        clean_entity = entity_value.strip().strip('"').strip("'")
        self.bound_variables[hop_id] = clean_entity

        # Also store short alias like #1
        match = re.search(r"\d+", hop_id)
        if match:
            self.bound_variables[f"#{match.group(0)}"] = clean_entity

        # Propagate to dependent nodes in DAG
        for nid, node in self.nodes.items():
            if node.depends_on == hop_id:
                node.resolved_entity = clean_entity
                node.sub_question = self._inject_entity(node.raw_template or node.sub_question, clean_entity, node.variable_placeholder)

    def _inject_entity(self, template: str, entity: str, placeholder: Optional[str] = None) -> str:
        """Injects discovered entity into the dependent sub-question template."""
        res = template
        if placeholder and placeholder in res:
            res = res.replace(placeholder, entity)
        
        # Generic replacements for common patterns
        res = re.sub(r"#\d+", entity, res)
        res = re.sub(r"\[resolved_entity\]", entity, res, flags=re.IGNORECASE)
        res = re.sub(r"\$entity", entity, res, flags=re.IGNORECASE)
        res = re.sub(r"\bthis person\b", entity, res, flags=re.IGNORECASE)
        res = re.sub(r"\bthat person\b", entity, res, flags=re.IGNORECASE)
        res = re.sub(r"\bthat team\b", entity, res, flags=re.IGNORECASE)
        res = re.sub(r"\bthis movie\b", entity, res, flags=re.IGNORECASE)
        res = re.sub(r"\bthat movie\b", entity, res, flags=re.IGNORECASE)
        return res

    def record_hop_execution(self, trace_hop: ReasoningTraceHop):
        """Records executed hop and marks query as visited."""
        self.history_traces.append(trace_hop)
        self.visited_queries.add(self.normalize_query(trace_hop.sub_query))

    def can_continue(self) -> bool:
        """Checks if budget permits another hop."""
        return len(self.history_traces) < self.guardrails.max_hops

    def to_dict(self) -> Dict[str, Any]:
        """Serializes current state to structured JSON format."""
        return {
            "root_question": self.root_question,
            "max_hops": self.guardrails.max_hops,
            "total_executed_hops": len(self.history_traces),
            "bound_variables": self.bound_variables,
            "nodes": [node.model_dump() for node in self.nodes.values()],
            "reasoning_trace": [trace.model_dump() for trace in self.history_traces],
        }
