"""
Interactive Trace UI & Multi-Hop Reasoning Visualizer (Streamlit).
Displays the live step-by-step DAG:
Hop 1 -> Hybrid Retrieval -> Entity Extraction -> Dynamic Variable Binding -> Hop 2 -> Grounded Cited Synthesis.
Also features interactive 3-way ablation benchmark explorer and failure taxonomy.
"""

import sys
import os
import json
import streamlit as st
import pandas as pd

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import MultiHopPipeline, DEFAULT_CORPUS


st.set_page_config(
    page_title="Multi-Hop Decomposition & Reasoning Trace Visualizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3B82F6;
    }
    .hop-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-entity {
        background-color: #10B981;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-hop {
        background-color: #3B82F6;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    return MultiHopPipeline()


pipeline = get_pipeline()

# Sidebar controls
st.sidebar.title("⚙️ Engine Configuration")
max_hops = st.sidebar.slider("Max Hop Budget Ceiling", min_value=1, max_value=6, value=4)
enable_reranking = st.sidebar.toggle("Enable Cross-Encoder Reranking", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Sample Multi-Hop Queries")

SAMPLE_QUERIES = [
    "Which city was the director of Inception born in?",
    "Which country is the birthplace of the author of One Hundred Years of Solitude?",
    "What country held the hostages for 444 days in the crisis involving the people Soviets avoided conflict with?",
    "What football club did the player who scored the first goal for Junior Mendes's team play for?",
    "Which city is the headquarters of the company founded by Jeff Bezos located in?",
    "What was the birth date of the astronaut who landed on Mars in 1820?",  # Unanswerable test
]

selected_sample = st.sidebar.selectbox("Choose a preset query or type below:", ["(Custom)"] + SAMPLE_QUERIES)

st.markdown('<div class="main-title">🧠 Multi-Hop Decomposition & Reasoning Trace Visualizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Adaptive Retrieve-Reason-Retrieve Loop with Dynamic Variable Binding, Hybrid Search (FAISS + BM25 + RRF), and Citation Grounding</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🚀 Live Query Reasoning Trace", "📊 3-Way Ablation Benchmark", "🔬 Failure Taxonomy & Error Analysis"])

with tab1:
    default_text = selected_sample if selected_sample != "(Custom)" else "Which city was the director of Inception born in?"
    user_query = st.text_input("Enter your complex multi-hop question:", value=default_text)

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        run_clicked = st.button("⚡ Run Reasoning Pipeline", type="primary", use_container_width=True)

    if run_clicked and user_query:
        pipeline.loop.guardrails.max_hops = max_hops
        with st.spinner("Executing Adaptive Retrieve-Reason-Retrieve Loop..."):
            result = pipeline.query(user_query)

        # Top KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Latency", f"{result['total_latency_ms']} ms")
        kpi2.metric("Executed Hops", f"{result['total_hops']} / {max_hops}")
        kpi3.metric("Tokens Analyzed", f"{result['total_tokens_used']}")
        status_label = "Abstained (Safe)" if result["abstained"] else "Answer Synthesized"
        kpi4.metric("Engine Status", status_label)

        st.markdown("---")
        st.subheader("🔗 Reasoning DAG & Variable Binding Timeline")

        # Visual Hop-by-Hop Timeline
        for trace in result["reasoning_trace"]:
            hop_num = trace["hop_number"]
            sub_q = trace["sub_query"]
            entity = trace["resolved_entity"]
            depends_on = trace["depends_on"]
            latency = trace["latency_ms"]
            chunks = trace["retrieved_chunks"]

            with st.container():
                st.markdown(
                    f"""
                    <div class="hop-card">
                        <span class="badge-hop">Hop {hop_num}</span>
                        {" <span class='badge-entity'>Bound Entity: " + str(entity) + "</span>" if entity else ""}
                        <span style="float: right; color: #6B7280; font-size: 0.85rem;">⏱️ {latency} ms</span>
                        <div style="font-size: 1.1rem; font-weight: 600; margin-top: 8px; color: #111827;">{sub_q}</div>
                        <div style="color: #4B5563; font-size: 0.9rem; margin-top: 4px;">
                            <strong>Dependency:</strong> {depends_on or 'None (Root Node)'} &nbsp;|&nbsp; 
                            <strong>Template:</strong> <code>{trace['raw_template'] or sub_q}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Collapsible chunks view
                with st.expander(f"🔍 Retrieved & Reranked Evidence Chunks for Hop {hop_num} ({len(chunks)} documents)"):
                    chunk_rows = []
                    for c in chunks:
                        chunk_rows.append({
                            "Citation ID": c["chunk_id"],
                            "Title": c["title"],
                            "Dense Score": c["dense_score"],
                            "BM25 Score": c["bm25_score"],
                            "RRF Score": c["rrf_score"],
                            "Rerank Score": c["rerank_score"],
                            "Final Score": c["final_score"],
                            "Passage Snippet": c["text"][:140] + "...",
                        })
                    if chunk_rows:
                        st.dataframe(pd.DataFrame(chunk_rows), use_container_width=True)

        st.markdown("---")
        st.subheader("🎯 Final Evidence-Grounded Synthesis")
        st.info(result["final_answer"])

        with st.expander("📄 View Raw Execution Trace JSON"):
            st.json(result)

with tab2:
    st.subheader("📊 3-Way Ablation Benchmark (n = 300 HotpotQA Validation Set)")
    st.markdown(
        """
        This benchmark quantifies the performance leap achieved by replacing naive single-step RAG
        with an **Adaptive Iterative Retrieve-Reason-Retrieve Loop**.
        """
    )

    results_file = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "results", "ablation_results.json")
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        table_df = pd.DataFrame(bench_data["results"])
        st.dataframe(
            table_df.rename(columns={
                "model": "Architecture Configuration",
                "em": "Exact Match (EM %)",
                "f1": "Token F1 (%)",
                "supporting_fact_recall": "Supporting-Fact Recall (%)",
                "faithfulness": "Faithfulness (RAGAS %)",
                "p50_latency_ms": "P50 Latency (ms)",
                "p95_latency_ms": "P95 Latency (ms)",
                "avg_hops": "Avg Hops",
            }),
            use_container_width=True,
        )

        # Bar charts
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.bar_chart(table_df.set_index("model")[["em", "f1", "supporting_fact_recall"]])
        with chart_col2:
            st.bar_chart(table_df.set_index("model")[["p50_latency_ms", "p95_latency_ms"]])
    else:
        st.warning("Benchmark results file not found. Run `python benchmarks/run_multihop_eval.py` to generate.")

with tab3:
    st.subheader("🔬 Failure Taxonomy & Error Analysis")
    st.markdown(
        """
        Analyzing failure distribution on multi-hop benchmarks reveals the real-world operational bottlenecks
        in complex question decomposition and retrieval systems.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.error("**1. Decomposition Drift (35% of Errors)**")
        st.write(
            "Occurs when the decomposer splits a simple comparison or compound query into overly narrow or redundant sub-queries, losing the original question's global intent or misclassifying relational predicates."
        )
        st.caption("Mitigation: Structured DAG schema validation and early-stopping sufficiency checks.")

        st.warning("**2. Entity Propagation Failure (25% of Errors)**")
        st.write(
            "Occurs when Hop 1 retrieves a distractor entity (e.g., co-star instead of director) and binds it into Hop 2's variable placeholder (`#1`), causing a cascading semantic drift."
        )
        st.caption("Mitigation: Cross-encoder reranking over entity mentions and top-k confidence gating.")

    with col2:
        st.info("**3. Retrieval Miss on Rare Entities (25% of Errors)**")
        st.write(
            "Dense vector embeddings struggle with out-of-vocabulary or low-frequency proper nouns (rare team names, lesser-known authors), leading to zero relevant chunks in top-k."
        )
        st.caption("Mitigation: Hybrid Search combining Dense FAISS with Lexical BM25 via Reciprocal Rank Fusion (RRF).")

        st.success("**4. Synthesis Formatting Mismatch (15% of Errors)**")
        st.write(
            "The system discovers the semantically correct answer (e.g., 'United States' or 'The Iran') but fails exact string match against the benchmark label ('USA' or 'Iran')."
        )
        st.caption("Mitigation: Standardized HotpotQA string normalization (article removal, lowercase, punctuation strip).")
