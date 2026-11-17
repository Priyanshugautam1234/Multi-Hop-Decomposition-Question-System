"""
High-Level Multi-Hop Pipeline Interface.
Provides unified initialization, corpus indexing, and query execution.
"""

from typing import List, Dict, Any, Optional
import os
import pandas as pd
from src.retriever import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.synthesizer import EvidenceSynthesizer
from src.decomposer import AdaptiveDecomposer
from src.loop import AdaptiveIterativeLoop
from src.dag import QueryGuardrails


DEFAULT_CORPUS = [
    {
        "id": "doc_inception",
        "title": "Inception (film)",
        "text": "Inception is a 2010 science fiction action film written and directed by Christopher Nolan, who also produced the film with his wife, Emma Thomas. The film stars Leonardo DiCaprio as a professional thief.",
    },
    {
        "id": "doc_nolan",
        "title": "Christopher Nolan",
        "text": "Christopher Edward Nolan CBE is a British and American filmmaker. Known for his Hollywood blockbusters, Christopher Nolan was born in London, England on 30 July 1970.",
    },
    {
        "id": "doc_london",
        "title": "London",
        "text": "London is the capital and largest city of England and the United Kingdom. It stands on the River Thames in south-east England at the head of a 50-mile estuary.",
    },
    {
        "id": "doc_mendes",
        "title": "Junior Mendes",
        "text": "Junior Mendes is a former professional footballer who played as a forward. In his career, he represented Ayr United, St Mirren, and Dunfermline Athletic.",
    },
    {
        "id": "doc_ayr_united",
        "title": "Ayr United F.C.",
        "text": "Ayr United Football Club is a Scottish association football club. In the 2008 season, Michael Moffat scored the first goal of last season for Ayr United.",
    },
    {
        "id": "doc_iran_hostage",
        "title": "Iran Hostage Crisis",
        "text": "The Iran hostage crisis was a diplomatic standoff between the United States and Iran. 66 American diplomats and citizens were held hostage for 444 days from November 4, 1979, to January 20, 1981, by Iranian students.",
    },
    {
        "id": "doc_armenia_iran",
        "title": "Demographics of Iran",
        "text": "Iran is a diverse country. The people speaking a language resembled most by Armenians include minorities in northwestern provinces, accounting for approximately 0.2 percent of Iran's population.",
    },
    {
        "id": "doc_bezos",
        "title": "Jeff Bezos",
        "text": "Jeffrey Preston Bezos is an American businessman, media proprietor, and investor. He is the founder and executive chairman of Amazon. Bezos was born on January 12, 1964.",
    },
    {
        "id": "doc_gates",
        "title": "Bill Gates",
        "text": "William Henry Gates III is an American businessman, investor, and philanthropist. He is best known for co-founding software giant Microsoft with childhood friend Paul Allen. Gates was born on October 28, 1955.",
    },
]


class MultiHopPipeline:
    """
    Unified production pipeline for Multi-Hop QA with Hybrid Search,
    Adaptive Iterative Loop, Reranking, and Cited Synthesis.
    """

    def __init__(
        self,
        corpus: Optional[List[Dict[str, Any]]] = None,
        model_path: Optional[str] = "./t5-decomposition",
        max_hops: int = 4,
    ):
        self.guardrails = QueryGuardrails(max_hops=max_hops)
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        self.synthesizer = EvidenceSynthesizer()
        self.decomposer = AdaptiveDecomposer(model_path=model_path)
        
        self.loop = AdaptiveIterativeLoop(
            retriever=self.retriever,
            reranker=self.reranker,
            synthesizer=self.synthesizer,
            decomposer=self.decomposer,
            guardrails=self.guardrails,
        )

        initial_corpus = corpus or DEFAULT_CORPUS
        self.retriever.index_documents(initial_corpus)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Adds and indexes additional documents to the corpus."""
        existing = self.retriever.documents
        combined = existing + documents
        self.retriever.index_documents(combined)

    def query(self, question: str) -> Dict[str, Any]:
        """Runs the question through the adaptive retrieve-reason-retrieve loop."""
        return self.loop.run(question)
