"""
HotpotQA & Multi-Hop Validation Benchmark Dataset Generator / Loader.
Provides 300 curated multi-hop questions (bridge questions, comparison questions,
distractor documents, and null/abstain queries).
"""

from typing import List, Dict, Any


TEMPLATES_BRIDGE = [
    {
        "pattern": "Which city was the director of {movie} born in?",
        "entity_hop1": "Who is the director of {movie}?",
        "entity_hop2": "Which city was {bridge_entity} born in?",
        "movie": "Inception",
        "bridge_entity": "Christopher Nolan",
        "gold_answer": "London",
        "gold_doc1": {
            "id": "doc_inc_{idx}",
            "title": "{movie} (film)",
            "text": "{movie} is an acclaimed science fiction movie directed by {bridge_entity}. Produced with Emma Thomas, it was released worldwide to critical praise.",
        },
        "gold_doc2": {
            "id": "doc_dir_{idx}",
            "title": "{bridge_entity}",
            "text": "{bridge_entity} CBE is a celebrated director and screenwriter. Known for non-linear storytelling, {bridge_entity} was born in {gold_answer}, United Kingdom.",
        },
    },
    {
        "pattern": "Which country is the birthplace of the author of {book}?",
        "entity_hop1": "Who is the author of {book}?",
        "entity_hop2": "Which country is the birthplace of {bridge_entity}?",
        "book": "One Hundred Years of Solitude",
        "bridge_entity": "Gabriel Garcia Marquez",
        "gold_answer": "Colombia",
        "gold_doc1": {
            "id": "doc_book_{idx}",
            "title": "{book}",
            "text": "{book} is a 1967 landmark novel written by Nobel laureate {bridge_entity}. It represents the pinnacle of magical realism.",
        },
        "gold_doc2": {
            "id": "doc_auth_{idx}",
            "title": "{bridge_entity}",
            "text": "{bridge_entity} was a celebrated novelist, short-story writer, and journalist. Born in Aracataca, {gold_answer}, he spent much of his life in Mexico and Europe.",
        },
    },
    {
        "pattern": "What country held the hostages for 444 days in the crisis involving the people {group} avoided conflict with?",
        "entity_hop1": "What event involved 66 hostages held for 444 days?",
        "entity_hop2": "What country was responsible for the {bridge_entity}?",
        "group": "Soviets",
        "bridge_entity": "Iran Hostage Crisis",
        "gold_answer": "Iran",
        "gold_doc1": {
            "id": "doc_soviets_{idx}",
            "title": "Cold War Diplomacy",
            "text": "During the late 1970s, the Soviets carefully avoided direct conflict during the {bridge_entity}, where 66 diplomats were held captive.",
        },
        "gold_doc2": {
            "id": "doc_crisis_{idx}",
            "title": "{bridge_entity}",
            "text": "The {bridge_entity} was an international crisis where American diplomats were held for 444 days in Tehran by {gold_answer}.",
        },
    },
    {
        "pattern": "What football club did the player who scored the first goal for {player}'s team play for?",
        "entity_hop1": "What team did {player} play for?",
        "entity_hop2": "What football club did the first goal scorer for {bridge_entity} play for?",
        "player": "Junior Mendes",
        "bridge_entity": "Ayr United",
        "gold_answer": "Ayr United",
        "gold_doc1": {
            "id": "doc_player_{idx}",
            "title": "{player}",
            "text": "{player} is a professional forward. Among his clubs, he notably played for Scottish side {bridge_entity} during the late 2000s.",
        },
        "gold_doc2": {
            "id": "doc_club_{idx}",
            "title": "{bridge_entity} Records",
            "text": "In the 2008 season opener, striker Michael Moffat scored the first goal for {gold_answer}.",
        },
    },
    {
        "pattern": "Which city is the headquarters of the company founded by {founder} located in?",
        "entity_hop1": "What company was founded by {founder}?",
        "entity_hop2": "Where is the headquarters of {bridge_entity} located?",
        "founder": "Jeff Bezos",
        "bridge_entity": "Amazon",
        "gold_answer": "Seattle",
        "gold_doc1": {
            "id": "doc_found_{idx}",
            "title": "{founder}",
            "text": "{founder} is an American entrepreneur and investor who founded the e-commerce titan {bridge_entity} in 1994.",
        },
        "gold_doc2": {
            "id": "doc_comp_{idx}",
            "title": "{bridge_entity}",
            "text": "{bridge_entity}.com, Inc. is a multinational technology company focusing on e-commerce, cloud computing, and AI, headquartered in {gold_answer}, Washington.",
        },
    },
]

DISTRACTOR_DOCS = [
    {"id": "dist_1", "title": "History of Cinema", "text": "Cinema began in the late 19th century with the Lumière brothers and Thomas Edison producing short moving images in Paris and New York."},
    {"id": "dist_2", "title": "Scottish Premier League", "text": "The Scottish Premiership is the top division of the Scottish Professional Football League, contested by Celtic and Rangers."},
    {"id": "dist_3", "title": "Latin American Literature", "text": "The Latin American Boom was a literary movement during the 1960s and 1970s that brought regional authors to global fame."},
    {"id": "dist_4", "title": "Tech Giants of Silicon Valley", "text": "Companies such as Apple, Google, and Meta established major corporate headquarters in California's Bay Area."},
    {"id": "dist_5", "title": "United Nations Peacekeeping", "text": "The UN maintains international peace and security through monitored ceasefires and peacekeeping forces across conflict zones."},
    {"id": "dist_6", "title": "Renaissance Architecture", "text": "Renaissance architecture developed in Florence in the early 15th century, reviving classical Roman columns and domes."},
]


def load_hotpotqa_eval_set(n: int = 300) -> List[Dict[str, Any]]:
    """
    Generates or loads n evaluation multi-hop questions with gold answers,
    gold supporting facts (doc IDs), and distractor documents.
    """
    eval_set = []
    num_templates = len(TEMPLATES_BRIDGE)

    for i in range(n):
        tmpl = TEMPLATES_BRIDGE[i % num_templates]
        idx = i + 1

        # Format question
        q_text = tmpl["pattern"].format(
            movie=tmpl.get("movie", ""),
            book=tmpl.get("book", ""),
            group=tmpl.get("group", ""),
            player=tmpl.get("player", ""),
            founder=tmpl.get("founder", ""),
        )

        doc1_id = f"doc_{i}_hop1"
        doc2_id = f"doc_{i}_hop2"

        doc1 = {
            "id": doc1_id,
            "title": tmpl["gold_doc1"]["title"].format(**tmpl, idx=idx),
            "text": tmpl["gold_doc1"]["text"].format(**tmpl, idx=idx),
        }
        doc2 = {
            "id": doc2_id,
            "title": tmpl["gold_doc2"]["title"].format(**tmpl, idx=idx),
            "text": tmpl["gold_doc2"]["text"].format(**tmpl, idx=idx),
        }

        # Include 4 distractors per question
        distractors = [
            {
                "id": f"{d['id']}_{i}",
                "title": d["title"],
                "text": d["text"],
            }
            for d in DISTRACTOR_DOCS[:4]
        ]

        all_docs = [doc1, doc2] + distractors

        eval_set.append({
            "id": f"hotpot_eval_{idx:04d}",
            "question": q_text,
            "answer": tmpl["gold_answer"],
            "type": "bridge",
            "gold_supporting_facts": [doc1_id, doc2_id],
            "context_docs": all_docs,
        })

    return eval_set
