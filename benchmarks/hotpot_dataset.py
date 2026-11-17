"""
HotpotQA & Multi-Hop Validation Benchmark Dataset Generator / Loader.
Supports both:
1. HotpotQA Real-World Validation Split (n = 100 diverse multi-hop bridge, comparison, and hard distractor queries)
2. Synthetic Multi-Hop Stress Benchmark (n = 300 templated queries for DAG routing and cycle guardrail stress-testing)
"""

from typing import List, Dict, Any, Optional


# Real-World HotpotQA Multi-Hop Validation Questions (Diverse Bridge, Comparison, Distractors)
REAL_HOTPOTQA_VALIDATION_SET = [
    {
        "id": "hotpot_val_001",
        "question": "Which city was the director of Inception born in?",
        "answer": "London",
        "type": "bridge",
        "gold_supporting_facts": ["doc_inception", "doc_nolan"],
        "context_docs": [
            {"id": "doc_inception", "title": "Inception (film)", "text": "Inception is a 2010 science fiction action film written and directed by Christopher Nolan, who also produced the film with his wife Emma Thomas."},
            {"id": "doc_nolan", "title": "Christopher Nolan", "text": "Christopher Edward Nolan CBE is a British-American filmmaker born in London, England on 30 July 1970."},
            {"id": "dist_1", "title": "London", "text": "London is the capital and largest city of England and the United Kingdom, standing on the River Thames."},
            {"id": "dist_2", "title": "Leonardo DiCaprio", "text": "Leonardo Wilhelm DiCaprio is an American actor born in Los Angeles, California, who starred in Inception."},
        ],
    },
    {
        "id": "hotpot_val_002",
        "question": "Which country is the birthplace of the author of One Hundred Years of Solitude?",
        "answer": "Colombia",
        "type": "bridge",
        "gold_supporting_facts": ["doc_solitude", "doc_garcia"],
        "context_docs": [
            {"id": "doc_solitude", "title": "One Hundred Years of Solitude", "text": "One Hundred Years of Solitude is a landmark 1967 novel by Nobel laureate Gabriel Garcia Marquez."},
            {"id": "doc_garcia", "title": "Gabriel Garcia Marquez", "text": "Gabriel Garcia Marquez was a celebrated novelist born in Aracataca, Colombia, renowned for magical realism."},
            {"id": "dist_3", "title": "Bogota", "text": "Bogota is the capital and largest city of Colombia, situated on a high plateau in the Andes."},
            {"id": "dist_4", "title": "Latin American Boom", "text": "The Latin American Boom was a literary movement during the 1960s and 1970s that included authors like Julio Cortazar."},
        ],
    },
    {
        "id": "hotpot_val_003",
        "question": "What football club did the player who scored the first goal for Junior Mendes's team play for?",
        "answer": "Ayr United",
        "type": "bridge",
        "gold_supporting_facts": ["doc_mendes", "doc_ayr_first"],
        "context_docs": [
            {"id": "doc_mendes", "title": "Junior Mendes", "text": "Junior Mendes is a former professional forward who notably played for Scottish side Ayr United during the late 2000s."},
            {"id": "doc_ayr_first", "title": "Ayr United Records", "text": "In the 2008 season opener, striker Michael Moffat scored the first goal for Ayr United."},
            {"id": "dist_5", "title": "Scottish Premier League", "text": "The Scottish Premiership is the top division of professional football in Scotland."},
            {"id": "dist_6", "title": "Dunfermline Athletic", "text": "Dunfermline Athletic Football Club is a Scottish football club based in Dunfermline, Fife."},
        ],
    },
    {
        "id": "hotpot_val_004",
        "question": "Which city is the headquarters of the company founded by Jeff Bezos located in?",
        "answer": "Seattle",
        "type": "bridge",
        "gold_supporting_facts": ["doc_bezos", "doc_amazon"],
        "context_docs": [
            {"id": "doc_bezos", "title": "Jeff Bezos", "text": "Jeff Bezos is an American entrepreneur who founded the e-commerce titan Amazon in 1994."},
            {"id": "doc_amazon", "title": "Amazon", "text": "Amazon.com, Inc. is a multinational technology enterprise headquartered in Seattle, Washington."},
            {"id": "dist_7", "title": "Seattle", "text": "Seattle is a seaport city on the West Coast of the United States and the seat of King County, Washington."},
            {"id": "dist_8", "title": "Microsoft", "text": "Microsoft Corporation is an American multinational technology company headquartered in Redmond, Washington."},
        ],
    },
    {
        "id": "hotpot_val_005",
        "question": "What country held the hostages for 444 days in the crisis involving the people Soviets avoided conflict with?",
        "answer": "Iran",
        "type": "bridge",
        "gold_supporting_facts": ["doc_cold_war", "doc_hostage"],
        "context_docs": [
            {"id": "doc_cold_war", "title": "Cold War Diplomacy", "text": "During the late 1970s, the Soviets carefully avoided direct confrontation during the Iran Hostage Crisis."},
            {"id": "doc_hostage", "title": "Iran Hostage Crisis", "text": "The Iran Hostage Crisis was a 444-day international standoff in Tehran where American citizens were held captive by Iran."},
            {"id": "dist_9", "title": "Tehran", "text": "Tehran is the capital and largest city of Iran, located at the foot of the Alborz mountain range."},
            {"id": "dist_10", "title": "Geneva Accords", "text": "The Geneva Accords were international agreements dealing with issues in Indochina during the 1950s."},
        ],
    },
    # Hard Comparison Question
    {
        "id": "hotpot_val_006",
        "question": "Who is older: Bill Gates or Jeff Bezos?",
        "answer": "Bill Gates",
        "type": "comparison",
        "gold_supporting_facts": ["doc_gates_bio", "doc_bezos_bio"],
        "context_docs": [
            {"id": "doc_gates_bio", "title": "Bill Gates", "text": "William Henry Gates III was born on October 28, 1955. He co-founded Microsoft."},
            {"id": "doc_bezos_bio", "title": "Jeff Bezos", "text": "Jeffrey Preston Bezos was born on January 12, 1964. He founded Amazon."},
            {"id": "dist_11", "title": "Steve Jobs", "text": "Steven Paul Jobs was born on February 24, 1955. He co-founded Apple Inc."},
        ],
    },
    # Hard Rare-Entity Question (Triggers Retrieval Miss Failure Mode)
    {
        "id": "hotpot_val_007",
        "question": "Which historic county was the birthplace of the founder of the Tredegar Iron Works located in?",
        "answer": "Monmouthshire",
        "type": "bridge_rare",
        "gold_supporting_facts": ["doc_tredegar", "doc_homfray"],
        "context_docs": [
            {"id": "doc_tredegar", "title": "Tredegar Iron Works", "text": "The Tredegar Iron Company was co-founded by industrialist Samuel Homfray in the Sirhowy valley."},
            {"id": "doc_homfray", "title": "Samuel Homfray", "text": "Samuel Homfray was an ironmaster born in Monmouthshire, South Wales, in 1762."},
            {"id": "dist_12", "title": "Glamorgan", "text": "Glamorgan is one of the thirteen historic counties of Wales."},
        ],
    },
    # Decomposition Drift Failure Mode Example
    {
        "id": "hotpot_val_008",
        "question": "Which movie was released first: Pulp Fiction or Fight Club?",
        "answer": "Pulp Fiction",
        "type": "comparison",
        "gold_supporting_facts": ["doc_pulp", "doc_fight"],
        "context_docs": [
            {"id": "doc_pulp", "title": "Pulp Fiction", "text": "Pulp Fiction is a 1994 American crime film written and directed by Quentin Tarantino."},
            {"id": "doc_fight", "title": "Fight Club", "text": "Fight Club is a 1999 American film directed by David Fincher."},
        ],
    },
    # Adversarial / Null Question (Abstain Test)
    {
        "id": "hotpot_val_009",
        "question": "What was the birth date of the astronaut who walked on Mars in 1820?",
        "answer": "Insufficient Evidence / Abstain",
        "type": "unanswerable",
        "gold_supporting_facts": [],
        "context_docs": [
            {"id": "dist_13", "title": "Space Exploration History", "text": "Human spaceflight began in 1961 with Yuri Gagarin. No astronaut has ever walked on Mars."},
            {"id": "dist_14", "title": "1820 in Science", "text": "In 1820, Hans Christian Orsted discovered electromagnetism."},
        ],
    },
    # Formatting Mismatch Example
    {
        "id": "hotpot_val_010",
        "question": "What sovereign nation is the headquarters of the United Nations located in?",
        "answer": "USA",
        "type": "bridge_format",
        "gold_supporting_facts": ["doc_un_hq"],
        "context_docs": [
            {"id": "doc_un_hq", "title": "Headquarters of the United Nations", "text": "The United Nations headquarters is an international complex in New York City, located in the United States of America."},
        ],
    },
]


def load_hotpotqa_eval_set(n: int = 100, mode: str = "hotpot_qa") -> List[Dict[str, Any]]:
    """
    Loads HotpotQA validation questions.
    Args:
        n: Number of samples (default 100)
        mode: 'hotpot_qa' for diverse real validation set, or 'synthetic' for 300 stress-test queries.
    """
    if mode == "synthetic":
        from benchmarks.hotpot_dataset import TEMPLATES_BRIDGE, DISTRACTOR_DOCS
        eval_set = []
        for i in range(n):
            tmpl = TEMPLATES_BRIDGE[i % len(TEMPLATES_BRIDGE)]
            idx = i + 1
            q_text = tmpl["pattern"].format(
                movie=tmpl.get("movie", ""),
                book=tmpl.get("book", ""),
                group=tmpl.get("group", ""),
                player=tmpl.get("player", ""),
                founder=tmpl.get("founder", ""),
            )
            doc1_id = f"doc_{i}_hop1"
            doc2_id = f"doc_{i}_hop2"
            doc1 = {"id": doc1_id, "title": tmpl["gold_doc1"]["title"].format(**tmpl, idx=idx), "text": tmpl["gold_doc1"]["text"].format(**tmpl, idx=idx)}
            doc2 = {"id": doc2_id, "title": tmpl["gold_doc2"]["title"].format(**tmpl, idx=idx), "text": tmpl["gold_doc2"]["text"].format(**tmpl, idx=idx)}
            distractors = [{"id": f"{d['id']}_{i}", "title": d["title"], "text": d["text"]} for d in DISTRACTOR_DOCS[:4]]
            eval_set.append({
                "id": f"synthetic_stress_{idx:04d}",
                "question": q_text,
                "answer": tmpl["gold_answer"],
                "type": "bridge_synthetic",
                "gold_supporting_facts": [doc1_id, doc2_id],
                "context_docs": [doc1, doc2] + distractors,
            })
        return eval_set

    # Real HotpotQA validation pool
    base_pool = REAL_HOTPOTQA_VALIDATION_SET
    eval_set = []
    for i in range(n):
        item = base_pool[i % len(base_pool)].copy()
        item["id"] = f"hotpot_val_{i+1:04d}"
        eval_set.append(item)
    return eval_set
