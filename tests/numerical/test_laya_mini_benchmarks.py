"""Tiny official-style Laya benches (AG News 4-way, email spam noul) vs Python Agent."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Hand-picked headlines in the AG News 4-label space used in Laya BENCHMARKS.md.
# This is a smoke replica, not the 400-case official split.
AG_NEWS = [
    ("World", "UN Security Council meets after overnight strikes on the capital"),
    ("World", "Peace talks resume between the two governments after a ceasefire"),
    ("Sports", "Forward scores twice as the club wins the championship final"),
    ("Sports", "Olympic sprinter breaks the 100 metre world record in Paris"),
    ("Business", "Federal Reserve signals another interest rate hold this quarter"),
    ("Business", "Oil prices jump after the cartel announces a surprise output cut"),
    ("Sci/Tech", "Researchers release an open large language model trained on public data"),
    ("Sci/Tech", "NASA rover finds new evidence of ancient water on Mars"),
]
AG_CRITERIA = {
    "World": "international politics, diplomacy, war, governments",
    "Sports": "games, athletes, scores, tournaments",
    "Business": "markets, companies, economy, rates",
    "Sci/Tech": "science, technology, space, software",
}

SPAM = [
    (True, "Congratulations! You won a FREE iPhone, click here to claim your prize now!!!"),
    (True, "URGENT: your mailbox is full. Verify your password at http://bit.ly/not-real"),
    (False, "Hi Sam, confirming our 3pm call tomorrow about the Q3 roadmap."),
    (False, "Invoice 4411 is attached for March hosting. Net 30 as usual."),
]


@pytest.fixture(scope="module")
def english_agent():
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    import laya

    return laya.load("convaiinnovations/laya", device="cpu")


def test_ag_news_mini_python_agent(english_agent):
    questions = {
        "topic": {
            "type": "choice",
            "instructions": "Which AG News topic is this headline?",
            "criteria": AG_CRITERIA,
        }
    }
    hits = 0
    for label, text in AG_NEWS:
        out = english_agent.predict({"headline": text}, questions)
        pred = out["answers"]["topic"]["choice"]
        hits += int(pred == label)
    acc = hits / len(AG_NEWS)
    # Official English Laya is 0.95 on the full set; 8 headlines should be well above chance (0.25).
    assert acc >= 0.5, f"AG News mini accuracy {acc:.2f} ({hits}/{len(AG_NEWS)})"


def test_spam_noul_mini_python_agent(english_agent):
    questions = {
        "spam": {
            "type": "noul",
            "instructions": "Is this unsolicited spam or a phishing attempt?",
        }
    }
    hits = 0
    for is_spam, text in SPAM:
        out = english_agent.predict({"body": text}, questions)
        pred = float(out["answers"]["spam"]["noul"]) >= 0.5
        hits += int(pred == is_spam)
    acc = hits / len(SPAM)
    assert acc >= 0.75, f"spam mini accuracy {acc:.2f} ({hits}/{len(SPAM)})"
