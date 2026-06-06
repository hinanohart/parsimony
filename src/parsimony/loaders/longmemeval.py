"""Load LongMemEval (MIT, xiaowu0162/longmemeval) into ordered session traces.

Gold is machine-checkable: ``answer_session_ids`` lists exactly which sessions
contain the answer, so retention can be scored without any LLM. The file is not
vendored; download ``longmemeval_oracle`` or ``longmemeval_s`` from Hugging Face
into ``data/`` first.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

_DATE = re.compile(r"(\d{4})/(\d{2})/(\d{2})\D+?(\d{2}):(\d{2})")


@dataclass(frozen=True, slots=True)
class LMEQuestion:
    question_id: str
    question: str
    answer: str
    question_type: str
    sessions: list[tuple[str, str, str]]  # (session_id, date, text)
    gold_session_ids: list[str]


def date_key(s: str, fallback: int) -> tuple[int, ...]:
    m = _DATE.search(s or "")
    if not m:
        return (9999, 99, 99, 99, 99, fallback)
    return (*(int(x) for x in m.groups()), fallback)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_longmemeval(path: str | Path) -> tuple[list[LMEQuestion], str]:
    """Return (questions, sha256-of-file)."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    sha = file_sha256(path)
    out: list[LMEQuestion] = []
    for r in raw:
        sids = r["haystack_session_ids"]
        dates = r["haystack_dates"]
        sessions = r["haystack_sessions"]
        sess: list[tuple[str, str, str]] = []
        for sid, date, turns in zip(sids, dates, sessions, strict=False):
            text = " ".join(str(t.get("content", "")) for t in turns)
            sess.append((str(sid), str(date), text))
        out.append(
            LMEQuestion(
                question_id=str(r["question_id"]),
                question=str(r.get("question", "")),
                answer=str(r.get("answer", "")),
                question_type=str(r.get("question_type", "")),
                sessions=sess,
                gold_session_ids=[str(x) for x in r["answer_session_ids"]],
            )
        )
    return out, sha
