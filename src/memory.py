import json
from pathlib import Path
from datetime import datetime
import hashlib
from typing import Dict, Any, List, Optional

class JsonMemory:
    def __init__(self, path="progress.json"):
        self.path = Path(path)
        if not self.path.exists():
            self._write({"sessions": [], "attempts": [], "questions": {}})
        else:
            # Ensure required keys exist for forward compatibility; repair corrupt files
            try:
                data = self._read()
            except Exception:
                # Reinitialize corrupt/empty file
                data = {"sessions": [], "attempts": [], "questions": {}}
                self._write(data)
            changed = False
            if not isinstance(data, dict):
                data = {"sessions": [], "attempts": [], "questions": {}}
                changed = True
            if "sessions" not in data or not isinstance(data["sessions"], list):
                data["sessions"] = []
                changed = True
            if "attempts" not in data or not isinstance(data["attempts"], list):
                data["attempts"] = []
                changed = True
            if "questions" not in data or not isinstance(data["questions"], dict):
                data["questions"] = {}
                changed = True
            if changed:
                self._write(data)

    def _read(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ----- Session Logging -----
    def log_session(self, topic: str, score: float, details: dict):
        data = self._read()
        data.setdefault("sessions", [])
        data["sessions"].append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "topic": topic,
            "score": score,
            "details": details
        })
        self._write(data)

    # ----- Question Attempts -----
    @staticmethod
    def question_id(prompt: str) -> str:
        """Stable ID for a question prompt using normalized SHA-256 (shortened)."""
        normalized = " ".join(prompt.strip().split()).lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def log_attempt(
        self,
        topic: str,
        prompt: str,
        student_answer: str,
        correct: bool,
        response_ms: Optional[int] = None,
    ) -> None:
        data = self._read()
        data.setdefault("attempts", [])
        data.setdefault("questions", {})

        qid = self.question_id(prompt)
        ts = datetime.utcnow().isoformat() + "Z"

        attempt_entry = {
            "timestamp": ts,
            "topic": topic,
            "question_id": qid,
            "prompt": prompt,
            "student_answer": student_answer,
            "correct": correct,
        }
        if response_ms is not None:
            attempt_entry["response_ms"] = int(response_ms)

        # Append attempt entry
        data["attempts"].append(attempt_entry)

        # Update questions aggregate
        qrec = data["questions"].get(qid, {
            "prompt": prompt,
            "times_asked": 0,
            "last_answer": None,
            "last_correct": None,
            "last_timestamp": None,
            "topics": [],
            "last_response_ms": None,
            "avg_response_ms": None,
        })
        prev_n = int(qrec.get("times_asked", 0))
        qrec["times_asked"] = prev_n + 1
        qrec["last_answer"] = student_answer
        qrec["last_correct"] = bool(correct)
        qrec["last_timestamp"] = ts
        topics = set(qrec.get("topics", []))
        topics.add(topic)
        qrec["topics"] = sorted(topics)

        if response_ms is not None:
            ms = int(response_ms)
            qrec["last_response_ms"] = ms
            prev_avg = qrec.get("avg_response_ms")
            if isinstance(prev_avg, (int, float)) and prev_n > 0:
                qrec["avg_response_ms"] = int(round((prev_avg * prev_n + ms) / (prev_n + 1)))
            else:
                qrec["avg_response_ms"] = ms

        data["questions"][qid] = qrec

        self._write(data)

    def get_excluded_prompts(self, mode: str = "all", topic: Optional[str] = None) -> List[str]:
        """Return prompts to exclude. mode: 'all' (default) or 'correct'.
        If topic is provided, only consider prompts associated with that topic.
        """
        data = self._read()
        questions = data.get("questions", {})
        def topic_match(rec: Dict[str, Any]) -> bool:
            if topic is None:
                return True
            rec_topics = rec.get("topics", [])
            return isinstance(rec_topics, list) and topic in rec_topics
        if mode == "correct":
            return [rec.get("prompt", "") for rec in questions.values() if rec.get("last_correct") is True and topic_match(rec)]
        # default: all previously asked prompts
        return [rec.get("prompt", "") for rec in questions.values() if topic_match(rec)]

    # ----- Adaptive Difficulty -----
    def get_topic_accuracy(self, topic: str, default: float = 0.7) -> float:
        """Return accuracy (0-1) for a topic based on attempts; falls back to session scores."""
        data = self._read()
        attempts = [a for a in data.get("attempts", []) if a.get("topic") == topic]
        if attempts:
            total = len(attempts)
            correct = sum(1 for a in attempts if a.get("correct") is True)
            return correct / total if total else default
        # Fallback to sessions
        sessions = [s for s in data.get("sessions", []) if s.get("topic") == topic]
        if sessions:
            ratios = []
            for s in sessions:
                try:
                    ratios.append(float(s.get("score", 0.0)) / 100.0)
                except Exception:
                    continue
            if ratios:
                return sum(ratios) / len(ratios)
        return default

    def get_adaptive_difficulty(self, topic: str) -> str:
        """Map topic accuracy to a difficulty label: easy/medium/hard."""
        acc = self.get_topic_accuracy(topic)
        if acc < 0.5:
            return "easy"
        if acc < 0.8:
            return "medium"
        return "hard"

    # ----- Missed Questions Summary -----
    def get_frequently_missed(self, topic: str, min_attempts: int = 1, limit: int = 5) -> List[Dict[str, Any]]:
        """Return up to 'limit' questions for the topic with the highest error rates.
        Each item: {prompt, attempts, correct, incorrect, error_rate, avg_response_ms}
        """
        data = self._read()
        attempts = [a for a in data.get("attempts", []) if a.get("topic") == topic]
        if not attempts:
            return []
        stats: Dict[str, Dict[str, Any]] = {}
        for a in attempts:
            qid = a.get("question_id")
            if not qid:
                continue
            st = stats.setdefault(qid, {"prompt": a.get("prompt", ""), "attempts": 0, "correct": 0, "incorrect": 0, "sum_ms": 0, "ms_count": 0})
            st["attempts"] += 1
            if a.get("correct") is True:
                st["correct"] += 1
            else:
                st["incorrect"] += 1
            if isinstance(a.get("response_ms"), int):
                st["sum_ms"] += a["response_ms"]
                st["ms_count"] += 1
        # compute error rate and avg ms
        items: List[Dict[str, Any]] = []
        for qid, st in stats.items():
            if st["attempts"] < min_attempts or st["incorrect"] == 0:
                continue
            avg_ms = int(round(st["sum_ms"] / st["ms_count"])) if st["ms_count"] > 0 else None
            error_rate = st["incorrect"] / st["attempts"]
            items.append({
                "prompt": st["prompt"],
                "attempts": st["attempts"],
                "correct": st["correct"],
                "incorrect": st["incorrect"],
                "error_rate": error_rate,
                "avg_response_ms": avg_ms,
            })
        items.sort(key=lambda x: (x["error_rate"], x["attempts"], (x["avg_response_ms"] or 0)), reverse=True)
        return items[:limit]