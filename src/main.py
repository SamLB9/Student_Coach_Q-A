import argparse, sys
from pathlib import Path
from .ingest import load_documents, chunk_documents
from .retriever import build_or_load_vectorstore, retrieve_context
from .quiz_engine import generate_quiz
from .evaluation import grade_answer
from .memory import JsonMemory
import time


def parse_args():
    ap = argparse.ArgumentParser(description="Study Coach — Milestone 1")
    ap.add_argument("--docs", default="data/notes", help="Folder with course notes")
    ap.add_argument("--topic", required=True, help="Topic to quiz")
    ap.add_argument("--n", type=int, default=4, help="Number of questions")
    ap.add_argument("--rebuild", action="store_true", help="Rebuild vector store from docs")
    ap.add_argument(
        "--avoid",
        choices=["all", "correct"],
        default="all",
        help="Avoid repeating: 'all' past prompts or only those answered 'correct'"
    )
    ap.add_argument(
        "--feedback",
        choices=["immediate", "end"],
        default="immediate",
        help="Show feedback right after each answer or at the end of the quiz"
    )
    ap.add_argument(
        "--missed",
        action="store_true",
        help="Show frequently missed questions for this topic at the end"
    )
    return ap.parse_args()


# --- Progress Logging (via JsonMemory) ---


def main():
    args = parse_args()

    memory = JsonMemory("progress.json")

    # Build or load vectorstore
    if args.rebuild or not Path("vectorstore").exists():
        print("Loading & chunking documents...")
        docs = load_documents(args.docs)
        if len(docs) == 0:
            print(f"No documents found in '{args.docs}'. Put PDFs/TXT/MD there and rerun with --rebuild.")
            sys.exit(1)
        chunks = chunk_documents(docs)
        print(f"Loaded {len(docs)} docs → {len(chunks)} chunks")
        vs = build_or_load_vectorstore(chunks)
        print("Vector store built.")
    else:
        vs = build_or_load_vectorstore([])  # load existing
        print("Loaded existing vector store.")

    print(f"Retrieving context for topic: {args.topic}")
    ctx = retrieve_context(vs, args.topic, k=6)
    if not ctx.strip():
        print("No relevant context retrieved. Did you ingest the right notes? Try --rebuild after adding docs.")
        sys.exit(1)

    # Determine excluded prompts and adaptive difficulty
    excluded_prompts = memory.get_excluded_prompts(mode=args.avoid, topic=args.topic)
    difficulty = memory.get_adaptive_difficulty(args.topic)

    print(f"Generating quiz... (difficulty: {difficulty})")
    quiz = generate_quiz(
        ctx,
        args.topic,
        n_questions=args.n,
        excluded_prompts=excluded_prompts,
        difficulty=difficulty,
    )
    questions = quiz.get("questions", []) if isinstance(quiz, dict) else []

    if not questions:
        print("Quiz generation returned no questions. Try refining the topic or rebuilding the vector store.")
        sys.exit(1)

    print("\n=== QUIZ ===")
    correct_count = 0

    if args.feedback == "immediate":
        # Present and grade one-by-one
        for i, q in enumerate(questions, 1):
            print(f"\nQ{i}. {q['prompt']}")
            if q.get("type") == "mcq":
                for opt in q.get("options", []):
                    print(f"  {opt}")
            start = time.perf_counter()
            answer = input(f"Answer Q{i}: ")
            end = time.perf_counter()
            took_ms = int(round((end - start) * 1000))

            # Grade immediately and log
            result = grade_answer(q['prompt'], q['answer'], answer)
            is_correct = bool(result.get('correct'))
            memory.log_attempt(
                topic=args.topic,
                prompt=q['prompt'],
                student_answer=answer,
                correct=is_correct,
                response_ms=took_ms,
            )

            if is_correct:
                correct_count += 1
            status = "✅ Correct" if is_correct else "❌ Incorrect"
            feedback = result.get('feedback', '')
            print(f"Q{i}: {status} (response: {took_ms} ms)")
            if feedback:
                print(f"Feedback: {feedback}")
    else:
        # Collect answers now, defer grading/printing until end
        answers: list[str] = []
        response_ms_list: list[int] = []
        for i, q in enumerate(questions, 1):
            print(f"\nQ{i}. {q['prompt']}")
            if q.get("type") == "mcq":
                for opt in q.get("options", []):
                    print(f"  {opt}")
            start = time.perf_counter()
            answer = input(f"Answer Q{i}: ")
            end = time.perf_counter()
            answers.append(answer)
            response_ms_list.append(int(round((end - start) * 1000)))

        print("\n=== FEEDBACK ===")
        for i, q in enumerate(questions, 1):
            result = grade_answer(q['prompt'], q['answer'], answers[i-1])
            is_correct = bool(result.get('correct'))
            memory.log_attempt(
                topic=args.topic,
                prompt=q['prompt'],
                student_answer=answers[i-1],
                correct=is_correct,
                response_ms=response_ms_list[i-1],
            )
            if is_correct:
                correct_count += 1
            status = "✅ Correct" if is_correct else "❌ Incorrect"
            feedback = result.get('feedback', '')
            took_ms = response_ms_list[i-1]
            print(f"\nQ{i}: {status} (response: {took_ms} ms)")
            if feedback:
                print(f"Feedback: {feedback}")

    total = len(questions)
    percent = (100 * correct_count / total) if total else 0.0
    print(f"\nScore: {correct_count}/{total} ({percent:.0f}%)")

    # Qualitative summary
    if percent < 50:
        print("Needs revision: focus on foundational concepts and definitions.")
    elif percent < 80:
        print("Fair progress: keep practicing and revisit tricky areas.")
    else:
        print("Good progress: you’re ready for more challenging, reasoning-based questions.")

    # Optional frequently missed report
    if args.missed:
        missed = memory.get_frequently_missed(args.topic, min_attempts=1, limit=5)
        if missed:
            print("\nFrequently missed questions (for this topic):")
            for idx, m in enumerate(missed, 1):
                err_pct = int(round(m["error_rate"] * 100))
                ms = m["avg_response_ms"]
                tail = f" | avg time: {ms} ms" if ms is not None else ""
                print(f"  {idx}. ({err_pct}% wrong over {m['attempts']} attempts){tail}\n     {m['prompt']}")
        else:
            print("\nNo frequently missed questions yet for this topic.")

    # Save progress using JsonMemory
    try:
        memory.log_session(
            topic=args.topic,
            score=percent,
            details={
                "raw": f"{correct_count}/{total}",
                "avoid_mode": args.avoid,
                "difficulty": difficulty,
                "feedback_mode": args.feedback,
                "show_missed": bool(args.missed),
            }
        )
        print("Progress saved to progress.json")
    except Exception as e:
        print(f"Warning: could not save progress: {e}")


if __name__ == "__main__":
    main()