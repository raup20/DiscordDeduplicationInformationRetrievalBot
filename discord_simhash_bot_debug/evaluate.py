import json
import time
from qa_store import QAStore

DATASET_FILE = "eval_dataset.json"

def load_dataset():
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    dataset = load_dataset()

    store = QAStore()

    # Clear existing data (important for clean evaluation)
    store.questions.clear()
    store.answers.clear()
    store.answers_by_qid.clear()
    store.recent_q_by_channel.clear()

    # Insert all questions and answers
    for i, qa in enumerate(dataset):
        qid = i
        aid = i + 1000

        store.add_question(
            msg_id=qid,
            channel_id=0,
            author_id=0,
            text=qa["question"],
            ts=time.time()
        )

        store.add_answer(
            msg_id=aid,
            channel_id=0,
            author_id=0,
            text=qa["answer"],
            reply_to_msg_id=qid,
            ts=time.time()
        )

    TP = FP = FN = 0

    # Query with the original questions (you can later paraphrase)
    for qa in dataset:
        results = store.search_questions(qa["question"], top_k=1, min_sim=0.75)

        if not results:
            FN += 1
            continue

        predicted_q, sim = results[0]

        if predicted_q.text == qa["question"]:
            TP += 1
        else:
            FP += 1

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0

    print("Evaluation results:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"TP={TP}, FP={FP}, FN={FN}")

if __name__ == "__main__":
    main()
