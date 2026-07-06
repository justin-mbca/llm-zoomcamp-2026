from pathlib import Path
import sys

import numpy as np
import pandas as pd
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch


ROOT = Path(__file__).resolve().parents[1]
HW2_DIR = ROOT / "llm-zoomcamp-hw2"
sys.path.append(str(HW2_DIR))

from embedder import Embedder  # noqa: E402


GROUND_TRUTH_URL = (
    "https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/"
    "cohorts/2026/04-evaluation/ground-truth.csv"
)


def load_documents():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    return [file.parse() for file in reader.read()]


def load_ground_truth():
    df = pd.read_csv(GROUND_TRUTH_URL)
    return df.to_dict(orient="records")


def build_text_search(chunks):
    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(chunks)

    def text_search(query, num_results=5):
        return index.search(
            query=query,
            boost_dict={"content": 1.0},
            num_results=num_results,
        )

    return text_search


def build_vector_search(chunks):
    model_path = HW2_DIR / "models" / "Xenova" / "all-MiniLM-L6-v2"
    embedder = Embedder(path=model_path)
    vectors = embedder.encode_batch([chunk["content"] for chunk in chunks])
    vectors = np.asarray(vectors)

    index = VectorSearch(keyword_fields=["filename"])
    index.fit(vectors, chunks)

    def vector_search(query, num_results=5):
        query_vector = embedder.encode(query)
        return index.search(query_vector, num_results=num_results)

    return vector_search


def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def compute_relevance(search_function, question_record):
    results = search_function(question_record["question"])
    return [
        int(result["filename"] == question_record["filename"])
        for result in results
    ]


def hit_rate(relevance_total):
    cnt = 0
    for line in relevance_total:
        if 1 in line:
            cnt += 1
    return cnt / len(relevance_total)


def mrr(relevance_total):
    total_score = 0.0
    for line in relevance_total:
        for rank, rel in enumerate(line):
            if rel == 1:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance_total)


def evaluate(ground_truth, search_function):
    relevance_total = [
        compute_relevance(search_function, record)
        for record in ground_truth
    ]
    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }


def main():
    documents = load_documents()
    chunks = chunk_documents(documents, size=2000, step=1000)
    ground_truth = load_ground_truth()

    print(f"documents: {len(documents)}")
    print(f"chunks: {len(chunks)}")
    print(f"ground_truth: {len(ground_truth)}")

    text_search = build_text_search(chunks)
    vector_search = build_vector_search(chunks)

    first_question = ground_truth[0]["question"]
    text_first = text_search(first_question)[0]["filename"]
    vector_first = vector_search(first_question)[0]["filename"]

    text_metrics = evaluate(ground_truth, text_search)
    vector_metrics = evaluate(ground_truth, vector_search)

    def hybrid_search(query, k=60):
        text_results = text_search(query, num_results=10)
        vector_results = vector_search(query, num_results=10)
        return rrf([text_results, vector_results], k=k)

    hybrid_metrics = {}
    for k in [1, 50, 100, 200]:
        hybrid_metrics[k] = evaluate(
            ground_truth,
            lambda query, k=k: hybrid_search(query, k=k),
        )

    best_hybrid_k = min(
        hybrid_metrics,
        key=lambda k: (-hybrid_metrics[k]["mrr"], k),
    )

    print()
    print("Homework answers")
    print("Q1 closest average input tokens: 1400")
    print(f"Q2 text first filename: {text_first}")
    print(f"Q3 vector first filename: {vector_first}")
    print(f"Q4 text hit rate: {text_metrics['hit_rate']:.4f}")
    print(f"Q5 vector MRR: {vector_metrics['mrr']:.4f}")
    print("Q6 hybrid MRR by k:")
    for k, metrics in hybrid_metrics.items():
        print(f"  k={k}: {metrics['mrr']:.4f}")
    print(f"Q6 best k: {best_hybrid_k}")


if __name__ == "__main__":
    main()
