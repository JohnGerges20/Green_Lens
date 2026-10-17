from pathlib import Path
import random

import torch
from PIL import Image

from similarity import (
    get_dataset_images,
    get_image_index,
    extract_embedding,
)


# ---------------------------------
# Configuration
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent

NUM_QUERIES = 100
TOP_K = 5

random.seed(42)


# ---------------------------------
# Evaluate Similarity Search
# ---------------------------------

def evaluate_similarity():

    print("=" * 70)
    print("SIMILARITY SEARCH EVALUATION")
    print("=" * 70)

    image_files = get_dataset_images()

    print(f"Total dataset images: {len(image_files)}")

    if len(image_files) < TOP_K + 1:
        raise ValueError("Dataset is too small for evaluation.")

    print()
    print("Loading / building similarity index...")

    index = get_image_index()

    print(f"Indexed images: {len(index['paths'])}")

    # ---------------------------------
    # Select random queries
    # ---------------------------------

    valid_queries = []

    path_to_index = {
        str(Path(path).resolve()): i
        for i, path in enumerate(index["paths"])
    }

    shuffled = image_files.copy()
    random.shuffle(shuffled)

    for path, label in shuffled:

        resolved_path = str(path.resolve())

        if resolved_path in path_to_index:
            valid_queries.append((path, label))

        if len(valid_queries) >= NUM_QUERIES:
            break

    if not valid_queries:
        raise ValueError("No valid query images found.")

    print(f"Queries selected: {len(valid_queries)}")
    print()

    # ---------------------------------
    # Evaluation counters
    # ---------------------------------

    top1_correct = 0
    top5_correct = 0

    processed = 0

    # ---------------------------------
    # Evaluate each query
    # ---------------------------------

    for query_path, query_label in valid_queries:

        try:
            with Image.open(query_path) as image:
                query_image = image.convert("RGB")

            query_embedding = extract_embedding(query_image).float()

            gallery_embeddings = index["embeddings"].float()

            scores = gallery_embeddings @ query_embedding

            # ---------------------------------
            # Remove the query image itself
            # ---------------------------------

            query_index = path_to_index.get(
                str(query_path.resolve())
            )

            if query_index is not None:
                scores[query_index] = -float("inf")

            k = min(TOP_K, len(index["paths"]))

            top_scores, top_indices = torch.topk(
                scores,
                k=k,
            )

            predicted_labels = [
                index["labels"][idx]
                for idx in top_indices.tolist()
            ]

            # ---------------------------------
            # Top-1
            # ---------------------------------

            if predicted_labels[0] == query_label:
                top1_correct += 1

            # ---------------------------------
            # Top-5
            # ---------------------------------

            if query_label in predicted_labels:
                top5_correct += 1

            processed += 1

            if processed % 10 == 0:
                print(
                    f"Processed {processed}/{len(valid_queries)}"
                )

        except Exception as error:

            print(
                f"Warning: failed to evaluate "
                f"{query_path}: {error}"
            )

    # ---------------------------------
    # Results
    # ---------------------------------

    if processed == 0:
        raise ValueError("No queries were successfully evaluated.")

    top1_accuracy = top1_correct / processed
    top5_accuracy = top5_correct / processed

    print()
    print("=" * 70)
    print("SIMILARITY RESULTS")
    print("=" * 70)

    print(f"Queries evaluated : {processed}")
    print(
        f"Top-1 Retrieval   : "
        f"{top1_correct}/{processed} "
        f"({top1_accuracy:.2%})"
    )
    print(
        f"Top-5 Retrieval   : "
        f"{top5_correct}/{processed} "
        f"({top5_accuracy:.2%})"
    )

    print("=" * 70)

    print()
    print("Interpretation:")
    print(
        "Top-1 Retrieval measures whether the most similar "
        "image belongs to the same class as the query."
    )

    print(
        "Top-5 Retrieval measures whether at least one of "
        "the five most similar images belongs to the same class."
    )


# ---------------------------------
# Main
# ---------------------------------

if __name__ == "__main__":
    evaluate_similarity()