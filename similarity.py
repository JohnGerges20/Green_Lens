
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import models


# ---------------------------------
# 1. Paths and settings
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "similarity_images"
CACHE_DIR = BASE_DIR / "models"
CACHE_PATH = CACHE_DIR / "similarity_index_test.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# TEST MODE: use a small number of images per class.
# Increase this later to search a larger part of the dataset.
MAX_IMAGES_PER_CLASS = 25
BATCH_SIZE = 32


# ---------------------------------
# 2. Load pretrained ResNet50
# ---------------------------------

def load_similarity_model():
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()
    model = model.to(DEVICE)
    model.eval()
    return model, weights.transforms()


similarity_model, similarity_transform = load_similarity_model()


# ---------------------------------
# 3. Extract image embedding
# ---------------------------------

def extract_embedding(image):
    image = image.convert("RGB")
    tensor = similarity_transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        embedding = similarity_model(tensor)
        embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.squeeze(0).cpu()


# ---------------------------------
# 4. Collect a small dataset sample
# ---------------------------------

def get_dataset_images():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}"
        )

    images = []

    for class_dir in sorted(DATASET_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        class_images = []

        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                class_images.append(path)

                if len(class_images) >= MAX_IMAGES_PER_CLASS:
                    break

        images.extend((path, class_dir.name) for path in class_images)

    if not images:
        raise ValueError(f"No images found in {DATASET_DIR}")

    return images


# ---------------------------------
# 5. Build or reuse the test index
# ---------------------------------

def get_image_index():
    image_files = get_dataset_images()

    signature = [
        (str(path), path.stat().st_size, path.stat().st_mtime_ns, label)
        for path, label in image_files
    ]

    if CACHE_PATH.exists():
        try:
            cached = torch.load(
                CACHE_PATH,
                map_location="cpu",
                weights_only=True,
            )

            if cached.get("signature") == signature:
                return cached

        except (OSError, RuntimeError, ValueError, TypeError):
            pass

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = []
    valid_paths = []
    valid_labels = []

    total = len(image_files)
    print(f"Building similarity test index: {total} images")

    for start in range(0, total, BATCH_SIZE):
        batch = image_files[start:start + BATCH_SIZE]
        tensors = []
        batch_paths = []
        batch_labels = []

        for path, label in batch:
            try:
                with Image.open(path) as image:
                    tensor = similarity_transform(image.convert("RGB"))

                tensors.append(tensor)
                batch_paths.append(str(path))
                batch_labels.append(label)

            except (OSError, ValueError) as error:
                print(f"Skipping image {path}: {error}")

        if tensors:
            inputs = torch.stack(tensors).to(DEVICE)

            with torch.inference_mode():
                features = similarity_model(inputs)
                features = F.normalize(features, p=2, dim=1)

            embeddings.append(features.cpu().half())
            valid_paths.extend(batch_paths)
            valid_labels.extend(batch_labels)

        completed = min(start + BATCH_SIZE, total)
        print(f"Processed {completed}/{total} images")

    if not embeddings:
        raise ValueError("Could not extract features from dataset images.")

    index = {
        "signature": signature,
        "embeddings": torch.cat(embeddings, dim=0),
        "paths": valid_paths,
        "labels": valid_labels,
    }

    torch.save(index, CACHE_PATH)
    print(f"Similarity index saved to: {CACHE_PATH}")

    return index


# ---------------------------------
# 6. Search for similar images
# ---------------------------------

def search_similar_images(query_image, top_k=5):
    if not isinstance(query_image, Image.Image):
        raise TypeError("query_image must be a PIL Image.")

    index = get_image_index()

    query_embedding = extract_embedding(query_image).float()
    gallery_embeddings = index["embeddings"].float()

    scores = gallery_embeddings @ query_embedding

    k = min(max(1, int(top_k)), len(index["paths"]))
    top_scores, top_indices = torch.topk(scores, k=k)

    results = []

    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        results.append({
            "path": index["paths"][idx],
            "class_name": index["labels"][idx],
            "similarity": float(score),
        })

    return results
