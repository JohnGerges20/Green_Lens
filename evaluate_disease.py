import torch
from pathlib import Path
from PIL import Image

from disease_detection import model, class_names, transform, DEVICE


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset" / "color"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

BATCH_SIZE = 32


# ============================================================
# Dataset folder -> Model class mapping
# ============================================================

DATASET_TO_MODEL = {
    "Apple___Apple_scab": "Apple_Apple_Scab",
    "Apple___Black_rot": "Apple_Black_Rot",
    "Apple___Cedar_apple_rust": "Apple_Cedar_Apple_Rust",
    "Apple___healthy": "Apple_Healthy",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        "Corn_(maize)_Cercospora_Leaf_Spot",

    "Corn_(maize)___Common_rust_":
        "Corn_(maize)_Common_Rust_",

    "Corn_(maize)___healthy":
        "Corn_(maize)_Healthy",

    "Corn_(maize)___Northern_Leaf_Blight":
        "Corn_(maize)_Northern_Leaf_Blight",

    "Tomato___Bacterial_spot":
        "Tomato_Bacterial_Spot",

    "Tomato___Early_blight":
        "Tomato_Early_Blight",

    "Tomato___healthy":
        "Tomato_Healthy",

    "Tomato___Late_blight":
        "Tomato_Late_Blight",

    "Tomato___Septoria_leaf_spot":
        "Tomato_Septoria_Leaf_Spot",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "Tomato_Yellow_Leaf_Curl_Virus",
}


# ============================================================
# Collect dataset images
# ============================================================

def collect_images():

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{DATASET_DIR}"
        )

    image_files = []

    for dataset_class in sorted(DATASET_TO_MODEL):

        folder = DATASET_DIR / dataset_class

        if not folder.exists():
            print(
                f"Warning: dataset folder not found: {dataset_class}"
            )
            continue

        model_class = DATASET_TO_MODEL[dataset_class]

        for path in sorted(folder.rglob("*")):

            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                image_files.append(
                    (path, model_class)
                )

    if not image_files:

        raise ValueError(
            f"No supported images found inside:\n{DATASET_DIR}"
        )

    return image_files


# ============================================================
# Evaluate model
# ============================================================

def evaluate():

    image_files = collect_images()

    print("=" * 70)
    print("GreenLens - Disease Model Evaluation")
    print("=" * 70)

    print(f"Dataset: {DATASET_DIR}")
    print(f"Images found: {len(image_files)}")
    print(f"Number of model classes: {len(class_names)}")
    print(f"Device: {DEVICE}")
    print()

    class_to_index = {
        name: index
        for index, name in enumerate(class_names)
    }

    total_images = 0
    top1_correct = 0
    top3_correct = 0

    class_total = {
        class_name: 0
        for class_name in class_names
    }

    class_correct = {
        class_name: 0
        for class_name in class_names
    }

    confusion_matrix = [
        [0 for _ in class_names]
        for _ in class_names
    ]

    failed_images = []

    model.eval()

    # ========================================================
    # Batch processing
    # ========================================================

    for start in range(
        0,
        len(image_files),
        BATCH_SIZE
    ):

        batch = image_files[
            start:start + BATCH_SIZE
        ]

        tensors = []
        valid_items = []

        for path, true_class in batch:

            try:

                with Image.open(path) as image:

                    image = image.convert("RGB")
                    tensor = transform(image)

                tensors.append(tensor)

                valid_items.append(
                    (path, true_class)
                )

            except Exception as error:

                failed_images.append(
                    (str(path), str(error))
                )

        if not tensors:
            continue

        inputs = torch.stack(tensors).to(DEVICE)

        with torch.inference_mode():

            outputs = model(inputs)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            _, top3_indices = torch.topk(
                probabilities,
                k=min(3, len(class_names)),
                dim=1
            )

        # ----------------------------------------------------
        # Analyze predictions
        # ----------------------------------------------------

        for i, (path, true_class) in enumerate(valid_items):

            true_index = class_to_index[true_class]

            predicted_index = int(
                top3_indices[i][0].item()
            )

            total_images += 1

            class_total[true_class] += 1

            # Top-1
            if predicted_index == true_index:

                top1_correct += 1
                class_correct[true_class] += 1

            # Top-3
            predicted_top3 = [
                int(index)
                for index in top3_indices[i].tolist()
            ]

            if true_index in predicted_top3:

                top3_correct += 1

            # Confusion matrix
            confusion_matrix[
                true_index
            ][predicted_index] += 1

        processed = min(
            start + BATCH_SIZE,
            len(image_files)
        )

        print(
            f"Processed {processed}/{len(image_files)}"
        )

    # ========================================================
    # Overall results
    # ========================================================

    if total_images == 0:

        raise RuntimeError(
            "No valid images could be evaluated."
        )

    top1_accuracy = (
        top1_correct / total_images
    )

    top3_accuracy = (
        top3_correct / total_images
    )

    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        f"Total valid images : {total_images}"
    )

    print(
        f"Top-1 Accuracy     : "
        f"{top1_accuracy:.2%}"
    )

    print(
        f"Top-3 Accuracy     : "
        f"{top3_accuracy:.2%}"
    )

    print(
        f"Failed images      : "
        f"{len(failed_images)}"
    )

    # ========================================================
    # Per-class accuracy
    # ========================================================

    print()
    print("=" * 70)
    print("PER-CLASS ACCURACY")
    print("=" * 70)

    for class_name in class_names:

        total = class_total[class_name]
        correct = class_correct[class_name]

        if total > 0:
            accuracy = correct / total
        else:
            accuracy = 0.0

        print(
            f"{class_name:<45} "
            f"{correct:>5}/{total:<5} "
            f"{accuracy:>7.2%}"
        )

    # ========================================================
    # Confusion Matrix
    # ========================================================

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print()
    print("Rows    = Actual class")
    print("Columns = Predicted class")
    print()

    print(
        "     "
        + " ".join(
            f"{i:>4}"
            for i in range(len(class_names))
        )
    )

    for i, class_name in enumerate(class_names):

        print(
            f"{i:>3}  "
            + " ".join(
                f"{value:>4}"
                for value in confusion_matrix[i]
            )
            + f"   {class_name}"
        )

    # ========================================================
    # Class mapping
    # ========================================================

    print()
    print("=" * 70)
    print("CLASS INDEX MAPPING")
    print("=" * 70)

    for index, class_name in enumerate(class_names):

        print(
            f"{index:>2} -> {class_name}"
        )

    # ========================================================
    # Failed images
    # ========================================================

    if failed_images:

        print()
        print("=" * 70)
        print("FAILED IMAGES")
        print("=" * 70)

        for path, error in failed_images[:20]:

            print(
                f"{path}\n"
                f"  Error: {error}"
            )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    try:

        evaluate()

    except Exception as error:

        print()
        print("=" * 70)
        print("EVALUATION FAILED")
        print("=" * 70)

        print(error)