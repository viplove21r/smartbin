import os
import json
import numpy as np
import cv2
import tensorflow as tf

from pathlib import Path

from django.conf import settings
from django.shortcuts import render
from django.core.paginator import Paginator

from urllib.parse import quote_plus

from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST
from django.db import transaction
from django.utils import timezone

from .models import CompostItem, CompostPit, CompostPitItem
from .models import WasteScan

from django.shortcuts import get_object_or_404, redirect
from django.views.decorators.http import require_POST
from django.db.models import Count, Q


WASTE_GUIDE = {

    "cardboard": {
        "description": (
            "Cardboard is a paper-based packaging material commonly "
            "used in boxes, cartons, and packaging."
        ),
        "recyclability": "Generally Recyclable",
        "disposal": (
            "Flatten the cardboard, keep it clean and dry, "
            "and place it in the dry-waste collection stream."
        ),
        "bin": "Dry Waste – commonly Blue Bin",
        "tips": [
            "Remove plastic tape and other packaging where practical.",
            "Flatten cardboard boxes to save storage space.",
            "Do not mix wet, oily, or food-contaminated cardboard with clean recyclables."
        ],
        "youtube_query": "how to manage cardboard waste at home"
    },

    "glass": {
        "description": (
            "Glass waste includes bottles, jars, and other glass containers."
        ),
        "recyclability": "Generally Recyclable",
        "disposal": (
            "Clean and empty glass containers before placing them "
            "in the dry-waste collection stream."
        ),
        "bin": "Dry Waste – commonly Blue Bin",
        "tips": [
            "Rinse containers to remove food residue.",
            "Handle broken glass carefully and wrap it securely.",
            "Check whether your local collection service accepts broken glass."
        ],
        "youtube_query": "how to manage glass waste at home"
    },

    "metal": {
        "description": (
            "Metal waste includes aluminium cans, steel containers, "
            "and other metal packaging."
        ),
        "recyclability": "Generally Recyclable",
        "disposal": (
            "Keep metal items separate from wet waste and send them "
            "through the dry-waste collection or recycling system."
        ),
        "bin": "Dry Waste – commonly Blue Bin",
        "tips": [
            "Empty and rinse food or beverage containers.",
            "Separate metal packaging from other materials where practical.",
            "Do not place batteries or chemical containers with ordinary metal recyclables."
        ],
        "youtube_query": "how to reuse metal objects"
    },

    "paper": {
        "description": (
            "Paper waste includes newspapers, notebooks, office paper, "
            "and other paper-based products."
        ),
        "recyclability": "Generally Recyclable",
        "disposal": (
            "Keep paper dry and separate from food waste. "
            "Place it in the dry-waste collection stream."
        ),
        "bin": "Dry Waste – commonly Blue Bin",
        "tips": [
            "Reuse paper whenever possible.",
            "Keep recyclable paper clean and dry.",
            "Heavily soiled or food-contaminated paper may not be recyclable."
        ],
        "youtube_query": "Best out of waste from paper waste"
    },

    "plastic": {
        "description": (
            "Plastic waste includes bottles, containers, packaging, "
            "and other plastic products."
        ),
        "recyclability": "Depends on plastic type and local facilities",
        "disposal": (
            "Empty and rinse recyclable plastic containers. "
            "Keep them in the dry-waste collection stream and follow "
            "your local recycling guidelines."
        ),
        "bin": "Dry Waste – commonly Blue Bin",
        "tips": [
            "Empty containers before disposal.",
            "Rinse off food residue where practical.",
            "Check local recycling facilities because not every plastic type is accepted."
        ],
        "youtube_query": "ideas to reuse plastic containers"
    },

    "trash": {
        "description": (
            "This category represents miscellaneous waste that the model "
            "could not classify into the other five material categories."
        ),
        "recyclability": "Cannot be determined from this prediction",
        "disposal": (
            "Inspect the object before disposal. Separate recyclable "
            "materials if possible. Sanitary and household hazardous "
            "waste should be collected separately."
        ),
        "bin": "Depends on the actual item – do not assume it is ordinary waste",
        "tips": [
            "Check whether the item can be reused or recycled.",
            "Keep sanitary waste separate.",
            "Keep batteries, chemicals, and other special-care waste separate."
        ],
        "youtube_query": "how to segregate and manage household waste"
    }

}

MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "models",
    "waste_classifier.keras"
)

CLASS_PATH = os.path.join(
    settings.BASE_DIR,
    "models",
    "class_names.json"
)

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)

# Create feature extractor from the trained model
feature_extractor = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer("dense_1").output
)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)



def preprocess_image(image_file):

    #its fun is to only preprocess the image user gives

    image_bytes = np.frombuffer(
        image_file.read(),
        np.uint8
    )

    img = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise ValueError("Invalid image")

    # BGR to RGB
    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    img = cv2.resize(
        img,
        (224, 224)
    )

    # Convert to float32
    img = img.astype(np.float32)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

def predict_waste(request):

    #this is the one accessing the model

    context = {}

    if request.method == "POST" and request.FILES.get("image"):

        image_file = request.FILES["image"]

        try:

            img = preprocess_image(image_file)

            predictions = model.predict(
                img,
                verbose=0
            )

            predicted_index = np.argmax(
                predictions[0]
            )

            confidence = float(
                predictions[0][predicted_index]
            ) * 100

            predicted_class = class_names[predicted_index]


            # Save this prediction in the database
            uploaded_image = request.FILES.get("image")

            if uploaded_image:
                uploaded_image.seek(0)

            scan = WasteScan.objects.create(
            user=request.user if request.user.is_authenticated else None,
            image=uploaded_image,
            predicted_class=predicted_class.lower(),
            confidence=confidence
        )

            similarity_result = get_similarity_result(
    img,
    current_scan_id=scan.id
)


            # Get information about the predicted waste category
            waste_details = WASTE_GUIDE.get(predicted_class.lower())

            if waste_details:
                youtube_url = (
                "https://www.youtube.com/results?search_query="
                + quote_plus(waste_details["youtube_query"])
            )
            else:
                youtube_url = None

            can_compost = predicted_class.lower() in [
                "paper",
                "cardboard"
]
            context = {
                "prediction": predicted_class,
                "confidence": round(confidence, 2),
                "waste_details":waste_details,
                "youtube_url":youtube_url,
                "can_compost":can_compost,
                "predicted_class":predicted_class,
                "scan": scan,
                "similarity_result": similarity_result,

            }

        except Exception:
            context = {
                "error": "Unable to process this image. Please try another."
            }

    return render(
        request,
        "classifier/index.html",
        context
    )


def dataset_view(request):

    dataset_dir = (
        Path(__file__).resolve().parent
        / "static"
        / "classifier"
        / "dataset_gallery"
    )

    manifest_path = dataset_dir / "dataset_manifest.json"

    with open(manifest_path, "r") as f:
        data = json.load(f)

    splits = ["train", "validation", "test"]
    statistics = data["statistics"]

    # Actual dataset size
    dataset_total = sum(
        statistics[split]["original_total"]
        for split in splits
    )

    # Number of images exported for the gallery
    gallery_total = sum(
        statistics[split]["total"]
        for split in splits
    )

    # Representative image and gallery count per category
    sample_images = {}
    category_gallery_counts = {}

    for split in splits:
        for item in data[split]:
            class_name = item["label"]

            if class_name not in sample_images:
                sample_images[class_name] = item["image"]

        for class_name, count in statistics[split]["classes"].items():
            category_gallery_counts[class_name] = (
                category_gallery_counts.get(class_name, 0) + count
            )

    category_overview = []

    for class_name in sorted(sample_images.keys()):
        category_overview.append({
            "name": class_name,
            "sample_image": sample_images[class_name],
            "gallery_count": category_gallery_counts.get(class_name, 0),
        })

    # Selected split
    split = request.GET.get("split", "train")

    if split not in splits:
        split = "train"

    images = data[split]

    # Class filter
    selected_class = request.GET.get("class", "")

    if selected_class:
        images = [
            item for item in images
            if item["label"] == selected_class
        ]

    # Pagination
    paginator = Paginator(images, 24)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    class_distribution = statistics[split]["classes"]

    context = {
        "page_obj": page_obj,
        "split": split,
        "statistics": statistics,
        "class_distribution": class_distribution,
        "selected_class": selected_class,

        "dataset_total": dataset_total,
        "gallery_total": gallery_total,
        "category_overview": category_overview,
    }

    return render(request, "classifier/dataset.html", context)

def model_info(request):
    return render(request, "classifier/model_info.html")



#compost bin

@require_POST
def add_compost_item(request):

    waste_type = request.POST.get("waste_type", "").lower()

    if waste_type not in ["paper", "cardboard"]:
        return redirect("predict_waste")

    CompostItem.objects.create(
        waste_type=waste_type
    )

    return redirect("compost_bin")

def compost_bin(request):

    available_items = CompostItem.objects.filter(
        is_in_pit=False
    ).order_by("-added_at")

    pits = CompostPit.objects.all().order_by("-created_at")

    return render(request, "classifier/compost.html", {
        "available_items": available_items,
        "pits": pits,
    })


@require_POST
def create_compost_pit(request):

    selected_ids = request.POST.getlist("selected_items")

    if not selected_ids:
        return redirect("compost_bin")

    with transaction.atomic():

        items = CompostItem.objects.filter(
            id__in=selected_ids,
            is_in_pit=False
        )

        if not items.exists():
            return redirect("compost_bin")

        start = timezone.now()

        pit = CompostPit.objects.create(
            name=f"Compost Pit {CompostPit.objects.count() + 1}",
            start_time=start,
            end_time=start + timezone.timedelta(days=30)
        )

        for item in items:
            CompostPitItem.objects.create(
                pit=pit,
                item=item
            )

            item.is_in_pit = True
            item.save(update_fields=["is_in_pit"])

    return redirect("compost_bin")


@require_POST
def remove_compost_item(request, item_id):

    item = get_object_or_404(
        CompostItem,
        id=item_id,
        is_in_pit=False
    )

    item.delete()

    return redirect("compost_bin")

@require_POST
def delete_compost_pit(request, pit_id):

    pit = get_object_or_404(CompostPit, id=pit_id)

    with transaction.atomic():

        # Return all items to the available compost bin
        for pit_item in pit.items.select_related("item").all():
            pit_item.item.is_in_pit = False
            pit_item.item.save(update_fields=["is_in_pit"])

        # Delete pit and its CompostPitItem relationships
        pit.delete()

    return redirect("compost_bin")


@require_POST
def submit_feedback(request):

    scan_id = request.POST.get("scan_id")
    feedback = request.POST.get("feedback")

    scan = get_object_or_404(WasteScan, id=scan_id)

    # Prevent changing feedback after it has been submitted
    if scan.feedback is not None:
        return redirect("my_waste")

    # Determine actual category
    if feedback == "correct":

        actual_class = scan.predicted_class

    elif feedback == "incorrect":

        actual_class = request.POST.get("actual_class")

        valid_classes = dict(WasteScan.WASTE_CHOICES)

        if actual_class not in valid_classes:
            return redirect("my_waste")

        if actual_class == scan.predicted_class:
            return redirect("my_waste")

    else:
        return redirect("my_waste")

    # Extract image embedding
    embedding = None

    if scan.image:

        try:
            scan.image.open("rb")

            img = preprocess_image(scan.image)

            features = feature_extractor.predict(
                img,
                verbose=0
            )

            embedding = features[0].astype(float).tolist()

        except Exception as e:
            print("Feature extraction error:", e)

        finally:
            scan.image.close()

    # Save everything
    scan.feedback = feedback
    scan.actual_class = actual_class
    scan.embedding = embedding

    scan.save(
        update_fields=[
            "feedback",
            "actual_class",
            "embedding"
        ]
    )

    return redirect("my_waste")

def my_waste(request):

    scans = WasteScan.objects.all().order_by("-scanned_at")

    total_scans = scans.count()

    correct_count = scans.filter(feedback="correct").count()

    incorrect_count = scans.filter(feedback="incorrect").count()

    feedback_count = correct_count + incorrect_count

    accuracy = (
        (correct_count / feedback_count) * 100
        if feedback_count > 0 else 0
    )

    feedback_coverage = (
        (feedback_count / total_scans) * 100
        if total_scans > 0 else 0
    )

    # Statistics for all six waste categories
    category_stats = scans.values(
        "predicted_class"
    ).annotate(
        total=Count("id"),
        correct=Count(
            "id",
            filter=Q(feedback="correct")
        ),
        incorrect=Count(
            "id",
            filter=Q(feedback="incorrect")
        )
    ).order_by("predicted_class")

    context = {
        "scans": scans,
        "total_scans": total_scans,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "accuracy": round(accuracy, 2),
        "feedback_coverage": round(feedback_coverage, 2),
        "category_stats": category_stats,
        "waste_choices": WasteScan.WASTE_CHOICES,
    }

    return render(request, "classifier/my_waste.html", context)



def get_similarity_result(image_tensor, current_scan_id=None, top_k=5):

    # Generate the 128-dimensional embedding for the uploaded image
    new_embedding = feature_extractor.predict(
        image_tensor,
        verbose=0
    )[0]

    new_embedding = np.array(new_embedding, dtype=np.float32)

    # Retrieve previous scans for which feedback and embeddings exist
    reference_scans = WasteScan.objects.exclude(
        embedding__isnull=True
    ).exclude(
        actual_class__isnull=True
    )

    if current_scan_id is not None:
        reference_scans = reference_scans.exclude(id=current_scan_id)

    reference_scans = list(reference_scans)

    if not reference_scans:
        return None

    # Convert stored embeddings into a NumPy array
    stored_embeddings = np.array(
        [scan.embedding for scan in reference_scans],
        dtype=np.float32
    )

    # Calculate cosine similarity
    new_norm = np.linalg.norm(new_embedding)
    stored_norms = np.linalg.norm(
        stored_embeddings,
        axis=1
    )

    if new_norm == 0:
        return None

    similarities = np.dot(
        stored_embeddings,
        new_embedding
    ) / (stored_norms * new_norm + 1e-10)

    # Select the top K most similar examples
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Weighted voting based on similarity
    class_votes = {}

    for index in top_indices:

        actual_class = reference_scans[index].actual_class
        similarity = float(similarities[index])

        # Convert cosine similarity from [-1, 1] to [0, 1]
        weight = (similarity + 1) / 2

        class_votes[actual_class] = (
            class_votes.get(actual_class, 0) + weight
        )

    total_votes = sum(class_votes.values())

    if total_votes == 0:
        return None

    similarity_class = max(
        class_votes,
        key=class_votes.get
    )

    vote_share = (
        class_votes[similarity_class] / total_votes
    ) * 100

    best_similarity = float(
        similarities[top_indices[0]]
    ) * 100

    return {
        "similarity_class": similarity_class,
        "similarity_score": round(vote_share, 2),
        "best_similarity": round(best_similarity, 2),
        "match_count": len(top_indices),
    }





