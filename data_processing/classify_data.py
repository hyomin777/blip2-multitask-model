import os
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import CLIPProcessor, CLIPModel

IMAGE_DIR = "/home/hyomin/workspace/k-digital/train/data/sentiment/positive"
OUTPUT_CSV = "classification_results.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.15

CLASS_PROMPTS = {
    "소중한 사람": [
        "a photo of family",
        "a photo of friends",
        "a photo of a child",
        "a photo of children",
        "a photo of parents and children",
        "a photo of a parent",
        "a photo of a person you love",
        "a photo of a close friend",
        "a portrait of someone smiling at the camera",
    ],
    "반려 동물": [
        "a photo of a cat or dog",
        "a photo of a pet sitting alone",
        "a photo with an animal",
        "a photo with animals",
        "a photo with a pet",
        "a photo with pets",
        "a photo of an animal without people",
        "a photo of animals"
    ],
    "음식": [
        "a photo of food",
        "a photo of delicious food",
        "a close-up of a meal on a plate",
        "a photo of a dish at a restaurant",
    ],
    "특별한 장소": [
        "a scenic photo from a trip",
        "a photo of a beautiful travel destination",
        "a picture of a meaningful place outdoors",
    ],
    "취미": [
        "a photo of someone playing an instrument",
        "a photo related to music",
        "a photo related to movies",
        "a photo related to hobbies"
    ],
    "기념일": [
        "a birthday party photo",
        "a photo of a graduation or celebration",
        "a photo of a wedding",
        "a picture from a special day",
    ],
    "애장품": [
        "a photo of a personal item like a letter or trophy",
        "a close-up of a keepsake or award",
        "a photo of a memorable object",
        "a photo of a small item",
        "a photo of an object"
    ]
}

# Load model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Prompt list mapping
all_prompts = []
prompt_to_class = {}

for class_name, prompts in CLASS_PROMPTS.items():
    for prompt in prompts:
        all_prompts.append(prompt)
        prompt_to_class[prompt] = class_name

# Text processor
text_inputs = clip_processor(
    text=all_prompts, return_tensors="pt", padding=True
).to(DEVICE)

# Image data paths
image_paths = []
for root, _, files in os.walk(IMAGE_DIR):
    for fname in files:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(root, fname))


# Classify
results = []
for path in tqdm(image_paths, desc="Classifying with multi-prompts + 기타 fallback"):
    try:
        image = Image.open(path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = clip_model(**inputs, **text_inputs)
            logits = outputs.logits_per_image  # [1, num_prompts]
            probs = logits.softmax(dim=1)

            best_idx = probs.argmax(dim=1).item()
            best_prompt = all_prompts[best_idx]
            predicted_class = prompt_to_class[best_prompt]
            confidence = probs[0, best_idx].item()

            if confidence < CONFIDENCE_THRESHOLD:
                predicted_class = "기타"
                best_prompt = "below threshold"

        results.append({
            "filename": os.path.relpath(path, IMAGE_DIR),
            "predicted_class": predicted_class,
            "best_prompt": best_prompt,
            "confidence": confidence
        })

    except Exception as e:
        print(f"[Error] {path}: {e}")
        continue

# Save csv
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
