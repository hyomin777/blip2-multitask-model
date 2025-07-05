import os
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import CLIPProcessor, CLIPModel

IMAGE_DIR = "/home/hyomin/workspace/k-digital/data"
OUTPUT_CSV = "sentiment.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.7

CLASS_PROMPTS = {
    "positive": [
        "a positive photo",
        "happy photo",
        "a neutral photo",
        "neutral image"
    ],
    "negative": [
        "a scary photo",
        "a creepy photo",
        "a terrifying scene",
        "a frightening photo",
        "a negative photo",
    ]
}

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

all_prompts = []
class_to_prompt_indices = {}
idx = 0
for class_name, prompts in CLASS_PROMPTS.items():
    indices = list(range(idx, idx + len(prompts)))
    class_to_prompt_indices[class_name] = indices
    all_prompts.extend(prompts)
    idx += len(prompts)

text_inputs = clip_processor(
    text=all_prompts, return_tensors="pt", padding=True
).to(DEVICE)

image_paths = []
for root, _, files in os.walk(IMAGE_DIR):
    for fname in files:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(root, fname))

results = []
for path in tqdm(image_paths):
    try:
        image = Image.open(path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = clip_model(**inputs, **text_inputs)
            sims = outputs.logits_per_image[0]  # shape: [num_prompts]

            avg_scores = {
                cls: sims[indices].mean().item()
                for cls, indices in class_to_prompt_indices.items()
            }

            # softmax
            score_tensor = torch.tensor(list(avg_scores.values()), device=DEVICE)
            probs = torch.softmax(score_tensor, dim=0)

            class_names = list(avg_scores.keys())
            best_idx = torch.argmax(probs).item()
            predicted_class = class_names[best_idx]
            confidence = probs[best_idx].item()

            if confidence < CONFIDENCE_THRESHOLD:
                predicted_class = "neutral"

        results.append({
            "filename": os.path.relpath(path, IMAGE_DIR),
            "predicted_class": predicted_class,
            "positive_prob": probs[class_names.index("positive")].item(),
            "negative_prob": probs[class_names.index("negative")].item(),
            "confidence": confidence
        })

    except Exception as e:
        print(f"[Error] {path}: {e}")
        continue

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
