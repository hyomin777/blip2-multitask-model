import os
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util


IMAGE_DIR = "/home/hyomin/workspace/k-digital/train/data/sentiment/positive"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_CSV = "filtered_results.csv"

CLASSES = [
    "person",
    "animal",
    "food",
    "travel",
    "interest",
    "anniversary",
    "item",
    "etc"
]


blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)
class_embeddings = embedder.encode(CLASSES, convert_to_tensor=True)

image_paths = []
for root, dirs, files in os.walk(IMAGE_DIR):
    for fname in files:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(root, fname))
results = []

for image_path in tqdm(image_paths):
    try:
        image = Image.open(image_path).convert("RGB")

        # Create Caption
        inputs = blip_processor(images=image, return_tensors="pt").to(DEVICE, torch.float16 if DEVICE == "cuda" else torch.float32)
        caption_ids = blip_model.generate(**inputs, max_new_tokens=20)
        caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)

        # Measure Embedding simularity
        caption_embedding = embedder.encode(caption, convert_to_tensor=True)
        similarities = util.cos_sim(caption_embedding, class_embeddings)[0]
        best_idx = torch.argmax(similarities).item()
        predicted_class = CLASSES[best_idx]
        similarity_score = similarities[best_idx].item()

        results.append({
            "filename": os.path.basename(image_path),
            "caption": caption,
            "predicted_class": predicted_class,
            "similarity": similarity_score
        })

    except Exception as e:
        print(f"[Error] {image_path}: {e}")
        continue

# Save csv
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
