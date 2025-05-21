import os
import json
from PIL import Image
import torch
from transformers import BlipModel, BlipProcessor, BlipForConditionalGeneration
from sklearn.cluster import KMeans
import numpy as np

# Load Models
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Load Dataset
def load_dataset(caption_file, image_dir):
    with open(caption_file, 'r') as f:
        captions_data = json.load(f)
    image_info = {}
    for item in captions_data:
        img_id = item['id']
        img_path = os.path.join(image_dir, img_id)
        caption = item['caption'][0]
        image_info[img_id] = {'path': img_path, 'caption': caption}
    return image_info

# Compute Image Embeddings
def compute_image_embeddings(image_info, blip_model, processor, device):
    image_embeddings = {}
    for img_id, info in image_info.items():
        img_path = info['path']
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=[image], return_tensors="pt").to(device)
            with torch.no_grad():
                vision_outputs = blip_model.vision_model(pixel_values=inputs.pixel_values)
                image_embed = vision_outputs.last_hidden_state[0,0,:].cpu().numpy()  # [CLS] token
                image_embed = image_embed / np.linalg.norm(image_embed)  # Normalize
            image_embeddings[img_id] = image_embed
    return image_embeddings

# Get Text Embedding
def get_text_embedding(blip_model, processor, text, device):
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        # text_outputs = blip_model.text_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        text_outputs = blip_model.text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        text_embed = text_outputs.last_hidden_state[0,0,:].cpu().numpy()  # [CLS] token
        text_embed = text_embed / np.linalg.norm(text_embed)  # Normalize
    return text_embed

# Get Top-n Similar Images
def get_top_n_similar(C_t, n, image_embeddings, blip_model, processor, device):
    text_embed = get_text_embedding(blip_model, processor, C_t, device)
    similarities = []
    for img_id, img_emb in image_embeddings.items():
        sim = np.dot(text_embed, img_emb)
        similarities.append((img_id, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_n = similarities[:n]
    top_n_ids = [x[0] for x in top_n]
    top_n_embs = np.array([image_embeddings[x[0]] for x in top_n])
    top_n_sims = {x[0]: x[1] for x in top_n}
    return top_n_ids, top_n_embs, top_n_sims

# Select Representatives
def select_representatives(top_n_ids, top_n_embs, top_n_sims, m):
    kmeans = KMeans(n_clusters=m)
    clusters = kmeans.fit_predict(top_n_embs)
    representatives = []
    for cluster in range(m):
        cluster_idx = np.where(clusters == cluster)[0]
        if len(cluster_idx) > 0:
            sims_in_cluster = [top_n_sims[top_n_ids[i]] for i in cluster_idx]
            max_sim_idx_in_cluster = cluster_idx[np.argmax(sims_in_cluster)]
            rep_id = top_n_ids[max_sim_idx_in_cluster]
            representatives.append(rep_id)
    return representatives

# Generate Captions
def generate_captions(representative_ids, caption_model, processor, image_dir):
    captions = {}
    for img_id in representative_ids:
        img_path = os.path.join(image_dir, img_id)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            out = caption_model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            # captions.append(caption)
            captions[img_id] = caption
    return captions

# Main Function
def retrieval_context_extraction(C_t, n, m, caption_file, image_dir):
    image_info = load_dataset(caption_file, image_dir)
    image_embeddings = compute_image_embeddings(image_info, blip_model, processor, device)
    top_n_ids, top_n_embs, top_n_sims = get_top_n_similar(C_t, n, image_embeddings, blip_model, processor, device)
    rep_ids = select_representatives(top_n_ids, top_n_embs, top_n_sims, m)
    captions = generate_captions(rep_ids, caption_model, processor, image_dir)
    return captions

# Example Usage
if __name__ == "__main__":
    caption_file = 'caption/visdial_captions.json'
    image_dir = './'
    C_t = "a bird flying through the air"
    n = 10
    m = 3
    retrieval_context = retrieval_context_extraction(C_t, n, m, caption_file, image_dir)
    print("Retrieval Context (Captions):")
    for i in retrieval_context:
        print(f'rep_ids: {i}   caption: {retrieval_context[i]}')