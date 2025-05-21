import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import json
import numpy as np
from tqdm import tqdm

class Retriever:
    """
    A class for performing text-image retrieval using CLIP.
    
    Attributes:
        model (CLIPModel): The pre-trained CLIP model.
        processor (CLIPProcessor): The CLIP processor for handling inputs.
        image_dir (str): The directory containing the pool of images.
        image_files (list): List of image file names (including subdirectories).
        image_features (torch.Tensor): Precomputed embeddings for all images.
        device (str): The device (CPU or GPU) to use for computations.
        captions (dict): Mapping from image_id to its caption.
    """

    def __init__(self, image_dir, captions_file, model_name="openai/clip-vit-base-patch32"):
        """
        Initialize the Retriever with an image directory, captions file, and a CLIP model.
        
        Args:
            image_dir (str): Path to the directory containing images (including subdirectories).
            captions_file (str): Path to the captions.json file.
            model_name (str): Name of the pre-trained CLIP model (default: "openai/clip-vit-base-patch32").
        """
        # Determine the device (GPU if available, else CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Store the image directory
        self.image_dir = image_dir
        
        # Load all image files recursively, including subdirectories
        self.image_files = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    rel_path = os.path.relpath(os.path.join(root, file), image_dir)
                    self.image_files.append(rel_path)
        
        # Load captions from captions.json
        with open(captions_file, 'r') as f:
            captions_data = json.load(f)
        self.captions = {item['id']: item['caption'][0] for item in captions_data}
        
        # Load all images and precompute their embeddings
        images = [Image.open(os.path.join(image_dir, f)).convert("RGB") for f in self.image_files]
        image_inputs = self.processor(images=images, return_tensors="pt")
        
        with torch.no_grad():
            pixel_values = image_inputs['pixel_values'].to(self.device)
            self.image_features = self.model.get_image_features(pixel_values=pixel_values).cpu()

    def retrieve_with_captions(self, query, n=5, batch_size=1000):
        """
        Retrieve the top n images from the pool that best match the given text query, along with their captions.
        
        Args:
            query (str): The text query for retrieval.
            n (int): Number of top-ranked images to return (default: 5).
            batch_size (int): Batch size for computing similarities (default: 1000).
        
        Returns:
            list: A list of dictionaries, each containing:
                - image_id (str): The file name of the image.
                - caption (str): The corresponding caption from captions.json.
                - score (float): The similarity score.
        """
        # Process the text query
        text_inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
        
        logit_scale = self.model.logit_scale.exp().item()
        
        # Compute similarities in batches with progress bar
        all_similarities = []
        for i in tqdm(range(0, len(self.image_features), batch_size), desc="Computing similarities", unit="batch"):
            batch_features = self.image_features[i:i+batch_size].to(self.device)
            similarities = (logit_scale * (batch_features @ text_features.T)).flatten().cpu()
            all_similarities.append(similarities)
        
        all_similarities = torch.cat(all_similarities)
        
        # Get the top n indices and their corresponding scores
        topk_scores, topk_indices = torch.topk(all_similarities, min(n, len(self.image_files)))
        
        # Convert indices to CPU and extract corresponding image files
        topk_indices = topk_indices.cpu().numpy()
        top_n_images = [self.image_files[i] for i in topk_indices]
        top_n_scores = topk_scores.cpu().numpy().tolist()
        
        # Create results with image_id, caption, and score
        top_n_results = []
        #form a complete img_id in json file
        img_id_prefix = 'unlabeled2017/'
        for img_id, score in zip(top_n_images, top_n_scores):
            complete_id = img_id_prefix + img_id
            print(f"complete_id: " + complete_id)
            caption = self.captions.get(complete_id, "No caption available")
            top_n_results.append({"image_id": complete_id, "caption": caption, "score": score})
        
        return top_n_results

    def retrieve(self, query, n=5, batch_size=1000):
        """
        Retrieve the top n images from the pool that best match the given text query.
        
        Args:
            query (str): The text query for retrieval.
            n (int): Number of top-ranked images to return (default: 5).
            batch_size (int): Batch size for computing similarities (default: 1000).
        
        Returns:
            tuple: A tuple containing:
                - top_n_images (list): List of file names of the top n images.
                - top_n_scores (list): List of similarity scores for the top n images.
        """
        # Process the text query
        text_inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)  # Keep on device
        
        logit_scale = self.model.logit_scale.exp().item()
        
        # Compute similarities in batches with progress bar
        all_similarities = []
        for i in tqdm(range(0, len(self.image_features), batch_size), desc="Computing similarities", unit="batch"):
            batch_features = self.image_features[i:i+batch_size].to(self.device)
            similarities = (logit_scale * (batch_features @ text_features.T)).flatten().cpu()
            all_similarities.append(similarities)
        
        all_similarities = torch.cat(all_similarities)
        
        # Get the top n indices and their corresponding scores
        topk_scores, topk_indices = torch.topk(all_similarities, min(n, len(self.image_files)))
        
        # Convert indices to CPU and extract corresponding image files
        topk_indices = topk_indices.cpu().numpy()
        top_n_images = [self.image_files[i] for i in topk_indices]
        top_n_scores = topk_scores.cpu().numpy().tolist()
        
        return top_n_images, top_n_scores


obj = Retriever('/home/shangrong/research/datasets/small_unlabeled2027', 'caption/visdial_captions.json')
print(str(obj.retrieve_with_captions('sea', 5)))
# print(obj.captions.get("unlabeled2017/000000000008.jpg", "not available"))