import os
import json
import pickle
import numpy as np
from PIL import Image
import torch
from transformers import BlipModel, BlipProcessor, BlipForConditionalGeneration
from sklearn.cluster import KMeans
from tqdm import tqdm


class ImageContextExtractor:
    def __init__(self, model_name_or_path="Salesforce/blip-image-captioning-base", embeddings_file="image_embeddings.pkl"):
        """
        Initializes the ImageContextExtractor with BLIP models, processor, and embeddings file path.

        Args:
            model_name_or_path (str): The Hugging Face model name or path to load.
            embeddings_file (str): Path to the file where image embeddings will be saved/loaded.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.processor = BlipProcessor.from_pretrained(model_name_or_path)
        self.blip_model = BlipModel.from_pretrained(model_name_or_path).to(self.device)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(model_name_or_path).to(self.device)

        # Set models to evaluation mode
        self.blip_model.eval()
        self.caption_model.eval()
        print("BLIP models and processor loaded successfully.")

        self.embeddings_file = embeddings_file

    def _load_image(self, image_path):
        """Helper function to load an image from a path."""
        if not os.path.exists(image_path):
            # print(f"Warning: Image path not found: {image_path}")
            return None
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def load_dataset_info(self, caption_file, image_dir):
        """
        Loads image IDs, paths, and original captions from a JSON file.

        Args:
            caption_file (str): Path to the JSON file containing caption data.
            image_dir (str): Directory where images are stored.

        Returns:
            dict: A dictionary mapping image_id to {'path': str, 'caption': str}.
                  Returns an empty dict if loading fails.
        """
        image_info_map = {}
        try:
            with open(caption_file, 'r') as f:
                captions_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Caption file not found at {caption_file}")
            return image_info_map
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {caption_file}")
            return image_info_map

        for item in captions_data:
            img_id = item.get('id')
            if img_id is None:
                print(f"Warning: Item found with no 'id'. Skipping: {item}")
                continue
            
            img_id_str = str(img_id)  # Ensure img_id is a string for dict keys and filenames

            # Construct image path. Assuming img_id from JSON is the filename (e.g., "image1.jpg")
            img_path = os.path.join(image_dir, img_id_str)

            # Safely get the first caption, provide default if 'caption' key is missing or list is empty
            original_captions_list = item.get('caption')
            if isinstance(original_captions_list, list) and original_captions_list:
                caption = original_captions_list[0]
            else:
                caption = ""  # Default caption

            image_info_map[img_id_str] = {'path': img_path, 'caption': caption}
        return image_info_map

    def _load_embeddings(self):
        """
        Loads image embeddings from the embeddings file if it exists.

        Returns:
            dict: A dictionary mapping image_id to its normalized 1D embedding, or None if file doesn't exist or fails to load.
        """
        if not os.path.exists(self.embeddings_file):
            print(f"No embeddings file found at {self.embeddings_file}. Will compute embeddings.")
            return None
        try:
            with open(self.embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"Loaded embeddings for {len(embeddings)} images from {self.embeddings_file}.")
            return embeddings
        except Exception as e:
            print(f"Error loading embeddings from {self.embeddings_file}: {e}. Will recompute embeddings.")
            return None

    def _save_embeddings(self, image_embeddings):
        """
        Saves image embeddings to the embeddings file.

        Args:
            image_embeddings (dict): A dictionary mapping image_id to its normalized 1D embedding.
        """
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(image_embeddings, f)
            print(f"Saved embeddings for {len(image_embeddings)} images to {self.embeddings_file}.")
        except Exception as e:
            print(f"Error saving embeddings to {self.embeddings_file}: {e}")

    def compute_image_embeddings(self, image_info_map, force_recompute=False):
        """
        Computes or loads normalized [CLS] token embeddings for images.

        Args:
            image_info_map (dict): Output from load_dataset_info.
            force_recompute (bool): If True, recomputes embeddings even if a saved file exists.

        Returns:
            dict: A dictionary mapping image_id to its normalized 1D embedding.
        """
        if not force_recompute:
            saved_embeddings = self._load_embeddings()
            if saved_embeddings is not None:
                # # Verify that all images in image_info_map have embeddings
                # missing_ids = [img_id for img_id in image_info_map if img_id not in saved_embeddings]
                # if not missing_ids:
                #     print("All images have precomputed embeddings. Skipping computation.")
                #     return saved_embeddings
                # else:
                #     print(f"Found {len(missing_ids)} images without embeddings. Computing embeddings for all images.")
                return saved_embeddings

        image_embeddings = {}

        # Wrap the loop with tqdm for a progress bar
        for img_id, info in tqdm(image_info_map.items(), desc="Computing embeddings", unit="image"):
            img_path = info['path']
            image = self._load_image(img_path)
            if image is None:
                # print(f"Skipping embedding for image ID {img_id} due to loading error.")
                continue

            inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
            with torch.no_grad():
                vision_outputs = self.blip_model.vision_model(pixel_values=inputs.pixel_values)
                # Get CLS token embedding: shape (1, hidden_dim)
                image_embed = vision_outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # Normalize the embedding (L2 norm)
            norm = np.linalg.norm(image_embed, axis=1, keepdims=True)
            if norm == 0:
                # print(f"Warning: Zero norm for image embedding {img_id}. Skipping.")
                continue
            
            image_embed_normalized = image_embed / norm
            image_embeddings[img_id] = image_embed_normalized.squeeze()  # Store as 1D array (hidden_dim,)


        # Save the computed embeddings
        if image_embeddings:
            self._save_embeddings(image_embeddings)
        
        return image_embeddings

    def get_text_embedding(self, text):
        """
        Computes normalized [CLS] token embedding for a given text.

        Args:
            text (str): The input text.

        Returns:
            np.ndarray: The normalized 2D text embedding (1, hidden_dim).
                        Returns None if embedding fails.
        """
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_outputs = self.blip_model.text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            # Get CLS token embedding: shape (1, hidden_dim)
            text_embed = text_outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Normalize the embedding
        norm = np.linalg.norm(text_embed, axis=1, keepdims=True)
        if norm == 0:
            print(f"Warning: Zero norm for text embedding '{text}'.")
            return np.zeros_like(text_embed)
            
        text_embed_normalized = text_embed / norm
        return text_embed_normalized  # Shape (1, hidden_dim)

    def get_top_n_similar_images(self, query_text, n, image_embeddings_map):
        """
        Finds the top N images most similar to the query text.

        Args:
            query_text (str): The text query.
            n (int): Number of top similar images to retrieve.
            image_embeddings_map (dict): Map of image_id to its 1D embedding.

        Returns:
            tuple: (top_n_ids, top_n_embs_array, top_n_sims_map)
                   top_n_ids (list): List of image IDs.
                   top_n_embs_array (np.ndarray): Array of embeddings for top N images.
                   top_n_sims_map (dict): Map of top N image_id to similarity score.
                   Returns empty lists/arrays/dicts if no images or embeddings.
        """
        if not image_embeddings_map:
            return [], np.array([]), {}

        query_text_embed = self.get_text_embedding(query_text)
        if query_text_embed is None:
            print("Error: Could not compute text embedding for query.")
            return [], np.array([]), {}

        img_ids_list = list(image_embeddings_map.keys())
        img_embs_list = np.array([image_embeddings_map[id] for id in img_ids_list if id in image_embeddings_map])
        
        if img_embs_list.size == 0:
            print("Warning: No valid image embeddings to compare against.")
            return [], np.array([]), {}

        similarities_scores = np.dot(query_text_embed, img_embs_list.T).flatten()
        num_available_scores = len(similarities_scores)
        actual_n = min(n, num_available_scores)
        
        if actual_n == 0:
            return [], np.array([]), {}

        sorted_indices = np.argsort(similarities_scores)[::-1][:actual_n]
        top_n_ids = [img_ids_list[i] for i in sorted_indices]
        top_n_embs_array = img_embs_list[sorted_indices]
        top_n_sims_map = {img_ids_list[i]: float(similarities_scores[i]) for i in sorted_indices}
        
        return top_n_ids, top_n_embs_array, top_n_sims_map

    def select_representative_images(self, top_n_ids, top_n_embs_array, top_n_sims_map, m):
        """
        Selects M representative images from the top N using KMeans clustering.

        Args:
            top_n_ids (list): List of top N image IDs.
            top_n_embs_array (np.ndarray): Embeddings of top N images.
            top_n_sims_map (dict): Similarity scores of top N images to the query.
            m (int): Number of representatives (clusters) to select.

        Returns:
            list: List of M representative image IDs.
        """
        if not top_n_ids or m <= 0:
            return []
        
        num_available_samples = len(top_n_ids)
        if m >= num_available_samples:
            print(f"Warning: Requested {m} representatives, but only {num_available_samples} available. Returning all.")
            return top_n_ids 

        kmeans = KMeans(n_clusters=m, random_state=42, n_init='auto')
        try:
            clusters = kmeans.fit_predict(top_n_embs_array)
        except ValueError as e:
            print(f"Error during KMeans fitting: {e}. Returning top {m} images by similarity instead.")
            return top_n_ids[:m]

        representatives = []
        for cluster_id in range(m):
            cluster_indices_in_top_n = [i for i, c_label in enumerate(clusters) if c_label == cluster_id]
            if cluster_indices_in_top_n:
                max_sim_in_cluster = -float('inf')
                representative_for_cluster = None
                for idx_in_top_n in cluster_indices_in_top_n:
                    img_id = top_n_ids[idx_in_top_n]
                    sim = top_n_sims_map[img_id]
                    if sim > max_sim_in_cluster:
                        max_sim_in_cluster = sim
                        representative_for_cluster = img_id
                if representative_for_cluster and representative_for_cluster not in representatives:
                    representatives.append(representative_for_cluster)
        return representatives

    def generate_captions_for_images(self, representative_ids, image_info_map):
        """
        Generates new captions for a list of representative image IDs.

        Args:
            representative_ids (list): List of image IDs to caption.
            image_info_map (dict): Full dataset info to get image paths.

        Returns:
            dict: A dictionary mapping representative image_id to its newly generated caption.
        """
        generated_captions_map = {}
        for img_id in representative_ids:
            if img_id not in image_info_map:
                print(f"Warning: Image ID {img_id} not found in image_info_map for captioning.")
                continue

            img_path = image_info_map[img_id]['path']
            image = self._load_image(img_path)
            if image is None:
                print(f"Skipping caption generation for {img_id} (image not loaded).")
                continue
            
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.caption_model.generate(**inputs)
            caption_text = self.processor.decode(out[0], skip_special_tokens=True)
            generated_captions_map[img_id] = caption_text.strip()
        return generated_captions_map

    def extract_contextual_captions(self, query_text, n_top_similar, m_representatives, caption_file, image_dir, force_recompute_embeddings=False):
        """
        Main orchestration method.

        Args:
            query_text (str): The text query for similarity search.
            n_top_similar (int): Number of top similar images to retrieve initially.
            m_representatives (int): Number of representative images to select via clustering.
            caption_file (str): Path to the JSON file with dataset caption info.
            image_dir (str): Directory containing the dataset images.
            force_recompute_embeddings (bool): If True, recomputes embeddings even if a saved file exists.

        Returns:
            dict: A dictionary mapping representative image IDs to their newly generated captions.
        """
        print("Step 1: Loading dataset info...")
        image_info_map = self.load_dataset_info(caption_file, image_dir)
        if not image_info_map:
            print("Critical Error: Failed to load dataset info or dataset is empty. Aborting.")
            return {}

        print("\nStep 2: Computing or loading image embeddings for the dataset...")
        image_embeddings_map = self.compute_image_embeddings(image_info_map, force_recompute=False)
        if not image_embeddings_map:
            print("Critical Error: Failed to compute/load image embeddings or no valid images found. Aborting.")
            return {}
        print(f"Using embeddings for {len(image_embeddings_map)} images.")

        print(f"\nStep 3: Finding top {n_top_similar} similar images for query: '{query_text}'...")
        top_n_ids, top_n_embs_array, top_n_sims_map = self.get_top_n_similar_images(
            query_text, n_top_similar, image_embeddings_map
        )
        if not top_n_ids:
            print("No similar images found for the query. Aborting.")
            return {}
        print(f"Found {len(top_n_ids)} similar images.")

        print(f"\nStep 4: Selecting {m_representatives} representative images from the top {len(top_n_ids)}...")
        representative_ids = self.select_representative_images(
            top_n_ids, top_n_embs_array, top_n_sims_map, m_representatives
        )
        if not representative_ids:
            print("No representative images were selected. Aborting.")
            return {}
        print(f"Selected {len(representative_ids)} representative image IDs: {representative_ids}")

        print("\nStep 5: Generating captions for representative images...")
        final_captions = self.generate_captions_for_images(representative_ids, image_info_map)
        print(f"Generated captions for {len(final_captions)} images.")
        
        return final_captions


if __name__ == "__main__":
    # --- Configuration ---
    caption_file_path = 'caption/visdial_captions.json'  # UPDATE THIS PATH
    image_directory = './'  # UPDATE THIS PATH
    embeddings_file_path = 'image_embeddings.pkl'  # Path to save/load embeddings

    # --- Initialize the extractor ---
    extractor = ImageContextExtractor(embeddings_file=embeddings_file_path)

    # --- Parameters for extraction ---
    query = "a snowboarder on a rail in the snow"
    num_top_similar = 10
    num_representatives = 5
    force_recompute = False  # Set to True to force recomputation of embeddings

    # --- Run the extraction process ---
    print(f"\n--- Starting Contextual Caption Extraction for query: '{query}' ---")
    retrieved_context_captions = extractor.extract_contextual_captions(
        query_text=query,
        n_top_similar=num_top_similar,
        m_representatives=num_representatives,
        caption_file=caption_file_path,
        image_dir=image_directory,
        force_recompute_embeddings=force_recompute
    )

    # --- Display results ---
    print("\n--- Retrieval Context (Generated Captions for Representatives) ---")
    if retrieved_context_captions:
        for img_id, caption_text in retrieved_context_captions.items():
            original_caption = extractor.load_dataset_info(caption_file_path, image_directory).get(img_id, {}).get('caption', 'N/A')
            print(f"Representative Image ID: {img_id}")
            print(f"  Original Caption: {original_caption}")
            print(f"  Generated Caption: {caption_text}\n")
    else:
        print("No contextual captions were generated.")