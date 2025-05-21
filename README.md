# Demo of Interactive Text-to-Image Retrieval with Large Language Models: A Plug-and-Play Approach

This project is an implementation of [Interactive Text-to-Image Retrieval with Large Language Models: A Plug-and-Play Approach](https://arxiv.org/abs/2406.03411), which uses the BLIP model with LLM API to extract and retrieve image contexts based on text queries, leveraging image embeddings and dialogue-based retrieval.

## Quick Start

### Prerequisites
- Python 3.8+
- Install required packages:
  ```bash
  pip install -r requirements.txt
  ```
- Ensure you have a JSON file with image IDs and captions (e.g., `visdial_captions.json`).
- Have a directory with images (e.g., COCO dataset).

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/freshfish15/PlugIR_Demo.git
   cd PlugIR_Demo
   ```

2. Configure the `main.py` file:
   - Update the paths for your caption file and image directory:
     ```python
     caption_file_path = 'path/to/your/visdial_captions.json'
     image_directory = 'path/to/your/coco/images'
     ```
   - Set your API key and base URL:
     ```python
     API_KEY = "your-api-key"
     BASE_URL = "your-base-url"
     ```
   - Customize the query and retrieval parameters:
     ```python
     query = "your text query here"  # e.g., "people playing sports"
     num_top_similar = 16  # Number of initial similar images
     num_representatives = 8  # Number of representative images
     dialogue_round = 2  # Number of dialogue rounds
     ```

### Running the Project
Run the main script:
```bash
python main.py
```

This will:
1. Initialize the `ImageContextExtractor` with the specified BLIP model and embeddings file.
2. Perform dialogue-based image retrieval based on your query.
3. Output the final set of representative images.

### Notes
- The default BLIP model is `Salesforce/blip-image-captioning-base`. You can change it to another model like `Salesforce/blip-itm-base-coco` by passing it to the `ImageContextExtractor`.
- Ensure the `image_embeddings.pkl` file exists or is generated during the first run.
- The project uses unlabeled2017 COCO datasets for demo. You can download from https://cocodataset.org/#download or https://scidata.sjtu.edu.cn/records/7rp0x-d6e31
- The project uses visdial_captions.json as an image directory of unlabeled2017 COCO datasets. 

## Troubleshooting
- If you encounter API-related errors, verify your `API_KEY` and `BASE_URL`.
- Ensure the image directory and caption file paths are correct and accessible.
