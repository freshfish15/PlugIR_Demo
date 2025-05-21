from ImageContextExtractor import ImageContextExtractor
from Questioner import LLM_Connector


def one_round_retrieval(query, dialogue):
    print(f"\n--- Starting Contextual Caption Extraction for query: '{query}' ---")
    retrieved_context_captions = extractor.extract_contextual_captions(
        query_text=query,
        n_top_similar=num_top_similar,
        m_representatives=num_representatives,
        caption_file=caption_file_path,
        image_dir=image_directory
    )

    # --- Display results ---
    print("\n--- Retrieval Context (Generated Captions for Representatives) ---")
    if retrieved_context_captions:
        retrieved_img_with_generated_caption = dict()
        for img_id, caption_text in retrieved_context_captions.items():
            original_caption = extractor.load_dataset_info(caption_file_path, image_directory).get(img_id, {}).get('caption', 'N/A')
            retrieved_img_with_generated_caption[img_id] = caption_text
            print(f"Representative Image ID: {img_id}")
            print(f"  Original Caption: {original_caption}")
            print(f"  Generated Caption: {caption_text}\n")\
            # print(retrieved_img_with_generated_caption)

        print("\nBeginning generating question through LLM...")
        generator = LLM_Connector(api_key=API_KEY, base_url=BASE_URL)
        description_example = query
        # dialogue_example = []  # First turn, no dialogue
        question1 = generator.generate_question(retrieved_img_with_generated_caption, description_example, dialogue)
        if question1:
            # print("Generated Question:", question1) 
            answer = input(question1)
            dialogue.append({"Question" : question1, "Answer": answer})
            print(dialogue)
            filter1 = generator.filter_question(description_example, dialogue)
            if "Unvertain" or "uncertain" in filter1:
                print(f'filtering pass: {filter1}')
                return dialogue
            else:
                print(f"Generated question did not pass filtering: {filter1}")
                return dialogue
        else:
            print("Failed to generate question.")
            return 0
        
    else:
        print("No contextual captions were generated.")


def final_retrieval(query, dialogue):

    generator = LLM_Connector(api_key=API_KEY, base_url=BASE_URL)
    description_example = query
    # dialogue_example = []  # First turn, no dialogue
    reformulated_description = generator.reformulate(description_example, dialogue)
    if reformulated_description:
        print("Reformulated Description:", reformulated_description) 
    else:
        print("Failed to reformulate description.")
        return 0
    
    print(f"\n--- Starting Contextual Caption Extraction for query: '{reformulated_description}' ---")
    retrieved_context_captions = extractor.extract_contextual_captions(
        query_text=reformulated_description,
        n_top_similar=num_top_similar,
        m_representatives=num_representatives,
        caption_file=caption_file_path,
        image_dir=image_directory
    )

    # --- Display results ---
    print("\n--- Final Retrieval Context ---")
    print(f"\n--- Refomulated Query: {retrieved_context_captions} ---")
    if retrieved_context_captions:
        retrieved_img_with_generated_caption = dict()
        for img_id, caption_text in retrieved_context_captions.items():
            original_caption = extractor.load_dataset_info(caption_file_path, image_directory).get(img_id, {}).get('caption', 'N/A')
            retrieved_img_with_generated_caption[img_id] = caption_text
            print(f"Representative Image ID: {img_id}")
            print(f"  Original Caption: {original_caption}")
            print(f"  Generated Caption: {caption_text}\n")\
            # print(retrieved_img_with_generated_caption)
        
    else:
        print("No contextual captions were generated.")

if __name__ == "__main__":
    # --- Configuration ---
    # Path to your JSON file containing image IDs and original captions
    caption_file_path = 'caption/visdial_captions.json'# <--- UPDATE THIS PATH

    # Directory where your images are stored
    image_directory = '../datasets/coco' # <--- UPDATE THIS PATH

    API_KEY = ""
    BASE_URL = ""

    # --- Initialize the extractor ---
    # You can specify a different BLIP model if needed, e.g., "Salesforce/blip-itm-base-coco"
    extractor = ImageContextExtractor(embeddings_file='image_embeddings.pkl') # Uses "Salesforce/blip-image-captioning-base" by default

    # --- Parameters for extraction ---
    query = "some people in the picture" # Your text query
    num_top_similar = 16    # Number of initial similar images to retrieve
    num_representatives = 8 # Number of representative images to select and caption
    dialogue_round = 2
    
    init_dialogue = []
    for i in range(dialogue_round):
        init_dialogue =  one_round_retrieval(query, init_dialogue)
    final = final_retrieval(query, init_dialogue)