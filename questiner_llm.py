from openai import OpenAI

# client = OpenAI(api_key="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")
client = OpenAI(api_key="sk-mR9xMzFRbOWD21nzrCC8mnjTXYeIxEaKVho0ftGXw5VzQ3VM", base_url="https://tbnx.plus7.plus/v1")
# Set up OpenAI API key (replace with your actual API key)

def generate_question(retrieval_context, description, dialogue):
    """
    Generate a question using the OpenAI API based on the PlugIR paper's Table 18 prompt template.
    
    Args:
        retrieval_context (dict): Dictionary mapping image IDs to generated captions.
        description (str): Initial user description (D0).
        dialogue (list): List of (question, answer) tuples; empty for first turn.
    
    Returns:
        str: The generated question, or None if an error occurs.
    """
    # Convert retrieval_context to a list of captions
    retrieval_candidates = list(retrieval_context.values())

    # System message: Task description from Table 18
    system_message = """
    You are a proficient question generator tasked with aiding in the retrieval of a target image. Your role is to generate questions about the target image of the description via leveraging three key information sources:
    [Retrieval Candidates]: These are captions of images which are the candidates of the retrieval task for the target image described in [Description]. [Description]: This is a concise explanation of the target image.
    [Dialogue]: Comprising question and answer pairs that seek additional details about the target image.
    You should craft a question that narrows down the options for the attributes of the target image through drawing the information from the retrieval candidates. The generated question about the target image must be clear, succinct, and concise. Also, the question should only be asked about common objects in the description and candidates, which cannot be answered only from the description and the dialogue.
    Please explain how did you utilize the information sources for generating a question.
    """

    # Few-shot example: User part from Table 18
    train_example_user = """
    [Retrieval Candidates]
    0. A man in a yellow shirt.
    1. A boy in a skateboard park.
    2. The biker is performing a trick.
    3. A man in a green hat doing half-pipe with a skateboard.
    4. A skateboarding man catches the air in the midst of a trick.
    [Description]
    A man is doing a trick on a skateboard.
    [Dialogue]
    Question: What type of trick is the man performing on the skateboard? Answer: a jump.
    Question: What is the location of the jump trick being performed? Answer: a skate park.
    Answer: a skate park Question:
    """

    # Few-shot example: Assistant part from Table 18
    train_example_assistant = """
    Question: What is the outfit of the man performing the jump trick at a skate park?
    Explanation: To generate a question about the description, I will utilize the retrieval candidates that mention the outfit of the man. Candidates 0 and 3 provide information about the man's wearing. The description mentions the man's trick on a skateboard, and the dialogue mentions the type and location of the trick. Since the attribute about the outfit does not appear in the description and the dialogue, the generated question cannot be answered from the information in the description and the dialogue about the target image. Additionally, the generated question is asking for the common objective, man, in the descriptions and candidates, not for the different objective from the description and the retrieval candidates 0 and 3, for example, a shirt and a half-pipe.
    """

    # Construct actual user query
    rc_text = "[Retrieval Candidates]\n" + "\n".join(f"{i+1}. {cap}" for i, cap in enumerate(retrieval_candidates))
    desc_text = "[Description]\n" + description
    dial_text = "[Dialogue]\n" + ("None" if not dialogue else "\n".join(f"Question: {q} Answer: {a}" for q, a in dialogue))
    actual_user_message = f"{rc_text}\n{desc_text}\n{dial_text}\nQuestion:"
    print(f'actual_user_message: {actual_user_message}')

    # Send messages to LLM
    try:
        # response = client.chat.completions.create(model="gpt-3.5-turbo",
        # messages=[
        #     {"role": "system", "content": system_message},
        #     {"role": "user", "content": train_example_user},
        #     {"role": "assistant", "content": train_example_assistant},
        #     {"role": "user", "content": actual_user_message}
        # ],
        # max_tokens=32,
        # temperature=0.7,
        # n=1)
        response = client.chat.completions.create(model="deepseek-chat",
            messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": train_example_user},
            {"role": "assistant", "content": train_example_assistant},
            {"role": "user", "content": actual_user_message}
        ],
        max_tokens=32,
        temperature=0.7,
        n=1)
        # Extract generated question
        full_response = response.choices[0].message.content.strip()
        # Parse to get just the question (assuming format includes "Question:")
        if "Question:" in full_response:
            question = full_response.split("Question:")[1].split("\n")[0].strip()
            return question
        return full_response
    except Exception as e:
        print(f"Error generating question: {e}")
        return None

# Example usage
if __name__ == "__main__":
    retrieval_context = {
        "unlabeled2017/000000000024.jpg": "a brick building",
        "unlabeled2017/000000000097.jpg ": "a group of people walking down a street next to a river",
        "unlabeled2017/000000000207.jpg": "a person skateboarding in a park"
    }
    description = "a man on a skateboard"
    dialogue = []  # First turn, no dialogue
    question = generate_question(retrieval_context, description, dialogue)
    if question:
        print("Generated Question:", question)