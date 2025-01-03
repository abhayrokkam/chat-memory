import transformers

# Function to extract user messages from the conversation file
def extract_user_messages(file_path):
    """
    Extracting 'User Messages' from ChatGPT generated conversations.
    Extraction for testing.
    """
    user_messages = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Iterate over lines and extract user messages (lines starting with '**User**:')
    for line in lines:
        if line.startswith("**User**:"):
            message = line.replace("**User**: ", "").strip()
            user_messages.append(message)
    
    return user_messages

# Caching models from HuggingFace
def chache_models():
    """
    Loading and saving the 'safetensor' files for the models from HuggingFace.
    """
    model_ids = ['microsoft/Phi-3.5-mini-instruct',
                 'Qwen/Qwen2.5-7B-Instruct',
                 'teknium/OpenHermes-2.5-Mistral-7B']
    
    for model_id in model_ids:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
        
        tokenizer.save_pretrained(f"./cache/tokenizers/{model_id}")
        model.save_pretrained(f"./cache/models/{model_id}")