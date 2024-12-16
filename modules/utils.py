# Function to extract user messages from the conversation file
def extract_user_messages(file_path):
    user_messages = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Iterate over lines and extract user messages (lines starting with '**User**:')
    for line in lines:
        if line.startswith("**User**:"):
            message = line.replace("**User**: ", "").strip()
            user_messages.append(message)
    
    return user_messages