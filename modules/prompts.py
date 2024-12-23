judgement_gen_prompt = """
            Your duty is to assist the personal companion. You have to decide if the personal companion will benefit by remebering the facts from the given text.
            The personal companion is expected to remember everything important to an individual. The personal companion is a best friend who takes care of the User.
            The personal companion is a companion for old individuals.
            The personal companion is expected to remember family and related matters, milestones in life, health issues, medication schedules.

            If you think the text contains relevant information, you have to respond with 'TRUE'.
            If there is no relevant information in the text, return 'FALSE'.
            Do NOT say anything else. You have to respond ONLY with either 'TRUE' or 'FALSE'.
            
            You are the Assistant in the given examples.
            
            Examples:
            
            User: Hey, my name is Adam.
            Assistant: TRUE
            
            User: I love my brother John.
            Assistant: TRUE
            
            User: The sky is blue today.
            Assistant: FALSE
            
            The above given conversations are examples. The user input is given below. Using the examples as a template, extract the memory from the below given message.
        """
        
memory_gen_prompt = """
            Your duty is to assist the personal companion. You have to extract the facts from the given text which will be the memory of the companion.
            The personal companion is expected to remember everything important to the user. The personal companion is a best friend who takes care of the user.
            The personal companion is a companion for old individuals.
            The personal companion is expected to remember facts about personal life, family, friends and health.
            
            You have to extract the memory from the given text. You have to ONLY extract information from the given text. Do NOT respond to the text.
            You are only supposed to extract the memories that are deemed important for long term. Do NOT extract the memory that is useful in short term.
            Each message can contain multiple memories.
            Make sure that each memory is compiled in one sentence. Each memory must have one of these tags: personal, family, friend, health
            
            You are the Assistant in the below given examples.
            
            Examples:
            
            User: Hey, my name is Adam.
            Assistant: (personal) User's name is Adam.
            
            User: I love my brother John.
            Assistant: (family) User has a brother named John
            
            User: Me and my wife Emily just came from Coldplay concert. I love them.
            Assistant: (family) User has a wife named Emily. (personal) User likes the band Coldplay.
            
            User: I met Linda today. We had a drink. She is such a good friend. I remember when I met her in the Beatles concert.
            Assitant: (friend) User has a friend named Linda. (friend) User met Linda in a Beatles concert.
            
            The above given conversations are examples. The user input is given below. Using the examples as a template, extract the memory from the below given message.
        """

memory_agent_prompt = """
            <|im_start|>system
            You are a helpful assistant.
            You have to give reasoning using the incoming memory.
            The solution should include the given format after the reasoning: {"name": "function_name", "arguments": {"arg_1": "value_1", "arg_2": value_2, ...}}.
            The solution can be NONE.
            <|im_end|>

            <|im_start|>user
            You have access to the following functions:

            {function_structure_save_memory}
            {function_structure_replace_memory}

            Edge cases you must handle:
            - If there are no functions that match the user request, you will respond with NONE.<|im_end|>

            <|im_start|>User Memories
            {memories}
            <|im_end|>

            <|im_start|>Incoming Memory
            {incoming_memory}
            <|im_end|>

            <|im_start|>question
            User Memories show the saved memories in the system.
            If the Incoming Memory or a similar memory already exists, your solution should be NONE.
            If the Incoming Memory or a similar memory does not exist in the User Memories, save the memory.
            If the Incoming Memory or a similar memory contradicts a memory in the User Memories, replace the old memory with a new memory.

            Which function should I use for the Incoming Memory?
            <|im_end|>

            <|im_start|>Solution
        """