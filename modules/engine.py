import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from typing import List, Dict

class JudgementGenerator():
    """
    A class that generates judgements on whether a user's message is relevant for memory.

    The `JudgementGenerator` class is responsible for evaluating user-provided content 
    to determine if the message contains relevant information that should be remembered 
    by a personal companion. The model evaluates the message and generates a "TRUE" or 
    "FALSE" judgement based on whether the message contains important personal information 
    for the user, such as family matters, milestones, health issues, or medication schedules.

    The class loads a pre-trained causal language model and uses it to generate judgements 
    based on system-defined instructions. The model pipeline is initialized when the class 
    is instantiated, and the `get_judgement` method is used to assess individual user messages.

    Attributes:
        model_id (str): The identifier for the pre-trained model to be used for text generation.
        pipe (transformers.pipelines.text_generation.TextGenerationPipeline): The text generation pipeline
            used to generate the judgements based on user input.

    Methods:
        get_pipeline(): Initializes and returns a text generation pipeline for the model.
        get_messages(user_content: str) -> List[Dict]: Prepares and returns the system and user messages 
                                                      for model evaluation.
        get_judgement(user_content: str) -> str: Evaluates the user content and generates a judgement 
                                                 indicating whether the message is relevant for memory.
    
    Example:
        > judgement_gen = JudgementGenerator()
        > judgement = judgement_gen.get_judgement("I moved to New York last week.")
        > print(judgement)
        'TRUE'

        > judgement = judgement_gen.get_judgement("Just had coffee.")
        > print(judgement)
        'FALSE'
    """
    def __init__(self) -> None:
        self.model_id = "microsoft/Phi-3.5-mini-instruct"
        self.pipe = self.get_pipeline()
    
    def get_pipeline(self) -> transformers.pipelines.text_generation.TextGenerationPipeline:
        """
        Initializes and returns a text generation pipeline for the model.

        This function loads a pre-trained causal language model and its corresponding 
        tokenizer using the specified model identifier. It then creates and returns a 
        text generation pipeline, which can be used for generating text based on 
        provided input. The model is loaded to run on the GPU using the `cuda` device 
        map for efficient computation.

        The returned pipeline can be used to generate text from the model by providing 
        input prompts.

        Returns:
            transformers.pipelines.text_generation.TextGenerationPipeline: A text generation 
            pipeline that can be used to generate text based on user inputs.
        """
        model = AutoModelForCausalLM.from_pretrained( 
            self.model_id,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )

        model_tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=model_tokenizer,
        )

        return pipe
    
    def get_messages(user_content: str) -> List[Dict]:
        """
        Generates a list of messages for the model to evaluate relevance.

        This function prepares a list of messages, including system instructions 
        and the user's content. The system content provides instructions to 
        the model on how to assess the user's message for relevance, and the 
        user's content is added to the list for evaluation.

        The model will use these messages to determine whether the user's 
        content contains information that the personal companion should remember, 
        based on the context provided in the system content.

        Args:
            user_content (str): The input message from the user that will be evaluated 
                                for relevance to the personal companion's memory.

        Returns:
            List[Dict]: A list of messages where the first message contains system 
                        instructions, and the second message contains the user's input.

        Example:
            > get_messages("I have an appointment with my doctor tomorrow.")
            [
                {"role": "system", "content": "Your duty is to assist the personal companion..."},
                {"role": "user", "content": "I have an appointment with my doctor tomorrow."}
            ]
        """
        
        system_content = """
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
        """
        
        messages = [ 
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        
        return messages
    
    def get_judgement(self, user_content: str) -> str:
        """
        Generates a judgement on whether the user's message is relevant for memory.

        This function evaluates the user's content by passing it through a model pipeline 
        that generates a judgement. The judgement determines if the content is considered 
        relevant information for a personal companion to remember or not. The judgement is 
        filtered to contain only alphabetic characters before being returned.

        The returned judgement is used to either select the message for memory extraction 
        or ignore it, based on its relevance.

        Args:
            user_content (str): The input message from the user that will be evaluated 
                                for relevance to the companion's memory.

        Returns:
            str: The generated judgement, which indicates whether the message should be 
                remembered or ignored. The result is a string consisting of alphabetic 
                characters only, such as "TRUE" or "FALSE".

        Example:
            > instance.get_judgement("I moved to New York last week.")
            'TRUE'

            > instance.get_judgement("Just had coffee.")
            'FALSE'
        """
        generation_args = {
            "max_new_tokens": 1,
            "return_full_text": False,
            "temperature": 0.1,
            "do_sample": True,
        }
        
        messages = self.get_messages(user_content)
        
        output = self.pipe(messages, **generation_args)
        judgement = output[0]['generated_text']
        judgement = ''.join(filter(str.isalpha, judgement))
        
        return judgement
    
class MemoryGenerator():
    """
    A class responsible for generating and extracting memory-related information 
    based on user input for a personal companion.

    This class utilizes a pre-trained language model and a text generation pipeline 
    to analyze user input and extract relevant memory information. The personal companion 
    will remember important details such as personal events, relationships, health issues, 
    and milestones based on the processed user input. The class handles the task of 
    generating structured memory details that the personal companion should remember.

    Attributes:
        model_id (str): The identifier for the pre-trained model used for text generation.
        pipe (transformers.pipelines.text_generation.TextGenerationPipeline): 
            A text generation pipeline used to generate memory details from the user's input.

    Methods:
        get_pipeline(): Initializes and returns the text generation pipeline for processing user input.
        get_messages(user_content: str) -> List[Dict]: Prepares a list of messages, including system instructions and the user's input, 
                                                       for processing by the model to extract relevant information for memory.
        get_memory(user_content: str) -> str: Processes the user's input and generates memory-related information 
                                              to be stored by the personal companion.
    """
    def __init__(self) -> None:
        self.model_id = "Qwen/Qwen2.5-7B-Instruct"
        self.pipe = self.get_pipeline()
    
    def get_pipeline(self) -> transformers.pipelines.text_generation.TextGenerationPipeline:
        """
        Initializes and returns a text generation pipeline for the model.

        This function loads a pre-trained causal language model and its corresponding 
        tokenizer using the specified model identifier. It then creates and returns a 
        text generation pipeline, which can be used for generating text based on 
        provided input. The model is loaded to run on the GPU using the `cuda` device 
        map for efficient computation.

        The returned pipeline can be used to generate text from the model by providing 
        input prompts.

        Returns:
            transformers.pipelines.text_generation.TextGenerationPipeline: A text generation 
            pipeline that can be used to generate text based on user inputs.
        """
        model = AutoModelForCausalLM.from_pretrained( 
            self.model_id,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )

        model_tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=model_tokenizer,
        )

        return pipe
    
    def get_messages(user_content: str) -> List[Dict]:
        """
        Prepares a list of messages for the model to extract relevant information for memory.

        This function creates a list of messages that includes a system instruction and the user's input. 
        The system message provides instructions to the model on how to extract relevant facts from the 
        user's content, such as important personal details, family matters, health information, or milestones. 
        The user's content is then evaluated by the model to extract information that should be remembered 
        by the personal companion.

        The model is instructed to extract only the relevant information from the user's input and provide it 
        in a structured format.

        Args:
            user_content (str): The user's input message that will be evaluated for relevant information 
                                that should be remembered by the personal companion.

        Returns:
            List[Dict]: A list of two messages, where the first message contains system instructions 
                        and the second message contains the user's content for evaluation.

        Example:
            > get_messages("I love my wife Emily and we just came from Coldplay concert.")
            [
                {"role": "system", "content": "Your duty is to assist the personal companion..."},
                {"role": "user", "content": "I love my wife Emily and we just came from Coldplay concert."}
            ]
        """
        
        system_content = """
            Your duty is to assist the personal companion. You have to extract the facts from the given text that will benefit hte personal companion.
            The personal companion is expected to remember everything important to an individual. The personal companion is a best friend who takes care of the User.
            The personal companion is a companion for old individuals.
            The personal companion is expected to remember family and related matters, milestones in life, health issues, medication schedules.
            
            You have to extract the relevant information from the given text. You have to ONLY extract information from the given text. Do NOT respond to the text.
            
            You are the Assistant in the given examples.
            
            Examples:
            
            User: Hey, my name is Adam.
            Assistant: Name is Adam.
            
            User: I love my brother John.
            Assistant: Brother named John.
            
            User: The sky is blue today.
            Assistant: 
            
            User: Me and my wife Emily just came from Coldplay concert. I love them.
            Assistant: Wife named Emily. Likes Coldplay.
        """
        
        messages = [ 
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        
        return messages
    
    def get_memory(self, user_content: str) -> str:
        """
        Generates memory-related information based on the user's input.

        This function processes the user's input to extract relevant memory details 
        that should be stored by the personal companion. The function utilizes a 
        text generation pipeline to analyze the provided content and generate a 
        structured response that represents memory-related information to be remembered.

        The generated memory includes relevant facts, such as personal details, 
        health information, or significant life events, based on the context of 
        the user's message.

        Args:
            user_content (str): The user's input message from which memory-related 
                                information will be extracted and generated.

        Returns:
            str: The generated memory output, which includes the relevant details 
                for the personal companion to remember. This is returned as a 
                string that contains the extracted information.

        Example:
            > get_memory("Hey, I was diagnosed with Arthritis")
            'User diagnosed with Arthritis'
        """
        generation_args = {
            "max_new_tokens": 50,
            "return_full_text": False,
            "temperature": 0.5,
            "do_sample": True,
        }
        
        messages = self.get_messages(user_content)
        
        output = self.pipe(messages, **generation_args)
        memory = output[0]['generated_text']
        
        return memory