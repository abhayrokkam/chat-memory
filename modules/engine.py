import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from .prompts import judgement_gen_prompt, memory_gen_prompt, memory_agent_prompt

import re
from pydantic import BaseModel, Field
from typing import List, Dict

class JudgementGenerator():
    """
    A class that determines if a user's message is relevant for memory.

    The `JudgementGenerator` evaluates user messages to decide if they contain important personal information 
    (e.g., family matters, health issues) that should be remembered by a personal companion. 
    It uses a pre-trained language model to generate a "TRUE" or "FALSE" judgement based on predefined instructions.

    Attributes:
        model_id (str): Identifier for the pre-trained model.
        pipe (transformers.pipelines.text_generation.TextGenerationPipeline): Pipeline for generating judgements.

    Methods:
        get_pipeline(): Initializes and returns the text generation pipeline.
        get_messages(user_content: str) -> List[Dict]: Prepares system and user messages for evaluation.
        get_judgement(user_content: str) -> str: Evaluates user content and returns whether it's relevant for memory.

    Example:
        > judgement_gen = JudgementGenerator()
        > judgement = judgement_gen.get_judgement("I moved to New York last week.")
        > print(judgement)  # 'TRUE'
        > judgement = judgement_gen.get_judgement("Just had coffee.")
        > print(judgement)  # 'FALSE'
    """
    def __init__(self) -> None:
        self.model_id = "microsoft/Phi-3.5-mini-instruct"
        self.pipe = self._init_pipeline()
    
    def _init_pipeline(self) -> transformers.pipelines.text_generation.TextGenerationPipeline:
        """
        Initializes and returns a text generation pipeline.

        Loads a pre-trained causal language model and tokenizer, then creates a 
        text generation pipeline that runs on the GPU for efficient computation. 
        The pipeline can be used to generate text from input prompts.

        Returns:
            transformers.pipelines.text_generation.TextGenerationPipeline: A text generation pipeline.
        """
        model = AutoModelForCausalLM.from_pretrained( 
            f"./cache/models/{self.model_id}",
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )

        model_tokenizer = AutoTokenizer.from_pretrained(f"./cache/tokenizers/{self.model_id}")

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=model_tokenizer,
        )

        return pipe
    
    def get_messages(self, user_content: str) -> List[Dict]:
        """
        Prepares messages for model evaluation of relevance.

        Creates a list of system instructions and user content for the model to assess if the user's message is relevant for memory.

        Args:
            user_content (str): The user's message to evaluate.

        Returns:
            List[Dict]: A list of messages, with system instructions first and user input second.

        Example:
            > get_messages("I have an appointment with my doctor tomorrow.")
            [
                {"role": "system", "content": "Your duty is to assist the personal companion..."},
                {"role": "user", "content": "I have an appointment with my doctor tomorrow."}
            ]
        """
        
        system_content = judgement_gen_prompt
        
        messages = [ 
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        
        return messages
    
    def get_judgement(self, user_content: str) -> str:
        """
        Generates a judgement on the relevance of the user's message for memory.

        Evaluates the user's content through a model pipeline to determine if it is relevant for memory. 
        The judgement is filtered to include only alphabetic characters to remove spaces.

        Args:
            user_content (str): The user's message to evaluate.

        Returns:
            str: The judgement ("TRUE" or "FALSE") indicating whether the message should be remembered.

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
    A class for generating and extracting memory-related information from user input.

    Uses a pre-trained language model and text generation pipeline to identify relevant details 
    (e.g., personal events, relationships, health) for the companion to remember.

    Attributes:
        model_id (str): The pre-trained model identifier.
        pipe (transformers.pipelines.text_generation.TextGenerationPipeline): Pipeline for generating memory details.

    Methods:
        get_pipeline(): Returns the text generation pipeline.
        get_messages(user_content: str) -> List[Dict]: Prepares system and user messages for memory extraction.
        get_memory(user_content: str) -> str: Generates memory-related information to be stored.

    """
    def __init__(self) -> None:
        self.model_id = "Qwen/Qwen2.5-7B-Instruct"
        self.pipe = self._init_pipeline()
    
    def _init_pipeline(self) -> transformers.pipelines.text_generation.TextGenerationPipeline:
        """
        Initializes and returns a text generation pipeline.

        Loads a pre-trained model and tokenizer, creating a pipeline for text generation on the GPU.

        Returns:
            transformers.pipelines.text_generation.TextGenerationPipeline: A text generation pipeline.
        """
        model = AutoModelForCausalLM.from_pretrained( 
            f"./cache/models/{self.model_id}",
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )

        model_tokenizer = AutoTokenizer.from_pretrained(f"./cache/tokenizers/{self.model_id}")

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=model_tokenizer,
        )

        return pipe
    
    def get_messages(self, user_content: str) -> List[Dict]:
        """
        Prepares messages for extracting relevant memory information.

        Creates a list with a system instruction and user input for the model to evaluate and 
        extract relevant details (e.g., personal events, relationships, milestones) for memory.

        Args:
            user_content (str): The user's input to evaluate for relevant memory details.

        Returns:
            List[Dict]: A list with a system message and the user's content for evaluation.

        Example:
            > get_messages("I love my wife Emily and we just came from Coldplay concert.")
            [
                {"role": "system", "content": "Your duty is to assist the personal companion..."},
                {"role": "user", "content": "I love my wife Emily and we just came from Coldplay concert."}
            ]
        """
        
        system_content = memory_gen_prompt
        
        messages = [ 
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        
        return messages
    
    def get_memory(self, user_content: str) -> str:
        """
        Generates memory-related information from user input.

        Analyzes the user's input to extract relevant memory details 
        (e.g., personal facts, health issues, life events) for the companion to remember.

        Args:
            user_content (str): The user's input to extract memory information from.

        Returns:
            str: The generated memory output with relevant details for the companion.

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

class MemoryAgent():
    """
    A class that manages memory and makes decisions on memory handling using an AI model.

    The `MemoryAgent` class is responsible for storing, saving, and replacing memories in a list. 
    It uses a pre-trained language model to make decisions about which function to apply to incoming memory. 
    The agent determines the appropriate action by evaluating the context of the memories and the new memory input.

    Attributes:
        memories (List[str]): A list of stored memories.
        model_id (str): Identifier for the pre-trained language model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding and decoding text.
        model (transformers.PreTrainedModel): Pre-trained model for causal language modeling.
        generation_config (transformers.GenerationConfig): Configuration settings for text generation.
    
    Methods:
        get_agent_decision(incoming_memory): Generates a decision on which function to apply to the incoming memory.
        save_memory(memory): Saves a new memory to the list of memories.
        replace_memory(target_memory, replacement_memory): Replaces an existing memory with a new one.
    """
    def __init__(self, memories: List[str]):
        self.memories = memories
        
        self.model_id = "teknium/OpenHermes-2.5-Mistral-7B"
        self._init_generation_components()
    
    def _init_generation_components(self):
        """
        Initializes the tokenizer, model, and generation settings for text generation.

        Loads the pre-trained tokenizer and model, then configures the model's generation parameters,
        including sampling behavior and token limits.

        Attributes:
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding and decoding text.
            model (transformers.PreTrainedModel): Pre-trained model for causal language modeling.
            generation_config (transformers.GenerationConfig): Generation configuration settings.
        """
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(f"./cache/tokenizers/{self.model_id}")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(f"./cache/models/{self.model_id}",
                                                                        device_map="cuda",
                                                                        torch_dtype="auto").eval()
        
        self.generation_config = self.model.generation_config
        self.generation_config.update(**{
            "use_cache": True,
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 512,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        })
    
    def get_agent_decision(self, incoming_memory):
        """
        Generates an agent's decision on which function to use for the incoming memory.

        This method constructs a prompt based on the current memories and incoming memory, 
        then uses the model to generate a decision. The decision is extracted from the model's output.

        Args:
            incoming_memory (str): The new memory input that needs to be processed.

        Returns:
            str: The agent's decision, indicating which function to use for the incoming memory.
        """
        prompt = memory_agent_prompt.format(function_structure_save_memory = convert_pydantic_to_openai_function(self.SaveMemory),
                                            function_structure_replace_memory = convert_pydantic_to_openai_function(self.ReplaceMemory),
                                            memories = self.memories,
                                            incoming_memory = incoming_memory)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generated_tokens = self.model.generate(**inputs, generation_config=self.generation_config)
        output_text = self.tokenizer.decode(generated_tokens.squeeze(), skip_special_tokens=False)
        
        pattern = r"<\|im_start\|> Solution(.*?)<\|im_end\|>"
        match = re.search(pattern, output_text, re.DOTALL)
        agent_decision = match.group(1).strip()
        
        return agent_decision
    
    # Save Memory
    class SaveMemory(BaseModel):
        """
        Save a memory to the list of memories.
        """
        memory: str = Field(..., description="The memory to be saved.")
    
    def save_memory(self, memory: str) -> None:
        self.memories.append(memory)
    
    # Replace Memory
    class ReplaceMemory(BaseModel):
        """
        Replace a memory in the memory list with a new one.
        """
        target_memory: str = Field(..., description="The old memory to be replaced.")
        replacement_memory: str = Field(..., description="The new memory that will replace the target memory.")
    
    def replace_memory(self, target_memory: str, replacement_memory: str) -> None:
        for i, memory in enumerate(self.memories):
            if memory == target_memory:
                self.memories[i] = replacement_memory
                
                print(f"[INFO] Memory has been replaced")
                print(f"[INFO] Old Memory: {target_memory}  |  New Memory: {replacement_memory}")