import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from prompts import judgement_gen_prompt, memory_gen_prompt, memory_agent_prompt

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
    A class that manages and evaluates memories for a personal agent.

    The `MemoryAgent` class stores a list of memories and uses a pre-trained model 
    to generate decisions based on incoming memory inputs. It can save and replace memories, 
    and it uses a text generation pipeline to analyze and respond to memory-related queries.

    Attributes:
        memories (List[str]): A list of stored memories.
        model_id (str): The model identifier for the pre-trained text generation model.
        generation_config (dict): Configuration settings for the text generation model.
        pipe (transformers.pipelines.TextGenerationPipeline): The text generation pipeline for decision making.

    Methods:
        _init_pipeline(): Initializes and returns a text generation pipeline.
        get_agent_decision(incoming_memory: str): Generates a decision based on incoming memory.
        save_memory(memory: str): Adds a new memory to the list.
        replace_memory(target_memory: str, replacement_memory: str): Replaces an existing memory with a new one.
    """

    def __init__(self, memories: List[str]):
        self.memories = memories
        
        self.model_id = "teknium/OpenHermes-2.5-Mistral-7B"
        self.generation_config = {}
        
        self.pipe = self._init_pipeline()
    
    def _init_pipeline(self):
        """
        Initializes and returns a text generation pipeline.

        Loads the tokenizer and model from the specified `model_id`, configures the model 
        for text generation with specific settings, and sets up the generation pipeline 
        to run on the GPU. The pipeline is returned for use in generating text.

        Returns:
            transformers.pipelines.TextGenerationPipeline: A configured text generation pipeline.
        """
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id,
                                                                  device_map="cuda",
                                                                  torch_dtype="auto").eval()
        
        self.generation_config = model.generation_config
        self.generation_config.update({
            "use_cache": True,
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 512,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        })
        
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        return pipe
    
    def get_agent_decision(self, incoming_memory):
        """
        Generates an agent decision based on incoming memory.

        Constructs a prompt using the incoming memory and current function structures, 
        then generates a decision using the model pipeline. The decision is returned 
        as the agent's response.

        Args:
            incoming_memory (str): The incoming memory input to be evaluated by the agent.

        Returns:
            str: The agent's decision based on the incoming memory.
        """
        prompt = memory_agent_prompt.format(function_structure_save_memory = convert_pydantic_to_openai_function(self.save_memory),
                                            function_structure_replace_memory = convert_pydantic_to_openai_function(self.replace_memory),
                                            memories = self.memories,
                                            incoming_memory = incoming_memory)
        
        output = self.pipe(prompt, **self.generation_config)
        agent_decision = output[0]['generated_text']
        
        return agent_decision
    
    # Save Memory
    class save_memory(BaseModel):
        """
        Save a memory to the list of memories.
        """
        memory: str = Field(..., description="The memory to be saved.")
    
    def save_memory(self, memory: str) -> None:
        self.memories.append(memory)
    
    # Replace Memory
    class replace_memory(BaseModel):
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