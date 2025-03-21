"""
Memory Capability Test Suite for LM2 Models

This script tests and compares the memory capabilities of 
base Llama-3 models vs Llama-3 models trained with memory extension.

Tests are designed to specifically evaluate long-context memory retention,
information retrieval over long contexts, and reasoning capabilities.

Usage:
    python test_memory.py --base_model_path /path/to/base_checkpoint.pth --memory_model_path /path/to/memory_checkpoint.pth

Where:
    - base_model_path: Path to a checkpoint of Llama-3 without memory extension 
      (trained with use_memory=False)
    - memory_model_path: Path to a checkpoint of Llama-3 with memory extension
      (trained with use_memory=True)

The script uses the same model loading mechanisms as the training system to ensure consistency.
It evaluates models on several memory-focused benchmarks with increasing context lengths
to quantify the improvement from memory extension.
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from src.model_memory_llama import CustomLlamaConfig, LlamaMem
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Constants
DEFAULT_CONTEXT_WINDOW = 4096  # Default Llama-3 context window size
MEMORY_BENCHMARKS = [
    "fact_retrieval",        # Tests ability to retrieve facts from context
    "entity_tracking",       # Tests ability to track entities through context
    "reasoning_chains",      # Tests ability to follow complex reasoning chains
    "contradictions",        # Tests ability to notice contradictions in distant parts
    "temporal_ordering",     # Tests ability to maintain temporal ordering
]

class MemoryEvaluator:
    """Evaluates memory capabilities of language models."""
    
    def __init__(
        self,
        base_model_path: str,
        memory_model_path: str,
        model_name: str = "meta-llama/Llama-3-8B",
        device: str = "cuda",
        batch_size: int = 1,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name
        self.base_model_path = base_model_path
        self.memory_model_path = memory_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base and memory models
        self.base_model = self._load_base_model()
        self.memory_model = self._load_memory_model()
        
        # Results storage
        self.results = {
            "base_model": defaultdict(list),
            "memory_model": defaultdict(list)
        }
        
    def _load_base_model(self) -> torch.nn.Module:
        """Load the base Llama-3 model without memory extensions."""
        print(f"Loading base model from {self.base_model_path}")
        
        # Use the same config creation pattern as in train.py
        base_config = AutoConfig.from_pretrained(self.model_name).to_dict()
        config = CustomLlamaConfig(
            use_memory=False,  # Disable memory for base model
            memory_slots=0,
            num_mem_heads=0,
            batch_size=self.batch_size,
            **base_config,
        )
        
        # Use conditional loading just like in train.py
        if os.path.exists(self.base_model_path):
            print(f"Loading from checkpoint: {self.base_model_path}")
            model = LlamaMem.from_ckpt(
                self.base_model_path,
                config=config,
                tokenizer=self.tokenizer,
                rank=self.device,
                load_memory=False,
                resume_training=False,
            )
        else:
            print(f"No checkpoint found at {self.base_model_path}, loading from model name: {self.model_name}")
            model = LlamaMem.from_config(
                config=config,
                tokenizer=self.tokenizer,
            )
            model = model.to(self.device)
        
        model.eval()
        return model
    
    def _load_memory_model(self) -> torch.nn.Module:
        """Load the Llama-3 model with memory extensions."""
        print(f"Loading memory-extended model from {self.memory_model_path}")
        
        # Use the same config creation pattern as in train.py
        base_config = AutoConfig.from_pretrained(self.model_name).to_dict()
        config = CustomLlamaConfig(
            use_memory=True,  # Enable memory
            memory_slots=16,  # Default memory slots from configs
            num_mem_heads=4,  # Default number of memory heads from configs
            batch_size=self.batch_size,
            **base_config,
        )
        
        if os.path.exists(self.memory_model_path):
            print(f"Loading from checkpoint: {self.memory_model_path}")
            model = LlamaMem.from_ckpt(
                self.memory_model_path,
                config=config,
                tokenizer=self.tokenizer,
                rank=self.device,
                load_memory=True,  # Load the trained memory state
                resume_training=False,
            )
        else:
            print(f"No checkpoint found at {self.memory_model_path}, loading from model name: {self.model_name}")
            model = LlamaMem.from_config(
                config=config,
                tokenizer=self.tokenizer,
            )
            model = model.to(self.device)
        
        model.eval()
        return model
    
    def generate_completion(
        self, 
        model: torch.nn.Module, 
        prompt: str, 
        max_tokens: int = 100, 
        temperature: float = 0.7,
    ) -> str:
        """Generate completion from a prompt using the given model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with memory-aware attention if it's the memory model
        with torch.no_grad():
            # Ensure model is in eval mode
            model.eval()
            
            # Use the model's generate method with appropriate settings
            output_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                # Include memory parameters if they exist in the generation config
                use_cache=True,
            )
        
        # Decode only the newly generated tokens
        generated_part = output_ids[0][inputs.input_ids.shape[1]:]
        output_text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
        
        return output_text.strip()
    
    def run_fact_retrieval_test(
        self, 
        context_lengths: List[int] = [1024, 2048, 4096, 8192],
        num_trials: int = 10,
    ) -> Dict:
        """
        Test the model's ability to retrieve facts from different positions in context.
        
        This test embeds facts at different positions in the context and then
        asks the model to retrieve them. The model's accuracy is measured
        as the distance from the context length increases.
        """
        print("\nRunning fact retrieval test...")
        
        # Generate a set of facts for testing
        facts = [
            ("The capital of France is Paris.", "What is the capital of France?", "Paris"),
            ("The Eiffel Tower is 330 meters tall.", "How tall is the Eiffel Tower?", "330 meters"),
            ("The Great Wall of China is over 21,000 kilometers long.", "How long is the Great Wall of China?", "21,000 kilometers"),
            ("The human body has 206 bones.", "How many bones are in the human body?", "206"),
            ("Mount Everest is 8,849 meters tall.", "How tall is Mount Everest?", "8,849 meters"),
            ("The speed of light is approximately 299,792,458 meters per second.", "What is the speed of light?", "299,792,458 meters per second"),
            ("Water boils at 100 degrees Celsius at sea level.", "At what temperature does water boil at sea level?", "100 degrees Celsius"),
            ("The first president of the United States was George Washington.", "Who was the first president of the United States?", "George Washington"),
            ("The chemical formula for water is H2O.", "What is the chemical formula for water?", "H2O"),
            ("The distance from Earth to the Moon is about 384,400 kilometers.", "What is the distance from Earth to the Moon?", "384,400 kilometers"),
        ]
        
        filler_text = """This is filler text to extend the context. It contains no relevant information to answer the questions. This paragraph is designed to create distance between the facts and the questions to test the model's memory capabilities. The more distance between facts and questions, the more it tests the model's ability to recall information from earlier in the context. This tests the effectiveness of both the base attention mechanism and any memory extensions that have been implemented. By varying the amount of filler text, we can see at what point the model's ability to retrieve facts degrades, which gives us insight into its effective context window.""" * 20
        
        results = {
            "base_model": defaultdict(list),
            "memory_model": defaultdict(list)
        }
        
        for context_length in context_lengths:
            print(f"Testing with context length: {context_length}")
            
            for _ in range(num_trials):
                # Select a random fact
                fact, question, answer = facts[np.random.randint(0, len(facts))]
                
                # Calculate how much filler is needed
                tokens_per_char = 0.25  # approximate conversion factor
                fact_tokens = len(fact) * tokens_per_char
                question_tokens = len(question) * tokens_per_char
                available_tokens = context_length - fact_tokens - question_tokens - 50  # margin
                
                # Create filler of appropriate length
                filler_tokens_needed = available_tokens
                filler_chars_needed = int(filler_tokens_needed / tokens_per_char)
                filler = filler_text[:filler_chars_needed]
                
                # Construct the prompt
                prompt = f"{fact}\n\n{filler}\n\n{question}"
                
                # Test base model
                base_response = self.generate_completion(self.base_model, prompt)
                base_correct = answer.lower() in base_response.lower()
                results["base_model"][context_length].append(base_correct)
                
                # Test memory model
                memory_response = self.generate_completion(self.memory_model, prompt)
                memory_correct = answer.lower() in memory_response.lower()
                results["memory_model"][context_length].append(memory_correct)
        
        # Calculate accuracy for each context length
        for model_type in ["base_model", "memory_model"]:
            for context_length in context_lengths:
                accuracy = sum(results[model_type][context_length]) / len(results[model_type][context_length])
                self.results[model_type]["fact_retrieval"].append((context_length, accuracy))
                print(f"{model_type} accuracy at {context_length} tokens: {accuracy:.2f}")
        
        return results
    
    def run_entity_tracking_test(
        self, 
        context_lengths: List[int] = [1024, 2048, 4096, 8192],
        num_trials: int = 10,
    ) -> Dict:
        """
        Test the model's ability to track multiple entities across a long context.
        
        This test introduces several characters and tracks their attributes and
        actions across the context. The model is then asked questions about
        specific entities and their properties/actions from earlier in the text.
        """
        print("\nRunning entity tracking test...")
        
        characters = [
            {"name": "Alice", "attributes": ["tall", "blonde", "doctor", "from London"]},
            {"name": "Bob", "attributes": ["short", "dark-haired", "teacher", "from Paris"]},
            {"name": "Charlie", "attributes": ["athletic", "red-haired", "engineer", "from Berlin"]},
            {"name": "Diana", "attributes": ["slender", "brunette", "scientist", "from Tokyo"]},
            {"name": "Ethan", "attributes": ["muscular", "bald", "chef", "from Rome"]},
        ]
        
        actions = [
            "went to the store",
            "read a book",
            "called a friend",
            "bought a new car",
            "visited a museum",
            "attended a concert",
            "took a long walk",
            "watched a movie",
            "planted some flowers",
            "wrote a letter",
        ]
        
        filler_text = """This is filler text to extend the context. It contains no relevant information about the characters in the story. This paragraph is designed to create distance between the character descriptions and the questions to test the model's memory capabilities. The more distance between descriptions and questions, the more it tests the model's ability to track entities across a long context window. This tests the effectiveness of both the base attention mechanism and any memory extensions that have been implemented. By varying the amount of filler text, we can see at what point the model's ability to track entities degrades.""" * 20
        
        results = {
            "base_model": defaultdict(list),
            "memory_model": defaultdict(list)
        }
        
        for context_length in context_lengths:
            print(f"Testing with context length: {context_length}")
            
            for _ in range(num_trials):
                # Create character descriptions
                story = []
                for char in characters:
                    char_desc = f"{char['name']} is {', '.join(char['attributes'])}."
                    story.append(char_desc)
                
                # Add actions for each character
                for char in characters:
                    char_action = f"{char['name']} {actions[np.random.randint(0, len(actions))]}."
                    story.append(char_action)
                
                story_text = " ".join(story)
                
                # Calculate how much filler is needed
                tokens_per_char = 0.25  # approximate conversion factor
                story_tokens = len(story_text) * tokens_per_char
                available_tokens = context_length - story_tokens - 100  # margin
                
                # Create filler of appropriate length
                filler_tokens_needed = available_tokens
                filler_chars_needed = int(filler_tokens_needed / tokens_per_char)
                filler = filler_text[:filler_chars_needed]
                
                # Select a random character to ask about
                char_idx = np.random.randint(0, len(characters))
                char = characters[char_idx]
                
                # Create a question about either attribute or action
                if np.random.random() < 0.5:
                    # Attribute question
                    attr_idx = np.random.randint(0, len(char["attributes"]))
                    attribute = char["attributes"][attr_idx]
                    
                    if "from" in attribute:
                        question = f"Where is {char['name']} from?"
                        answer = attribute.replace("from ", "")
                    else:
                        question = f"What is {char['name']}'s physical description?"
                        answer = attribute
                else:
                    # Look for action
                    for s in story:
                        if s.startswith(char["name"]) and " is " not in s:
                            action = s[len(char["name"]) + 1:]
                            break
                    
                    question = f"What did {char['name']} do?"
                    answer = action.strip(".")
                
                # Construct the prompt
                prompt = f"{story_text}\n\n{filler}\n\n{question}"
                
                # Test base model
                base_response = self.generate_completion(self.base_model, prompt)
                base_correct = answer.lower() in base_response.lower()
                results["base_model"][context_length].append(base_correct)
                
                # Test memory model
                memory_response = self.generate_completion(self.memory_model, prompt)
                memory_correct = answer.lower() in memory_response.lower()
                results["memory_model"][context_length].append(memory_correct)
        
        # Calculate accuracy for each context length
        for model_type in ["base_model", "memory_model"]:
            for context_length in context_lengths:
                accuracy = sum(results[model_type][context_length]) / len(results[model_type][context_length])
                self.results[model_type]["entity_tracking"].append((context_length, accuracy))
                print(f"{model_type} accuracy at {context_length} tokens: {accuracy:.2f}")
        
        return results
    
    def run_reasoning_chains_test(
        self, 
        context_lengths: List[int] = [1024, 2048, 4096, 8192],
        num_trials: int = 10,
    ) -> Dict:
        """
        Test the model's ability to follow chains of reasoning across long contexts.
        
        This test presents a series of logical statements that build upon each other,
        with conclusions that depend on remembering earlier premises.
        """
        print("\nRunning reasoning chains test...")
        
        # Template for reasoning chains
        reasoning_chains = [
            # Chain 1: Transitive property
            [
                "All A are B.",
                "All B are C.",
                "Therefore, all A are C.",
                "All C are D.",
                "If all A are C, and all C are D, what can we conclude about A and D?",
                "All A are D"
            ],
            # Chain 2: Categorical syllogism
            [
                "No X are Y.",
                "All Z are X.",
                "Therefore, no Z are Y.",
                "Some W are Z.",
                "If no Z are Y, and some W are Z, what can we conclude about some W and Y?",
                "Some W are not Y"
            ],
            # Chain 3: Conditional reasoning
            [
                "If P, then Q.",
                "If Q, then R.",
                "Therefore, if P, then R.",
                "If R, then S.",
                "If P, then what?",
                "S"
            ],
            # Chain 4: Disjunctive syllogism
            [
                "Either J or K is true.",
                "J is false.",
                "Therefore, K is true.",
                "If K is true, then L is true.",
                "If J is false and either J or K is true, what can we conclude about L?",
                "L is true"
            ],
        ]
        
        filler_text = """This text serves as a separator between logical statements. It does not contain any relevant information for the reasoning task. This paragraph is designed to test the model's ability to maintain logical connections across distant parts of the context. The ability to follow chains of reasoning is a crucial aspect of long-context understanding and requires maintaining precise relationships between concepts introduced earlier. This tests both the base attention mechanism and any memory extensions, especially when the distance between premises and conclusions increases.""" * 20
        
        results = {
            "base_model": defaultdict(list),
            "memory_model": defaultdict(list)
        }
        
        for context_length in context_lengths:
            print(f"Testing with context length: {context_length}")
            
            for _ in range(num_trials):
                # Select a random reasoning chain
                chain_idx = np.random.randint(0, len(reasoning_chains))
                chain = reasoning_chains[chain_idx]
                
                premises = chain[:-2]  # All statements except the question and answer
                question = chain[-2]
                answer = chain[-1]
                
                # Calculate how much filler is needed
                tokens_per_char = 0.25  # approximate conversion factor
                premises_text = " ".join(premises)
                premises_tokens = len(premises_text) * tokens_per_char
                question_tokens = len(question) * tokens_per_char
                available_tokens = context_length - premises_tokens - question_tokens - 50  # margin
                
                # Create filler of appropriate length
                filler_tokens_needed = available_tokens
                filler_chars_needed = int(filler_tokens_needed / tokens_per_char)
                filler = filler_text[:filler_chars_needed]
                
                # Construct the prompt
                prompt = f"{premises_text}\n\n{filler}\n\n{question}"
                
                # Test base model
                base_response = self.generate_completion(self.base_model, prompt)
                base_correct = answer.lower() in base_response.lower()
                results["base_model"][context_length].append(base_correct)
                
                # Test memory model
                memory_response = self.generate_completion(self.memory_model, prompt)
                memory_correct = answer.lower() in memory_response.lower()
                results["memory_model"][context_length].append(memory_correct)
        
        # Calculate accuracy for each context length
        for model_type in ["base_model", "memory_model"]:
            for context_length in context_lengths:
                accuracy = sum(results[model_type][context_length]) / len(results[model_type][context_length])
                self.results[model_type]["reasoning_chains"].append((context_length, accuracy))
                print(f"{model_type} accuracy at {context_length} tokens: {accuracy:.2f}")
        
        return results
    
    def run_contradictions_test(
        self, 
        context_lengths: List[int] = [1024, 2048, 4096, 8192],
        num_trials: int = 10,
    ) -> Dict:
        """
        Test the model's ability to detect contradictions across long contexts.
        
        This test introduces a statement early in the context and then a contradictory
        statement later. The model is tested on its ability to recognize the contradiction.
        """
        print("\nRunning contradictions test...")
        
        # Pairs of contradictory statements
        contradictions = [
            (
                "The meeting is scheduled for Tuesday at 3 PM.",
                "The meeting is scheduled for Thursday at 3 PM.",
                "Is there a contradiction about when the meeting is scheduled?"
            ),
            (
                "The company reported a profit of $10 million this quarter.",
                "The company reported a loss of $5 million this quarter.",
                "Is there a contradiction about the company's financial results?"
            ),
            (
                "The building has 20 floors total.",
                "The building has 25 floors total.",
                "Is there a contradiction about the number of floors in the building?"
            ),
            (
                "The movie was released in 2018.",
                "The movie was released in 2020.",
                "Is there a contradiction about when the movie was released?"
            ),
            (
                "The speed limit on this highway is 65 mph.",
                "The speed limit on this highway is 55 mph.",
                "Is there a contradiction about the speed limit?"
            ),
            (
                "The document was signed by five people.",
                "The document was signed by three people.",
                "Is there a contradiction about how many people signed the document?"
            ),
            (
                "The product weighs exactly 2.5 kilograms.",
                "The product weighs exactly 3.2 kilograms.",
                "Is there a contradiction about the weight of the product?"
            ),
            (
                "The airplane has a seating capacity of 180 passengers.",
                "The airplane has a seating capacity of 220 passengers.",
                "Is there a contradiction about the seating capacity of the airplane?"
            ),
            (
                "The minimum age requirement is 18 years.",
                "The minimum age requirement is 21 years.",
                "Is there a contradiction about the minimum age requirement?"
            ),
            (
                "The test results were negative.",
                "The test results were positive.",
                "Is there a contradiction about the test results?"
            ),
        ]
        
        filler_text = """This paragraph contains filler text to create distance between statements. It is designed to test the model's ability to remember information across a long context and detect inconsistencies. The ability to recognize contradictions requires maintaining precise representations of earlier statements and comparing them with later ones. This capability is essential for tasks requiring critical thinking and fact-checking. As the distance between contradictory statements increases, it becomes more challenging for models to detect the inconsistencies, making this a good test of effective context utilization.""" * 20
        
        results = {
            "base_model": defaultdict(list),
            "memory_model": defaultdict(list)
        }
        
        for context_length in context_lengths:
            print(f"Testing with context length: {context_length}")
            
            for _ in range(num_trials):
                # Select a random contradiction pair
                idx = np.random.randint(0, len(contradictions))
                statement1, statement2, question = contradictions[idx]
                
                # Calculate how much filler is needed
                tokens_per_char = 0.25  # approximate conversion factor
                statements_tokens = (len(statement1) + len(statement2)) * tokens_per_char
                question_tokens = len(question) * tokens_per_char
                available_tokens = context_length - statements_tokens - question_tokens - 50  # margin
                
                # Create filler of appropriate length
                filler_tokens_needed = available_tokens
                filler_chars_needed = int(filler_tokens_needed / tokens_per_char)
                filler = filler_text[:filler_chars_needed]
                
                # Construct the prompt
                prompt = f"{statement1}\n\n{filler}\n\n{statement2}\n\n{question}"
                answer = "Yes"  # All pairs have contradictions
                
                # Test base model
                base_response = self.generate_completion(self.base_model, prompt)
                base_correct = "yes" in base_response.lower()
                results["base_model"][context_length].append(base_correct)
                
                # Test memory model
                memory_response = self.generate_completion(self.memory_model, prompt)
                memory_correct = "yes" in memory_response.lower()
                results["memory_model"][context_length].append(memory_correct)
        
        # Calculate accuracy for each context length
        for model_type in ["base_model", "memory_model"]:
            for context_length in context_lengths:
                accuracy = sum(results[model_type][context_length]) / len(results[model_type][context_length])
                self.results[model_type]["contradictions"].append((context_length, accuracy))
                print(f"{model_type} accuracy at {context_length} tokens: {accuracy:.2f}")
        
        return results
    
    def run_temporal_ordering_test(
        self, 
        context_lengths: List[int] = [1024, 2048, 4096, 8192],
        num_trials: int = 10,
    ) -> Dict:
        """
        Test the model's ability to maintain temporal ordering across long contexts.
        
        This test presents a sequence of events in chronological order and then
        asks questions about their relative timing.
        """
        print("\nRunning temporal ordering test...")
        
        # Event sequences with timestamps
        event_sequences = [
            [
                "At 9:00 AM, Alice arrived at the office.",
                "At 10:30 AM, Bob called a meeting.",
                "At 12:15 PM, everyone went to lunch.",
                "At 2:45 PM, the client presentation began.",
                "At 4:30 PM, the team started wrapping up for the day.",
                "At 5:15 PM, Alice left the office.",
                "Which happened first: Alice leaving the office or the client presentation?",
                "client presentation"
            ],
            [
                "On Monday, the project planning began.",
                "On Wednesday, the initial designs were completed.",
                "On Friday, the team presented the designs to stakeholders.",
                "On the following Tuesday, revisions were requested.",
                "On the following Thursday, the revised designs were submitted.",
                "On the following Friday, the project was approved.",
                "Which happened first: project approval or the initial designs?",
                "initial designs"
            ],
            [
                "In January, the company announced its yearly goals.",
                "In March, the first quarter review showed positive results.",
                "In June, the team reached 50% of the yearly targets.",
                "In August, a new product was launched.",
                "In October, the team celebrated exceeding the yearly targets.",
                "In December, the annual report was published.",
                "Which happened first: the product launch or reaching 50% of targets?",
                "reaching 50% of targets"
            ],
            [
                "At 8:15 AM, the flight departed from New York.",
                "At 9:45 AM, the flight experienced turbulence.",
                "At 11:20 AM, a meal was served to passengers.",
                "At 1:30 PM, the pilot announced they were beginning descent.",
                "At 2:10 PM, the flight landed in Chicago.",
                "At 2:35 PM, passengers began disembarking.",
                "Which happened first: the meal service or the turbulence?",
                "turbulence"
            ],
        ]
        
        filler_text = """This text serves as a buffer between the event sequence and the questions about temporal ordering. It does not contain any relevant information about the timing of events. This paragraph tests the model's ability to maintain the chronological sequence of events across a long context. Remembering when events occurred relative to each other is crucial for narrative understanding and causal reasoning. As the context grows longer, maintaining this temporal information becomes more challenging, making this a good test of the model's memory capabilities.""" * 20
        
        results = {
            "base_model": defaultdict(list),
            "memory_model": defaultdict(list)
        }
        
        for context_length in context_lengths:
            print(f"Testing with context length: {context_length}")
            
            for _ in range(num_trials):
                # Select a random event sequence
                seq_idx = np.random.randint(0, len(event_sequences))
                sequence = event_sequences[seq_idx]
                
                events = sequence[:-2]  # All events except the question and answer
                question = sequence[-2]
                answer = sequence[-1]
                
                # Calculate how much filler is needed
                tokens_per_char = 0.25  # approximate conversion factor
                events_text = " ".join(events)
                events_tokens = len(events_text) * tokens_per_char
                question_tokens = len(question) * tokens_per_char
                available_tokens = context_length - events_tokens - question_tokens - 50  # margin
                
                # Create filler of appropriate length
                filler_tokens_needed = available_tokens
                filler_chars_needed = int(filler_tokens_needed / tokens_per_char)
                filler = filler_text[:filler_chars_needed]
                
                # Construct the prompt
                prompt = f"{events_text}\n\n{filler}\n\n{question}"
                
                # Test base model
                base_response = self.generate_completion(self.base_model, prompt)
                base_correct = answer.lower() in base_response.lower()
                results["base_model"][context_length].append(base_correct)
                
                # Test memory model
                memory_response = self.generate_completion(self.memory_model, prompt)
                memory_correct = answer.lower() in memory_response.lower()
                results["memory_model"][context_length].append(memory_correct)
        
        # Calculate accuracy for each context length
        for model_type in ["base_model", "memory_model"]:
            for context_length in context_lengths:
                accuracy = sum(results[model_type][context_length]) / len(results[model_type][context_length])
                self.results[model_type]["temporal_ordering"].append((context_length, accuracy))
                print(f"{model_type} accuracy at {context_length} tokens: {accuracy:.2f}")
        
        return results
    
    def plot_results(self, output_dir: str = "results"):
        """Plot the results of all memory tests."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot each test
        for test_name in MEMORY_BENCHMARKS:
            plt.figure(figsize=(10, 6))
            
            # Extract data
            base_data = self.results["base_model"].get(test_name, [])
            memory_data = self.results["memory_model"].get(test_name, [])
            
            if not base_data or not memory_data:
                print(f"Skipping plot for {test_name}: no data available")
                continue
            
            # Sort by context length
            base_data.sort(key=lambda x: x[0])
            memory_data.sort(key=lambda x: x[0])
            
            # Extract x and y values
            base_x = [item[0] for item in base_data]
            base_y = [item[1] for item in base_data]
            memory_x = [item[0] for item in memory_data]
            memory_y = [item[1] for item in memory_data]
            
            # Plot
            plt.plot(base_x, base_y, 'o-', label="Base Model", color="blue")
            plt.plot(memory_x, memory_y, 'o-', label="Memory Model", color="red")
            
            plt.xlabel("Context Length (tokens)")
            plt.ylabel("Accuracy")
            plt.title(f"{test_name.replace('_', ' ').title()} Test")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save plot
            plt.savefig(os.path.join(output_dir, f"{test_name}_test.png"))
            plt.close()
        
        # Create a combined plot
        plt.figure(figsize=(12, 8))
        
        # For each test type
        for i, test_name in enumerate(MEMORY_BENCHMARKS):
            base_data = self.results["base_model"].get(test_name, [])
            memory_data = self.results["memory_model"].get(test_name, [])
            
            if not base_data or not memory_data:
                continue
            
            # Sort by context length
            base_data.sort(key=lambda x: x[0])
            memory_data.sort(key=lambda x: x[0])
            
            # Extract x and y values
            base_x = [item[0] for item in base_data]
            base_y = [item[1] for item in base_data]
            memory_x = [item[0] for item in memory_data]
            memory_y = [item[1] for item in memory_data]
            
            # Calculate improvement
            improvement = [(m - b) for b, m in zip(base_y, memory_y)]
            
            # Plot on new subplot
            plt.subplot(2, 3, i+1)
            plt.plot(base_x, base_y, 'o-', label="Base", color="blue", alpha=0.7)
            plt.plot(memory_x, memory_y, 'o-', label="Memory", color="red", alpha=0.7)
            
            plt.xlabel("Context Length")
            plt.ylabel("Accuracy")
            plt.title(test_name.replace('_', ' ').title())
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "all_tests_comparison.png"))
        plt.close()
        
        # Save results to JSON
        with open(os.path.join(output_dir, "memory_test_results.json"), 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def run_all_tests(
        self, 
        context_lengths: List[int] = [1024, 2048, 4096, 8192],
        num_trials: int = 5,
        output_dir: str = "results"
    ):
        """Run all memory tests and save results."""
        self.run_fact_retrieval_test(context_lengths, num_trials)
        self.run_entity_tracking_test(context_lengths, num_trials)
        self.run_reasoning_chains_test(context_lengths, num_trials)
        self.run_contradictions_test(context_lengths, num_trials)
        self.run_temporal_ordering_test(context_lengths, num_trials)
        
        self.plot_results(output_dir)
        
        # Print summary
        print("\n=== MEMORY EVALUATION SUMMARY ===")
        for test_name in MEMORY_BENCHMARKS:
            print(f"\n{test_name.replace('_', ' ').title()}:")
            for model_type in ["base_model", "memory_model"]:
                for context_length, accuracy in self.results[model_type][test_name]:
                    print(f"  {model_type} @ {context_length} tokens: {accuracy:.4f}")
            
            # Calculate average improvement
            base_acc = dict(self.results["base_model"][test_name])
            mem_acc = dict(self.results["memory_model"][test_name])
            
            common_lengths = set(base_acc.keys()).intersection(set(mem_acc.keys()))
            if common_lengths:
                avg_improvement = sum(mem_acc[cl] - base_acc[cl] for cl in common_lengths) / len(common_lengths)
                print(f"  Average improvement: {avg_improvement:.4f} ({avg_improvement*100:.1f}%)")
                
                # Find length with max improvement
                max_improve_len = max(common_lengths, key=lambda cl: mem_acc[cl] - base_acc[cl])
                max_improve = mem_acc[max_improve_len] - base_acc[max_improve_len]
                print(f"  Max improvement: {max_improve:.4f} ({max_improve*100:.1f}%) at {max_improve_len} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test memory capabilities of language models")
    parser.add_argument("--base_model_path", type=str, required=True, 
                        help="Path to the base model checkpoint (trained without memory extension)")
    parser.add_argument("--memory_model_path", type=str, required=True, 
                        help="Path to the memory-augmented model checkpoint (trained with memory extension)")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3-8B", 
                        help="Hugging Face model name/identifier for the base Llama-3 model")
    parser.add_argument("--context_lengths", type=str, default="1024,2048,4096,8192", 
                        help="Comma-separated list of context lengths to test (in tokens)")
    parser.add_argument("--num_trials", type=int, default=5, 
                        help="Number of trials per test configuration")
    parser.add_argument("--output_dir", type=str, default="memory_test_results", 
                        help="Directory to save test results and plots")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run tests on (cuda or cpu)")
    
    args = parser.parse_args()
    
    print("\n=== LM2 Memory Capability Testing ===")
    print(f"Base model checkpoint: {args.base_model_path}")
    print(f"Memory model checkpoint: {args.memory_model_path}")
    print(f"Model name: {args.model_name}")
    print(f"Device: {args.device}")
    
    # Convert context lengths string to list of integers
    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    print(f"Testing context lengths: {context_lengths}")
    print(f"Running {args.num_trials} trials per configuration")
    print(f"Saving results to: {args.output_dir}")
    print("=====================================\n")
    
    # Run tests
    evaluator = MemoryEvaluator(
        base_model_path=args.base_model_path,
        memory_model_path=args.memory_model_path,
        model_name=args.model_name,
        device=args.device,
    )
    
    evaluator.run_all_tests(
        context_lengths=context_lengths,
        num_trials=args.num_trials,
        output_dir=args.output_dir
    ) 