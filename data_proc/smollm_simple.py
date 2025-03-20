"""
Simplified SmoLLM Corpus dataset processor (Windows-compatible)
"""
import os
import argparse
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from data_common import write_datafile

def main():
    parser = argparse.ArgumentParser(description="Dataset preprocessing")
    parser.add_argument("-v", "--version", type=str, default="python", help="Dataset version: cosmo, python, fineweb, smollm")
    parser.add_argument("-m", "--model_desc", type=str, default="llama-3", help="Model descriptor llama-3")
    parser.add_argument("-s", "--shard_size", type=int, default=10**6, help="Size of each data shard in tokens")
    parser.add_argument("-n", "--num_examples", type=int, default=100, help="Number of examples to process")
    parser.add_argument("--streaming", action="store_true", help="Use streaming to avoid downloading full dataset")
    args = parser.parse_args()

    # Dataset mapping
    directories = {
        ("cosmo", "llama-3"): ("cosmo-llama3", "cosmopedia-v2"),
        ("python", "llama-3"): ("python-llama3", "python-edu"),
        ("fineweb", "llama-3"): ("fineweb-ddp-llama3", "fineweb-edu-dedup"),
        ("smollm", "llama-3"): ("smollm-llama3", "cosmopedia-v2"),
    }
    local_dir, remote_name = directories[(args.version, args.model_desc)]
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    print(f"Processing {args.num_examples} examples from {args.version} dataset")
    
    # Load dataset with streaming if requested
    if args.streaming:
        print(f"Using streaming mode to avoid downloading full dataset")
        fw = load_dataset("HuggingFaceTB/smollm-corpus", name=remote_name, split="train", streaming=True)
        # Take only the requested number of examples
        fw = fw.take(args.num_examples)
    else:
        # Download the dataset normally
        fw = load_dataset("HuggingFaceTB/smollm-corpus", name=remote_name, split="train")
        if args.num_examples is not None:
            fw = fw.select(range(min(args.num_examples, len(fw))))
            print(f"Selected {len(fw)} examples for processing")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    # Add special tokens if not already in tokenizer
    special_tokens_dict = {'additional_special_tokens': ['<PROMPT>', '<RESPONSE>']}
    
    def get_text_from_doc(doc):
        """Extract text from document based on available fields"""
        # Inspect document structure for first document
        print(f"Document keys: {list(doc.keys())}")
        
        # Try different possible field names
        if "text" in doc:
            return doc["text"]
        elif "content" in doc:
            return doc["content"]
        elif "code" in doc:
            return doc["code"]
        elif "repo" in doc:  # For python-edu dataset
            # For python-edu, combine file_path and content
            file_path = doc.get("file_path", "")
            content = doc.get("content", "")
            return f"File: {file_path}\n\n{content}"
        else:
            # Fallback: stringify the whole document
            print(f"Warning: Couldn't find text field. Document structure: {doc}")
            return str(doc)
    
    def tokenize_llama(doc):
        # Tokenize a single document
        encode = lambda s: tokenizer.encode(s, add_special_tokens=False, verbose=False, split_special_tokens=True)
        eot = tokenizer.encode('')[0]  # EOT token (128000)
        tokens = [eot]  # Special <|endoftext|> token to delimit documents
        
        # Tokenize the text
        text = get_text_from_doc(doc)
        tokens.extend(encode(text))
        
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**32).all(), "token dictionary too large for uint32"
        return tokens_np.astype(np.uint32)
    
    # Process the dataset
    token_dtype = np.uint32
    name = "smollm-corpus"
    
    # Setup for processing
    shard_index = 0
    all_tokens_np = np.empty((args.shard_size,), dtype=token_dtype)
    token_count = 0
    progress_bar = None
    
    # Process each document
    print("Processing documents...")
    for i, doc in enumerate(tqdm(fw, desc="Documents")):
        # Print the first document to debug
        if i == 0:
            print(f"First document structure: {doc}")
        
        # Tokenize the document
        tokens = tokenize_llama(doc)
        
        # Is there enough space in the current shard?
        if token_count + len(tokens) < args.shard_size:
            # Append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # Update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # Write current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # Split document between shards if needed
            remainder = args.shard_size - token_count
            if progress_bar:
                progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np.tolist(), args.model_desc)
            print(f"Wrote {filename}")
            shard_index += 1
            progress_bar = None
            # Start next shard with leftover tokens
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder
    
    # Write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_tokens_np[:token_count]).tolist(), args.model_desc)
        print(f"Wrote final shard: {filename}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main() 