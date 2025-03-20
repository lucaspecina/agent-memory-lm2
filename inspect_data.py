import os
import sys
import numpy as np
import glob
from transformers import AutoTokenizer
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime

def read_llama_shard(filename):
    """Read a binary shard file created for Llama-3 training."""
    print(f"Opening binary file: {filename}")
    with open(filename, 'rb') as f:
        tokens = np.fromfile(f, dtype=np.uint32)
    print(f"  - File size: {os.path.getsize(filename) / 1024:.2f} KB")
    print(f"  - Data format: uint32 tokens ({tokens.nbytes / 1024:.2f} KB in memory)")
    return tokens

def analyze_token_distribution(tokens, tokenizer, max_tokens=50):
    """Analyze the distribution of tokens in the data."""
    token_counter = Counter(tokens)
    print(f"\n=== Token Distribution Analysis ===")
    print(f"  - Unique tokens: {len(token_counter)}")
    print(f"  - Most common tokens:")
    
    for token, count in token_counter.most_common(max_tokens):
        try:
            token_text = tokenizer.decode([token])
            if token_text.strip() == '':
                token_text = f"<special or whitespace token>"
            print(f"    - Token {token}: '{token_text}' ({count} occurrences, {count/len(tokens)*100:.2f}%)")
        except:
            print(f"    - Token {token}: <decoding error> ({count} occurrences, {count/len(tokens)*100:.2f}%)")

def visualize_document_lengths(doc_lengths, output_file=None):
    """Create a histogram of document lengths."""
    plt.figure(figsize=(10, 6))
    plt.hist(doc_lengths, bins=20, alpha=0.7)
    plt.title('Document Length Distribution')
    plt.xlabel('Tokens per Document')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file)
        print(f"Document length histogram saved to {output_file}")
    else:
        print("(Visualization available if you uncomment the plt.show() line)")
        # plt.show()  # Uncomment to display plot

def inspect_shard(filename, num_samples=5, max_tokens_per_sample=100, analyze_tokens=True, visualize=False):
    """Perform a detailed inspection of a shard file."""
    print("\n" + "="*80)
    print(f"DETAILED ANALYSIS OF: {filename}")
    print("="*80)
    
    tokens = read_llama_shard(filename)
    print(f"\n--- Basic Information ---")
    print(f"Total tokens: {len(tokens):,}")
    
    # Load tokenizer
    print("\nLoading tokenizer (meta-llama/Llama-3.2-1B)...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    # Find EOT tokens which separate documents
    eot_token = tokenizer.encode('')[0]  # Get EOT token
    print(f"EOT (End of Text) token: {eot_token}")
    
    eot_indices = np.where(tokens == eot_token)[0]
    
    # If no EOT tokens found, just show beginning of file
    if len(eot_indices) == 0:
        print("\n⚠ No document separators (EOT tokens) found in this file!")
        print("Showing sample from beginning:")
        sample_tokens = tokens[:min(max_tokens_per_sample, len(tokens))]
        print("\n" + "-"*50)
        print(tokenizer.decode(sample_tokens.tolist()))
        print("-"*50)
        return
    
    # Calculate document statistics
    doc_lengths = []
    for i in range(len(eot_indices)):
        if i == 0:
            # First document (may start from beginning of file or from previous EOT)
            doc_start = 0
        else:
            doc_start = eot_indices[i-1] + 1
        
        doc_end = eot_indices[i]
        doc_lengths.append(doc_end - doc_start)
    
    # Show document statistics
    print(f"\n--- Document Statistics ---")
    print(f"Number of documents: {len(eot_indices):,}")
    print(f"Average document length: {np.mean(doc_lengths):.2f} tokens")
    print(f"Shortest document: {min(doc_lengths):,} tokens")
    print(f"Longest document: {max(doc_lengths):,} tokens")
    print(f"Median document length: {np.median(doc_lengths):.2f} tokens")
    
    if visualize:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"doc_lengths_{os.path.basename(filename)}_{timestamp}.png"
        visualize_document_lengths(doc_lengths, output_file)
    
    # Show samples from different documents
    print(f"\n--- Document Samples ({min(num_samples, len(eot_indices))} of {len(eot_indices)}) ---")
    
    for i in range(min(num_samples, len(eot_indices))):
        if i == 0:
            # For first document
            start_idx = 0
        else:
            start_idx = eot_indices[i-1] + 1
            
        end_idx = min(start_idx + max_tokens_per_sample, eot_indices[i])
        
        # Get document length
        doc_len = eot_indices[i] - start_idx + 1
        
        print(f"\n📄 Document {i+1}:")
        print(f"  - Total length: {doc_len:,} tokens")
        print(f"  - Showing tokens {start_idx}-{end_idx} ({end_idx-start_idx} tokens, {(end_idx-start_idx)/doc_len*100:.1f}% of document)")
        print(f"  - Content sample:")
        print("-" * 50)
        
        if start_idx < len(tokens):
            sample_tokens = tokens[start_idx:end_idx]
            try:
                decoded_text = tokenizer.decode(sample_tokens.tolist())
                # Truncate if too long for display
                if len(decoded_text) > 1000:
                    decoded_text = decoded_text[:997] + "..."
                print(decoded_text)
            except Exception as e:
                print(f"Error decoding: {e}")
                print(f"Raw tokens: {sample_tokens}")
        print("-" * 50)
    
    # Token distribution analysis
    if analyze_tokens:
        analyze_token_distribution(tokens, tokenizer)

def analyze_dataset(pattern, max_files=5):
    """Analyze all files matching the pattern."""
    print(f"\n{'='*40} DATASET ANALYSIS {'='*40}")
    print(f"Pattern: {pattern}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*91}")
    
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"❌ No files found matching pattern: {pattern}")
        return
    
    # Basic dataset statistics
    print(f"\n--- Dataset Overview ---")
    print(f"Found {len(files)} files matching pattern")
    total_size_kb = sum(os.path.getsize(f) for f in files) / 1024
    print(f"Total dataset size: {total_size_kb:.2f} KB ({total_size_kb/1024:.2f} MB)")
    
    # Group files by type
    train_files = [f for f in files if "_train_" in f]
    val_files = [f for f in files if "_val_" in f]
    other_files = [f for f in files if "_train_" not in f and "_val_" not in f]
    
    print(f"  - Training files: {len(train_files)}")
    print(f"  - Validation files: {len(val_files)}")
    print(f"  - Other files: {len(other_files)}")
    
    # Detailed file info
    print("\n--- Files in Dataset ---")
    for i, filepath in enumerate(files):
        filesize = os.path.getsize(filepath) / 1024
        filename = os.path.basename(filepath)
        file_type = "train" if "_train_" in filename else "val" if "_val_" in filename else "unknown"
        print(f"  {i+1}. {filename} ({filesize:.2f} KB) - {file_type}")
    
    # Analyze a subset of files in detail
    print(f"\nAnalyzing {min(max_files, len(files))} files in detail...")
    for filename in files[:max_files]:
        inspect_shard(filename, num_samples=3, analyze_tokens=False)
        
    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_data.py <path_to_bin_file_or_pattern> [--full]")
        print("  --full    Perform full analysis (more samples, token distribution, etc.)")
        return
    
    pattern = sys.argv[1]
    full_analysis = "--full" in sys.argv
    
    # If analyzing a single file
    if os.path.isfile(pattern):
        inspect_shard(pattern, 
                      num_samples=10 if full_analysis else 5, 
                      max_tokens_per_sample=200 if full_analysis else 100,
                      analyze_tokens=full_analysis,
                      visualize=full_analysis)
    # If analyzing multiple files
    else:
        analyze_dataset(pattern, max_files=5 if full_analysis else 2)

if __name__ == "__main__":
    main() 