import csv
import json
from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt

def weighted_sample_by_word_count(data_list, sample_size=20000, random_seed=42):
    """
    Sample items using weighted sampling to reduce dominance of short responses.
    
    Args:
        data_list: List of tuples [(word_count, string, string, string)]
        sample_size: Number of items to sample (default: 30000)
        random_seed: Random seed for reproducibility
    
    Returns:
        List of sampled tuples
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Extract word counts
    word_counts = [item[0] for item in data_list]
    
    # Create weights: exponentially increase weight with word count
    # This heavily penalizes 1-2 word responses
    weights = []
    for wc in word_counts:
        if wc <= 2:
            weight = 0.5  # Very low weight for 1-2 words
        elif wc <= 4:
            weight = 1  # Normal weight for 3-4 words
        elif wc <= 6:
            weight = 1.5
        elif wc <= 8:
            weight = 6.0  # Higher weight for 5-8 words
        else:
            weight = 256  # Highest weight for 9+ words
        weights.append(weight)
    
    # Convert to numpy arrays for efficient sampling
    weights = np.array(weights)
    probs = weights / weights.sum()  # Normalize to probabilities

    # Sample indices based on weights
    sampled_indices = np.random.choice(
        len(data_list), 
        size=sample_size, 
        replace=False,  # Allow replacement since we need 30k samples
        p=probs
    )
    
    # Return sampled items
    sampled_data = [data_list[i] for i in sampled_indices]
    
    return sampled_data

def produce_full_squad():
    input_csv = '../data/squad_v1.csv'
    output_jsonl = '../data/squad_v1.jsonl'
    with open(input_csv, mode='r', encoding='utf-8') as csv_file, open(output_jsonl, mode='w', encoding='utf-8') as jsonl_file:

        reader = csv.DictReader(csv_file)
        
        for row in reader:
            json_line = {
                "postText": [row["question"]],
                "targetParagraphs": [row["context"]],
                "spoiler": [row["answer"]]
            }
            jsonl_file.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"Conversion complete. Output saved to {output_jsonl}")

def produce_small_squad():
    input_csv = '../data/squad_v1.csv'
    output_jsonl = '../data/squad_v1_small.jsonl'
    seen_contexts = set()
    with open(input_csv, mode='r', encoding='utf-8') as csv_file, open(output_jsonl, mode='w', encoding='utf-8') as jsonl_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            context = row["context"]
            if context not in seen_contexts:
                seen_contexts.add(context)
                json_line = {
                    "postText": [row["question"]],
                    "targetParagraphs": [context],
                    "spoiler": [row["answer"]]
                }
                jsonl_file.write(json.dumps(json_line, ensure_ascii=False) + "\n")

def produce_spoiler_weighted_squad(k: int = 3, bias_exp: int = 1):
    input_csv = '../data/squad_v1.csv'
    train_output_jsonl = '../data/squad_v1_weighted_train.jsonl'
    val_output_jsonl = '../data/squad_v1_weighted_val.jsonl'
    context_groups = defaultdict(list)
    with open(input_csv, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            context_groups[row["context"]].append(row)

    selected_entries = []

    for context, rows in context_groups.items():
        # Number of rows for this context
        n = len(rows)
        sample_size = min(k, n)

        # Calculate weights proportional to spoiler length (answer length)
        weights = [len(row["answer"])**bias_exp for row in rows]

        # Use weighted sampling **without replacement**:
        # random.choices samples WITH replacement,
        # so we implement weighted sampling without replacement manually.

        # Prepare list of indices to pick from
        indices = list(range(n))
        chosen_indices = []

        # Copy weights so we can modify them during sampling
        current_weights = weights.copy()

        for _ in range(sample_size):
            total_weight = sum(current_weights)
            if total_weight == 0:
                # All weights zero, pick random index from remaining
                choices = [i for i in indices if i not in chosen_indices]
                chosen_idx = random.choice(choices)
            else:
                # Compute cumulative weights
                cum_weights = []
                cum_sum = 0
                for w in current_weights:
                    cum_sum += w
                    cum_weights.append(cum_sum)
                r = random.uniform(0, total_weight)
                # Find index where r would fit
                chosen_idx = None
                for i, cw in enumerate(cum_weights):
                    if r <= cw:
                        chosen_idx = i
                        break
            
            # Add chosen index and set its weight to zero to avoid re-picking
            chosen_indices.append(chosen_idx)
            current_weights[chosen_idx] = 0

        # Add selected rows
        for idx in chosen_indices:
            row = rows[idx]
            entry = {
                "postText": [row["question"]],
                "targetParagraphs": [row["context"]],
                "spoiler": [row["answer"]]
            }
            selected_entries.append(entry)

    # Shuffle entries randomly before splitting
    random.shuffle(selected_entries)

    # Split 80-20 train-validation
    split_index = int(0.8 * len(selected_entries))
    train_entries = selected_entries[:split_index]
    val_entries = selected_entries[split_index:]

    # Write train set
    with open(train_output_jsonl, mode='w', encoding='utf-8') as train_file:
        for entry in train_entries:
            train_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Write validation set
    with open(val_output_jsonl, mode='w', encoding='utf-8') as val_file:
        for entry in val_entries:
            val_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

def squad_analysis(BINS = 20):
    input_csv = '../data/squad_v1.csv'
    lengths = []
    with open(input_csv, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            lengths.append(len(row["answer"].split()))

    # Calculate the 99.99th percentile value
    threshold = 20

    # Filter the data to exclude values above the threshold
    filtered_lengths = [x for x in lengths if x <= threshold]

    plt.figure(figsize=(6, 4))
    plt.xticks(range(0, 20, 2))
    m = np.mean(lengths)
    s = np.std(lengths)
    plt.hist(filtered_lengths, bins=BINS, edgecolor='black')
    plt.title(f"Answer word counts (µ: {m:.2f}, std: {s:2f})")
    plt.xlabel("Word count")
    plt.ylabel("Frequency")
    plt.tight_layout()  # Adjust layout to reduce whitespace
    plt.savefig("../graphs/squad_answer_word_counts.png")

def squad_output_analysis(BINS = 20, threshold: int = None):
    input_file = '../data/squad_v1_weighted_train.jsonl'
    lengths = []
    with open(input_file, "r") as jsonl_file:
        for row in jsonl_file:
            row = json.loads(row)
            lengths.append(len(row["spoiler"][0].split()))

    # Calculate the 99.99th percentile value
    threshold = threshold or np.percentile(lengths, 99.9)

    # Filter the data to exclude values above the threshold
    filtered_lengths = [x for x in lengths if x <= threshold]

    plt.figure(figsize=(6, 4))
    plt.xticks(range(0, 20, 2))
    m = np.mean(lengths)
    s = np.std(lengths)
    plt.hist(filtered_lengths, bins=BINS, edgecolor='black')
    plt.title(f"Answer word counts (µ: {m:.2f}, std: {s:2f})")
    plt.xlabel("Word count")
    plt.ylabel("Frequency")
    plt.tight_layout()  # Adjust layout to reduce whitespace
    # plt.show()
    plt.savefig("../graphs/squad_output_analysis.png")

def squad_weighted_v2(k: int = 20000, n: float = 3):
    input_csv = '../data/squad_v1.csv'
    train_output_jsonl = '../data/squad_v1_weighted_train.jsonl'
    val_output_jsonl = '../data/squad_v1_weighted_val.jsonl'
    contexts = []
    with open(input_csv, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for i, row in enumerate(reader):
            ans = row["answer"]
            contexts.append((len(ans.split()), row["question"], row["context"], ans))

    contexts = [(q,t,a) for _,q,t,a in sorted(contexts)]
    
    # Construct weights: index raised to 'power'
    # Use indices starting from 1 to avoid zero weights
    indices = np.arange(1, len(contexts)+1)
    
    # Compute log weights to avoid large numbers: log(weights) = power * log(indices)
    log_weights = n * np.log(indices)

    # Use softmax to get normalized probabilities without overflow
    max_log = np.max(log_weights)
    exp_weights = np.exp(log_weights - max_log)  # subtract max for numerical stability
    probs = exp_weights / np.sum(exp_weights)

    chosen_indices = np.random.choice(len(contexts), size=k, replace=False, p=probs)
    chosen = [contexts[i] for i in chosen_indices]

    # Shuffle entries randomly before splitting
    random.shuffle(chosen)

    selected_entries = []
    for q, t, a in chosen:
        entry = {
            "postText": [q],
            "targetParagraphs":[t],
            "spoiler": [a]
        }
        selected_entries.append(entry)

    # Split 90-10 train-validation
    split_index = int(0.9 * len(selected_entries))
    train_entries = selected_entries[:split_index]
    val_entries = selected_entries[split_index:]

    # Write train set
    with open(train_output_jsonl, mode='w', encoding='utf-8') as train_file:
        for entry in train_entries:
            train_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Write val set
    with open(val_output_jsonl, mode='w', encoding='utf-8') as val_file:
        for entry in val_entries:
            val_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

def squad_weighted_v3(k: int = 30000):

    input_csv = '../data/squad_v1.csv'
    train_output_jsonl = '../data/squad_v1_weighted_train.jsonl'
    val_output_jsonl = '../data/squad_v1_weighted_val.jsonl'
    contexts = []
    with open(input_csv, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for i, row in enumerate(reader):
            ans = row["answer"]
            contexts.append((len(ans.split()), row["question"], row["context"], ans))

    chosen = weighted_sample_by_word_count(contexts, sample_size=k)

    random.shuffle(chosen)

    selected_entries = []
    for _, q, t, a in chosen:
        entry = {
            "postText": [q],
            "targetParagraphs":[t],
            "spoiler": [a]
        }
        selected_entries.append(entry)

    # Split 90-10 train-validation
    split_index = int(0.9 * len(selected_entries))
    train_entries = selected_entries[:split_index]
    val_entries = selected_entries[split_index:]

    # Write train set
    with open(train_output_jsonl, mode='w', encoding='utf-8') as train_file:
        for entry in train_entries:
            train_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Write train set
    with open(val_output_jsonl, mode='w', encoding='utf-8') as val_file:
        for entry in val_entries:
            val_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    squad_weighted_v2(k = 20000, n=3)
    # squad_output_analysis(BINS=15, threshold=20)