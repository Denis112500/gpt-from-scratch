import tiktoken
import numpy as np
from datasets import load_dataset

print("Dataset loading...")
dataset = load_dataset("code-search-net/code_search_net", "python")
print(dataset) 

#print(dataset["train"][0]["whole_func_string"])  # Print the first example from the training set 

enc = tiktoken.get_encoding("gpt2")

def tokenize(sample):
    tokens = enc.encode_ordinary(sample["whole_func_string"])
    tokens.append(enc.eot_token)
    return {"ids": tokens, "len": len(tokens)}
    
tokenized = dataset.map(
    tokenize,
    remove_columns=dataset["train"].column_names,
    num_proc=8,
    desc="Tokenizing"
)

for split, dset in tokenized.items():
    total_len = sum(dset["len"])
    arr = np.empty(total_len, dtype=np.uint16)
    idx = 0
    for sample in dset:
        arr[idx : idx + sample["len"]] = sample["ids"]
        idx += sample["len"]
    arr.tofile(f"data_{split}.bin")
    print(f"Saved {split}: {total_len} tokens")