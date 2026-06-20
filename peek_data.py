import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")

data = np.fromfile("data_train.bin", dtype=np.uint16)
tokens = data[:500].tolist()
print(enc.decode(tokens))
