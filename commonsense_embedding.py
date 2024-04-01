import subprocess
import numpy as np
from text_to_uri import standardized_uri
import pickle
import re
import pickle
import numpy as np
import torch
import os
import json
import itertools

numberbatch_file = "./numberbatch-19.08.txt"

def process_line(line):
    values = line.rstrip().split(' ')
    word = values[0]
    if re.match(r'/c/zh/.+', word):
        match = re.search(r'/c/zh/(.+)', word)
        content = match.group(1)
        vector = np.array([float(x) for x in values[1:]])
        return content, vector
    else:
        return None

def read_file(file_path):
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            result = process_line(line)
            if result is not None:
                yield result

word_vectors = {}
for content, vector in read_file(numberbatch_file):
    word_vectors[content] = vector
k=word_vectors['快乐']
dict_size = len(word_vectors)
print("字典大小：", dict_size)
f_save = open('dict_file.pkl', 'wb')
pickle.dump(word_vectors, f_save)
f_save.close()











