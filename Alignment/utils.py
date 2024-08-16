import pandas as pd
import numpy as np
import json
import time
import os
import argparse
import pickle
import numpy as np

class Logger():
    def __init__(self):
        self.log_file = None
        self.log_dir = None

    def log(self, str):
        if self.log_file is not None:
            with open(self.log_file, 'a') as f:
                f.write(str)

    def set_logdir(self, log_dir):
        self.log_dir = log_dir

    def set_filename(self, filename):
        if self.log_dir is not None:
            self.log_file = self.log_dir + filename

    def close(self):
        if self.log_file is not None:
            self.log_file.close()

def transform(user, item, item2knowledge, score, batch_size):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]])] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64),
        torch.tensor(item, dtype=torch.int64),
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

def load_from_json(filepath):

        with open(filepath, 'r') as f:
            similar_students = json.load(f)
        return {int(student_id): similar_list for student_id, similar_list in similar_students.items()}

def sort_and_fill_dict(input_dict, max_key, min_key=0):
    example_length = len(next(iter(input_dict.values())))
    all_keys = list(range(min_key, max_key + 1))

    for key in all_keys:
        if key not in input_dict:
            input_dict[key] = [0] * example_length

    sorted_dict = {key: input_dict[key] for key in sorted(all_keys)}

    return sorted_dict