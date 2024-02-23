import os
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str, default="data/pg19.jsonl")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

dset = load_dataset("pg19")["train"]
with open(args.output_path, "w") as f:
    for elem in tqdm(dset):
        data = {"text": elem["text"]}
        f.write(f"{json.dumps(data)}\n")