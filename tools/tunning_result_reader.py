import os
import re
import sys
import csv
import subprocess
import argparse

def extract_info(path):
    """Extract layer, dim, numvector, loss, epoch from folder name"""
    match = re.search(r'layer(\d+)_dim(\d+)_numvector(\d+)_loss([\d.]+)_ep(\d+)', path)
    if not match:
        raise ValueError("No information found in the path")
    return match.groups()  # (layer, dim, numvector, loss, epoch)

def get_accuracy(cmd):
    """Run command and extract accuracy from output"""
    output = subprocess.check_output(cmd, shell=True).decode()
    match = re.search(r"\* accuracy: ([\d.]+)%", output)
    return float(match.group(1)) if match else None

def main(train_base, results_file="results_coprompt.csv"):
    test_new = train_base.replace("/train_base/", "/test_new/")

    info = extract_info(train_base)
    acc_base = get_accuracy(f"python tools/parse_test_res.py {train_base}")
    acc_test = get_accuracy(f"python tools/parse_test_res.py --test-log {test_new}")

    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["layer", "dim", "numvector", "loss", "epoch", "train_base_acc", "test_new_acc"])
        writer.writerow([*info, acc_base, acc_test])

    print(f"✓ Done. Results written to {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-base",
        type=str,
        required=True,
        help="Path to the train_base directory containing training results."
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results_coprompt.csv",
        help="Name of the results file (default is results_coprompt.csv)."
    )
    args = parser.parse_args()
    main(args.train_base, args.results_file)