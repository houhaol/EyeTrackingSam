import os
import json
import csv
import argparse
from collections import defaultdict

def load_predictions(predictions_csv):
    predictions = []
    with open(predictions_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append((row["timestamp_ns"], row["label"]))
    return predictions

def load_mapping(mapping_file):
    with open(mapping_file, "r") as f:
        return json.load(f)

def compute_distribution(predictions, prompt_to_category):
    category_counter = defaultdict(int)
    category_samples = defaultdict(list)

    for ts, label in predictions:
        category = prompt_to_category.get(label, "Uncategorized")
        category_counter[category] += 1
        category_samples[category].append((ts, label))

    return category_counter, category_samples

def save_distribution_csv(category_counter, output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Count"])
        for cat, count in category_counter.items():
            writer.writerow([cat, count])

def main(args):
    predictions = load_predictions(args.predictions)
    prompt_to_category = load_mapping(args.mapping)
    category_counter, _ = compute_distribution(predictions, prompt_to_category)

    print("\n🎯 Prediction Distribution by Category:")
    for cat, count in sorted(category_counter.items(), key=lambda x: -x[1]):
        print(f"{cat}: {count}")

    save_distribution_csv(category_counter, args.output)
    print(f"\n📄 Category distribution saved to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize CLIP prompt predictions into high-level categories.")
    parser.add_argument("--predictions", required=True, help="CSV file containing timestamp and predicted label")
    parser.add_argument("--mapping", required=True, help="JSON file mapping prompt labels to high-level categories")
    parser.add_argument("--output", default="clip_category_distribution.csv", help="Output CSV file for category counts")
    args = parser.parse_args()

    main(args)
