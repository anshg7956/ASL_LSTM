import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
FILE_PATH = 'validation_csv/combined_dataset_final_v1.csv'
COLUMN_NAME = 'sentence'
SAVE_FIGURE = True
FIGURE_PATH = 'sentence_length_distribution.png'

# --- LOAD DATA ---
df = pd.read_csv(FILE_PATH)

# --- ANALYZE SENTENCE LENGTH (CHARACTERS) ---
sentence_lengths = df[COLUMN_NAME].astype(str).apply(len).to_numpy()

# Five-number summary + mean
summary = {
    "min": int(np.min(sentence_lengths)),
    "q1": float(np.percentile(sentence_lengths, 25)),
    "median": float(np.median(sentence_lengths)),
    "q3": float(np.percentile(sentence_lengths, 75)),
    "max": int(np.max(sentence_lengths)),
    "mean": float(np.mean(sentence_lengths)),
}

# --- OUTPUT STATISTICS ---
print("=== Sentence length (characters) summary ===")
for k, v in summary.items():
    print(f"{k}: {v}")

# --- PLOT DISTRIBUTION ---
sns.set(style="whitegrid")
plt.figure(figsize=(12,6))
sns.histplot(sentence_lengths, bins=30, kde=True, color='skyblue')
plt.title("Distribution of Sentence Lengths (Characters)")
plt.xlabel("Sentence Length (characters)")
plt.ylabel("Count of Sentences")
plt.axvline(summary['median'], color='red', linestyle='--', label='Median')
plt.axvline(summary['q1'], color='green', linestyle='--', label='Q1')
plt.axvline(summary['q3'], color='orange', linestyle='--', label='Q3')
plt.legend()

if SAVE_FIGURE:
    plt.savefig(FIGURE_PATH)
    print(f"Figure saved to {FIGURE_PATH}")
#else:
#    plt.show()


# --- FUNCTION FOR CUSTOM PERCENTILES ---
def analyze_sequence_percentiles(length_array, percentiles=[5, 10, 25, 50, 75, 90, 95]):
    """
    Calculate summary statistics and custom percentiles for sentence lengths.
    """
    length_array = np.array(length_array)
    
    # Five-number summary + mean
    summary = {
        "min": int(np.min(length_array)),
        "q1": float(np.percentile(length_array, 25)),
        "median": float(np.median(length_array)),
        "q3": float(np.percentile(length_array, 75)),
        "max": int(np.max(length_array)),
        "mean": float(np.mean(length_array)),
    }
    
    # Custom percentiles
    percentile_values = {}
    for p in percentiles:
        percentile_values[f"{p}th"] = float(np.percentile(length_array, p))
    
    return {
        "summary": summary,
        "percentiles": percentile_values
    }


# --- RUN ANALYSIS ---
results = analyze_sequence_percentiles(sentence_lengths, percentiles=[5, 10, 25, 50, 75, 90, 95])

print("\nFive-number summary + mean:")
for k, v in results["summary"].items():
    print(f"{k}: {v}")

print("\nCustom percentiles:")
for k, v in results["percentiles"].items():
    print(f"{k}: {v}")
