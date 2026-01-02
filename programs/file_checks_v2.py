import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
FILE_PATH = 'validation_csv/combined_dataset_final_v1.csv'
COLUMN_NAME = 'json_data'
SAVE_FIGURE = True        # Set True if you want to save instead of showing
FIGURE_PATH = 'frames_distribution.png'

# --- LOAD DATA ---
df = pd.read_csv(FILE_PATH)

# --- PARSE JSON DATA ---
def parse_json_column(data):    
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return []
    return data

df[COLUMN_NAME] = df[COLUMN_NAME].apply(parse_json_column)

# --- ANALYZE FRAMES PER SEQUENCE ---
frames_per_sequence = df[COLUMN_NAME].apply(len).to_numpy()

# Five-number summary + mean
summary = {
    "min": int(np.min(frames_per_sequence)),
    "q1": float(np.percentile(frames_per_sequence, 25)),
    "median": float(np.median(frames_per_sequence)),
    "q3": float(np.percentile(frames_per_sequence, 75)),
    "max": int(np.max(frames_per_sequence)),
    "mean": float(np.mean(frames_per_sequence)),
}

# --- OUTPUT STATISTICS ---
print("=== Frames per sequence summary ===")
for k, v in summary.items():
    print(f"{k}: {v}")

# --- PLOT DISTRIBUTION ---
sns.set(style="whitegrid")
plt.figure(figsize=(12,6))
sns.histplot(frames_per_sequence, bins=30, kde=True, color='skyblue')
plt.title("Distribution of Frames per Sequence")
plt.xlabel("Number of Frames")
plt.ylabel("Count of Sequences")
plt.axvline(summary['median'], color='red', linestyle='--', label='Median')
plt.axvline(summary['q1'], color='green', linestyle='--', label='Q1')
plt.axvline(summary['q3'], color='orange', linestyle='--', label='Q3')
plt.legend()

if SAVE_FIGURE:
    plt.savefig(FIGURE_PATH)
    print(f"Figure saved to {FIGURE_PATH}")
#else:
    #plt.show()



def analyze_sequence_percentiles(frames_array, percentiles=[5, 10, 25, 50, 75, 90, 95]):
    """
    Calculate summary statistics and custom percentiles for a sequence length array.

    Args:
        frames_array (np.ndarray or list): Array of frames per sequence.
        percentiles (list): List of percentiles to compute (0-100).

    Returns:
        dict: Contains five-number summary, mean, and custom percentile values.
    """
    frames_array = np.array(frames_array)
    
    # Five-number summary + mean
    summary = {
        "min": int(np.min(frames_array)),
        "q1": float(np.percentile(frames_array, 25)),
        "median": float(np.median(frames_array)),
        "q3": float(np.percentile(frames_array, 75)),
        "max": int(np.max(frames_array)),
        "mean": float(np.mean(frames_array)),
    }
    
    # Custom percentiles
    percentile_values = {}
    for p in percentiles:
        percentile_values[f"{p}th"] = float(np.percentile(frames_array, p))
    
    return {
        "summary": summary,
        "percentiles": percentile_values
    }



# Suppose frames_per_sequence is a numpy array of frame counts
results = analyze_sequence_percentiles(frames_per_sequence, percentiles=[5, 10, 25, 50, 75, 90, 95])

print("Five-number summary + mean:")
for k, v in results["summary"].items():
    print(f"{k}: {v}")

print("\nCustom percentiles:")
for k, v in results["percentiles"].items():
    print(f"{k}: {v}")

