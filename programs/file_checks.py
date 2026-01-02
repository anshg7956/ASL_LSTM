import json
from collections import Counter
import pandas as pd
import ast
import numpy as np

def analyze_keypoints(df, column="json_data"):
    total_frames = 0
    keypoints_per_frame_counter = Counter()
    keypoint_length_counter = Counter()
    frames_per_sequence = []

    for i, row in df.iterrows():
        data = row[column]
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                continue

        # Record how many frames in this sequence
        frames_per_sequence.append(len(data))
        total_frames += len(data)

        for frame in data:
            frame_keypoints = 0

            for section in frame:
                # Each section is a flat list, reshape into groups of 3
                n_keypoints = len(section) // 3
                frame_keypoints += n_keypoints

                # Count keypoint vector lengths (usually 3)
                for j in range(n_keypoints):
                    keypoint_length_counter[3] += 1  # all chunks of 3

            keypoints_per_frame_counter[frame_keypoints] += 1

    # Compute five-number summary + mean for frames per sequence
    frames_array = np.array(frames_per_sequence)
    five_number_summary = {
        "min": int(np.min(frames_array)),
        "q1": float(np.percentile(frames_array, 25)),
        "median": float(np.median(frames_array)),
        "q3": float(np.percentile(frames_array, 75)),
        "max": int(np.max(frames_array)),
        "mean": float(np.mean(frames_array)),
    }

    return {
        "total_frames": total_frames,
        "keypoints_per_frame_freq": dict(keypoints_per_frame_counter),
        "keypoint_length_freq": dict(keypoint_length_counter),
        "frames_per_sequence_summary": five_number_summary,
    }

def analyze_keypoints_ast(df, column="json_data"):
    total_frames = 0
    keypoints_per_frame_counter = Counter()
    keypoint_length_counter = Counter()

    for i, row in df.iterrows():
        data = row[column]
        if isinstance(data, str):
            try:
                data = ast.literal_eval(data)
            except Exception:
                continue  # skip rows that fail to parse

        total_frames += len(data)

        for frame in data:
            frame_keypoints = 0

            for section in frame:
                # Each section is a flat list, reshape into groups of 3
                n_keypoints = len(section) // 3
                frame_keypoints += n_keypoints

                # Count keypoint vector lengths (usually 3)
                for j in range(n_keypoints):
                    keypoint_length_counter[3] += 1  # all chunks of 3

            keypoints_per_frame_counter[frame_keypoints] += 1

    return {
        "total_frames": total_frames,
        "keypoints_per_frame_freq": dict(keypoints_per_frame_counter),
        "keypoint_length_freq": dict(keypoint_length_counter),
    }


df = pd.read_csv("validation_csv/combined_dataset_final_v1.csv", sep=",")
print(df.head())
print(df.shape)


results = analyze_keypoints(df, column="json_data")

print("Total frames:", results["total_frames"])
print("Keypoints per frame freq:", results["keypoints_per_frame_freq"])
print("Keypoint length freq:", results["keypoint_length_freq"])
print("Frames per sequence summary:", results["frames_per_sequence_summary"])


#stats = analyze_keypoints_ast(df, "json_data")
#print("Total frames (AST):", stats["total_frames"])
#print("Keypoints per frame frequency (AST):", stats["keypoints_per_frame_freq"])
#print("Keypoint length frequency (AST):", stats["keypoint_length_freq"])


