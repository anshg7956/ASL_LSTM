import numpy as np
import pandas as pd

df = pd.read_csv('validation_csv/combined_dataset_final_v1.csv')

row= df.iloc[2]


pose_sequence = row["json_data"]
sentence_name = row["sentence_name"]




TOTAL_KEYPOINTS = 137 #hard coded var for the number of keypoints in a given frame

# --- DEBUGGING COUNTERS ---
keypoint_dim_fails = 0
frame_len_fails = 0
sequence_len_fails = 0
# --------------------------

processed_frames = [] #stores processed frames
pose_dim = 3
max_pose_len = 100



for frame in pose_sequence: #go through every frame in the list of frames (pose sequence)
    processed_keypoints = [] #stores one level deeper - processed keypoints

    if isinstance(frame, list): #if the frame is a list
        for keypoint_triplet in frame: #if there is an item in this frame
            # Add a check here to ensure the element is a list before checking its length
            if isinstance(keypoint_triplet, list): #if the item in the frame (keypoint set) is a list
                # Pad or truncate the keypoint to match the pose_dim
                if len(keypoint_triplet) > pose_dim: #if the keypoint dim is longer than (always 3)
                    processed_keypoints.extend(keypoint_triplet[:pose_dim]) #keeps only the first (always 3) keypoint values. Then adds to processed_keypoints list we defined earlier
                    keypoint_dim_fails += 1
                elif len(keypoint_triplet) < pose_dim: #if the keypoint dim is less than (always 3)
                    padded_triplet = keypoint_triplet + [0.0] * (pose_dim - len(keypoint_triplet)) #pads the rest of the keypoint values (till always 3) w/ 0s
                    keypoint_dim_fails += 1
                    processed_keypoints.extend(padded_triplet) #add the padded triplet to the processed_keypoints list
                else:
                    processed_keypoints.extend(keypoint_triplet) #nothing is wrong, so we can just add the keypoint triplet to our processed_keypoints
            else:
                # If the element is not a list, it's a corrupted keypoint.
                # We will treat it as a single invalid keypoint and pad it.
                # This prevents the TypeError.
                processed_keypoints.extend([0.0] * pose_dim) #change everything to zeroes and add to processed_keypoints

    # Pad or truncate the entire frame to a consistent size
    frame_list = processed_keypoints #now our frame_list is just the list of our finalized keypoint triplets - this is for a given frame's keypoints
    print("Length of frame list (keypoints):", len(frame_list))

    if len(frame_list) < TOTAL_KEYPOINTS * pose_dim: #now if we have less than 137 keypoints - remember ts just one frame
        frame_list.extend([0.0] * (TOTAL_KEYPOINTS * pose_dim - len(frame_list))) #pad w/ 0s
        frame_len_fails += 1

    if len(frame_list) > TOTAL_KEYPOINTS * pose_dim: #if we have more than 137 keypoints
        frame_list = frame_list[:TOTAL_KEYPOINTS * pose_dim] #keep only the first 137 keypoints in the frame's list
        frame_len_fails += 1

    processed_frames.append(frame_list) #add the final frame list to the processed_frames list which we defined outside the per-frame loop

# Convert the list of fixed-size arrays to a numpy array

pose_np = np.array(processed_frames, dtype=np.float32) #now convert all of the frames into a numpy array
print("Number of frames:", pose_np.shape[0])

# Truncate pose sequence if it exceeds MAX_POSE_LEN
#if pose_np.shape[0] > max_pose_len: #if the number of final frames are larger than our max length (param value that we pass in)
#    pose_np = pose_np[:max_pose_len] #keep only the first set of these till our max length
#   sequence_len_fails += 1

# Pad the pose sequence to the fixed MAX_POSE_LEN
padded_pose_sequence = np.zeros((max_pose_len, TOTAL_KEYPOINTS * pose_dim), dtype=np.float32)
padded_pose_sequence[:pose_np.shape[0], :] = pose_np


total_fails = keypoint_dim_fails + frame_len_fails + sequence_len_fails
if total_fails > 0:
    print(f"(Sentence: {sentence_name}) Summary -> "
        f"Keypoint dim fails: {keypoint_dim_fails}, "
        f"Frame len fails: {frame_len_fails}, "
        f"Sequence len fails: {sequence_len_fails}")
