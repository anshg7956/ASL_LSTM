import pandas as pd
file_path = '/content/drive/MyDrive/ASL_BFH_2D_DATA/combined_dataset_final_v1.csv'
train_v_df = pd.read_csv(file_path)



train_v_df.head() #preview ds to make sure


import torch
import pandas as pd
import numpy as np
import ast

from torch import nn
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader


train_v_df['json_data'] = train_v_df['json_data'].apply(ast.literal_eval)


import torch
from torch.utils.data import Dataset
import numpy as np

class ASLDataset(Dataset):
    def __init__(self, texts, poses, tokenizer, max_text_len, max_pose_len, pose_dim):
        self.texts = texts
        self.poses = poses
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_pose_len = max_pose_len
        self.pose_dim = pose_dim
        self.total_keypoints = 137  # hardcoded for How2Sign

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        pose_sequence = self.poses[idx]

        processed_frames = []
        failed_counter = 0

        for frame in pose_sequence:
            processed_keypoints = []

            if isinstance(frame, list):
                for keypoint_triplet in frame:
                    if isinstance(keypoint_triplet, list):
                        if len(keypoint_triplet) > self.pose_dim:
                            processed_keypoints.extend(keypoint_triplet[:self.pose_dim])
                            failed_counter += 1
                        elif len(keypoint_triplet) < self.pose_dim:
                            padded_triplet = keypoint_triplet + [0.0]*(self.pose_dim - len(keypoint_triplet))
                            processed_keypoints.extend(padded_triplet)
                            failed_counter += 1
                        else:
                            processed_keypoints.extend(keypoint_triplet)
                    else:
                        processed_keypoints.extend([0.0]*self.pose_dim)

            # Pad/truncate frame to fixed size
            if len(processed_keypoints) < self.total_keypoints * self.pose_dim:
                processed_keypoints.extend([0.0]*(self.total_keypoints*self.pose_dim - len(processed_keypoints)))
                failed_counter += 1
            elif len(processed_keypoints) > self.total_keypoints * self.pose_dim:
                processed_keypoints = processed_keypoints[:self.total_keypoints*self.pose_dim]
                failed_counter += 1

            processed_frames.append(processed_keypoints)

        pose_np = np.array(processed_frames, dtype=np.float32)

        # Truncate/pad sequence
        if pose_np.shape[0] > self.max_pose_len:
            pose_np = pose_np[:self.max_pose_len]
            failed_counter += 1

        padded_pose_sequence = np.zeros((self.max_pose_len, self.total_keypoints, self.pose_dim), dtype=np.float32)
        padded_pose_sequence[:pose_np.shape[0], :, :] = pose_np.reshape(-1, self.total_keypoints, self.pose_dim)

        # ---- Center keypoints on mid-shoulder ----
        # Assuming OpenPose indices: right_shoulder=2, left_shoulder=5
        right_shoulder_idx = 2
        left_shoulder_idx = 5
        mid_shoulder = (padded_pose_sequence[:, right_shoulder_idx, :2] + padded_pose_sequence[:, left_shoulder_idx, :2]) / 2.0
        padded_pose_sequence[:, :, :2] -= mid_shoulder[:, np.newaxis, :]

        # Convert to torch tensor
        final_pose_tensor = torch.tensor(padded_pose_sequence, dtype=torch.float32)

        # Tokenize text
        text_enc = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_text_len, return_tensors='pt')

        return {
            'input_ids': text_enc['input_ids'].squeeze(0),
            'attention_mask': text_enc['attention_mask'].squeeze(0),
            'poses': final_pose_tensor
        }



import torch
import torch.nn as nn
import math

# LSTM decoder with relative keypoints, time embeddings, and optional bone-length loss
def get_sinusoidal_embedding(seq_len, dim, device):
    '''
    Generates sinusoidal positional embeddings (like Transformers) for time steps.
    '''
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # (seq_len,1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # shape (seq_len, dim)


class LSTMDecoderRelative(nn.Module):
    def __init__(self, bert_output_dim, hidden_size, num_lstm_layers, max_pose_len, total_keypoints=137, pose_dim=3, use_bone_loss=False, bone_pairs=None):
        super(LSTMDecoderRelative, self).__init__()

        self.total_keypoints = total_keypoints
        self.max_pose_len = max_pose_len
        self.pose_dim = pose_dim
        self.hidden_size = hidden_size
        self.use_bone_loss = use_bone_loss
        self.bone_pairs = bone_pairs  # list of (i,j) joint indices representing bones

        # LSTM network
        self.lstm = nn.LSTM(input_size=bert_output_dim,
                            hidden_size=hidden_size,
                            num_layers=num_lstm_layers,
                            batch_first=True)

        # Fully connected to predict (x,y) for each keypoint
        self.fc = nn.Linear(hidden_size, self.total_keypoints * 2)  # x,y only

        # Time embedding dimension (same as LSTM hidden size)
        self.time_embed_dim = hidden_size

    def forward(self, bert_output_embeddings):
        batch_size = bert_output_embeddings.size(0)
        device = bert_output_embeddings.device

        # ---- 1. Create LSTM input ----
        # Use full BERT token sequence
        seq_len = bert_output_embeddings.size(1)

        # If BERT seq shorter than MAX_POSE_LEN, interpolate / pad embeddings
        if seq_len < self.max_pose_len:
            repeat_factor = math.ceil(self.max_pose_len / seq_len)
            lstm_input = bert_output_embeddings.repeat(1, repeat_factor, 1)[:, :self.max_pose_len, :]
        else:
            lstm_input = bert_output_embeddings[:, :self.max_pose_len, :]

        # Add time embeddings
        time_embeddings = get_sinusoidal_embedding(self.max_pose_len, self.time_embed_dim, device)  # (max_pose_len, hidden_size)
        time_embeddings = time_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch, max_pose_len, hidden_size)

        lstm_input = lstm_input + time_embeddings  # (batch, max_pose_len, bert_output_dim)

        # ---- 2. Pass through LSTM ----
        lstm_output, _ = self.lstm(lstm_input)

        # ---- 3. Fully connected to predict keypoints ----
        pose_output = self.fc(lstm_output)  # (batch, max_pose_len, total_keypoints*2)
        pose_output = pose_output.view(batch_size, self.max_pose_len, self.total_keypoints, 2)  # (batch, frames, keypoints, xy)

        return pose_output

    def compute_bone_loss(self, pred_relative, target_relative):
        '''
        Optional: enforces bone-length consistency.
        pred_relative & target_relative: (batch, frames, keypoints, 2)
        bone_pairs: list of tuples (i,j)
        '''
        if not self.use_bone_loss or self.bone_pairs is None:
            return 0.0

        loss = 0.0
        for (i, j) in self.bone_pairs:
            pred_dist = torch.norm(pred_relative[:, :, i] - pred_relative[:, :, j], dim=-1)
            target_dist = torch.norm(target_relative[:, :, i] - target_relative[:, :, j], dim=-1)
            loss += torch.mean((pred_dist - target_dist)**2)
        return loss / len(self.bone_pairs)


import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# ---- Hyperparameters ----
MAX_TEXT_LEN = 175
MAX_POSE_LEN = 300
POSE_DIM = 3
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPOCHS = 20
TOTAL_KEYPOINTS = 137
HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 3

# ---- Tokenizer ----
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# ---- Dataset ----
dataset = ASLDataset(train_v_df['sentence'].tolist(), train_v_df['json_data'].tolist(), tokenizer, MAX_TEXT_LEN, MAX_POSE_LEN, POSE_DIM)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ---- Define bone pairs for consistency (example upper body bones) ----
bone_pairs = [(2,3),(3,4),(5,6),(6,7),(2,5)]  # right shoulder-elbow-wrist, left shoulder-elbow-wrist, shoulders

# ---- Decoder & Model ----
decoder = LSTMDecoderRelative(bert_output_dim=768, hidden_size=HIDDEN_SIZE, num_lstm_layers=NUM_LSTM_LAYERS, max_pose_len=MAX_POSE_LEN, total_keypoints=TOTAL_KEYPOINTS, pose_dim=POSE_DIM, use_bone_loss=True, bone_pairs=bone_pairs)

class TextToASLModel(nn.Module):
    def __init__(self, decoder):
        super(TextToASLModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.decoder = decoder

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        bert_seq = bert_outputs.last_hidden_state  # (batch, seq_len, 768)
        pose_output = self.decoder(bert_seq)
        return pose_output

# ---- Instantiate Model ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TextToASLModel(decoder).to(device)

# ---- Loss & Optimizer ----
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ---- Training Loop ----
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target_poses = batch['poses'][:, :, :, :2].to(device)  # relative xy only

        optimizer.zero_grad()

        # Forward
        predictions = model(input_ids, attention_mask)

        # Compute MSE Loss
        mse_loss = loss_fn(predictions, target_poses)

        # Compute optional bone-length loss
        bone_loss = decoder.compute_bone_loss(predictions, target_poses)

        total_batch_loss = mse_loss + 0.1 * bone_loss  # weight bone loss
        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}')

print("Training complete!")


import torch

# Define a path to save the model.
# A good practice is to save it in Google Drive to persist it between sessions.
model_save_path = '/content/drive/MyDrive/ASL_BFH_2D_DATA/asl_model_BERT_LSTM_v2.pth'

# Save the trained model's state_dict
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Assuming the model, tokenizer, and device are already instantiated
# and the model has been trained.
# The `MAX_TEXT_LEN` and `POSE_DIM` variables should be defined.
# You will also need to have `TOTAL_KEYPOINTS` defined from your dataset.
# In your case, that was 137.

import torch
import os

# The text you want to translate
text_input = "hello"

# Set the model to evaluation mode
model.eval()

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# --- 1. Prepare the text input ---
# Tokenize and format the text
text_enc = tokenizer(
    text_input,
    padding='max_length',
    truncation=True,
    max_length=MAX_TEXT_LEN,
    return_tensors='pt'
)

# Move the processed tensors to the same device as the model
input_ids = text_enc['input_ids'].to(device)
attention_mask = text_enc['attention_mask'].to(device)

# --- 2. Run a forward pass ---
# Disable gradient calculation for efficiency
with torch.no_grad():
    predicted_poses = model(input_ids, attention_mask)

# --- 3. Process the output ---
# Move the tensor back to the CPU
predicted_poses_cpu = predicted_poses.cpu()

# Reshape the output tensor to (frames, keypoints, dimensions)
TOTAL_KEYPOINTS = 137
final_pose_output = predicted_poses_cpu.reshape(MAX_POSE_LEN, TOTAL_KEYPOINTS, POSE_DIM)


# Now, `final_pose_output` contains the predicted ASL pose data, ready for visualization.
print(final_pose_output.shape)

# --- 4. SAVE THE OUTPUT ---
import json
import os

# Define the number of keypoints for each body part based on your original data source.
# You must get these numbers from the tool that generated your dataset.
POSE_KEYPOINTS_COUNT = 25  # Example: Pose keypoints (body, arms, legs)
FACE_KEYPOINTS_COUNT = 70  # Example: Facial landmarks
HAND_KEYPOINTS_COUNT = 21  # Example: Hand landmarks

# Define the output path in Google Drive
output_path = '/content/drive/MyDrive/ASL_BFH_2D_DATA/output_hello_v2.json'

# Create the directory if it doesn't exist
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Convert the PyTorch tensor to a NumPy array for easier slicing
final_pose_array = final_pose_output.cpu().numpy()

# Create a list to hold all the frames
json_output_list = []

# Iterate through each frame in the sequence
for frame_data in final_pose_array:
    # Slice the single frame's data into different keypoint types
    pose_keypoints = frame_data[:POSE_KEYPOINTS_COUNT]
    face_keypoints = frame_data[POSE_KEYPOINTS_COUNT : POSE_KEYPOINTS_COUNT + FACE_KEYPOINTS_COUNT]
    hand_left_keypoints = frame_data[POSE_KEYPOINTS_COUNT + FACE_KEYPOINTS_COUNT : POSE_KEYPOINTS_COUNT + FACE_KEYPOINTS_COUNT + HAND_KEYPOINTS_COUNT]
    hand_right_keypoints = frame_data[POSE_KEYPOINTS_COUNT + FACE_KEYPOINTS_COUNT + HAND_KEYPOINTS_COUNT :]

    # Flatten the keypoint arrays to match the JSON format's single list
    pose_keypoints_flat = pose_keypoints.flatten().tolist()
    face_keypoints_flat = face_keypoints.flatten().tolist()
    hand_left_keypoints_flat = hand_left_keypoints.flatten().tolist()
    hand_right_keypoints_flat = hand_right_keypoints.flatten().tolist()

    # Create the dictionary for the current frame, matching the JSON structure
    frame_dict = {
        "version": 1.3,
        "people": [
            {
                "person_id": [-1],
                "pose_keypoints_2d": pose_keypoints_flat,
                "face_keypoints_2d": face_keypoints_flat,
                "hand_left_keypoints_2d": hand_left_keypoints_flat,
                "hand_right_keypoints_2d": hand_right_keypoints_flat,
                # These can be empty lists since your model doesn't predict them
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": [],
            }
        ]
    }
    json_output_list.append(frame_dict)

# Write the final list of dictionaries to a JSON file
with open(output_path, 'w') as f:
    json.dump(json_output_list, f, indent=4) # Use indent for a human-readable file

print(f"Pose output saved successfully to {output_path}")