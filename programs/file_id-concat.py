import pandas as pd
import json
import os
from pathlib import Path
import glob

def extract_keypoints(json_data):
    """
    Extract specific keypoints data from JSON in the required order:
    1. pose_keypoints_2d
    2. face_keypoints_2d
    3. hand_left_keypoints_2d
    4. hand_right_keypoints_2d
    """
    extracted = {
        'pose_keypoints_2d': [],
        'face_keypoints_2d': [],
        'hand_left_keypoints_2d': [],
        'hand_right_keypoints_2d': []
    }
    
    # Check if there are people in the JSON
    if 'people' in json_data and len(json_data['people']) > 0:
        # Get the first person's data (assuming single person per frame)
        person_data = json_data['people'][0]
        
        # Extract keypoints in the specified order
        extracted['pose_keypoints_2d'] = person_data.get('pose_keypoints_2d', [])
        extracted['face_keypoints_2d'] = person_data.get('face_keypoints_2d', [])
        extracted['hand_left_keypoints_2d'] = person_data.get('hand_left_keypoints_2d', [])
        extracted['hand_right_keypoints_2d'] = person_data.get('hand_right_keypoints_2d', [])


        # Normalize keypoints
        extracted['pose_keypoints_2d'] = normalize_keypoints(extracted['pose_keypoints_2d'], 1280, 720)
        extracted['face_keypoints_2d'] = normalize_keypoints(extracted['face_keypoints_2d'], 1280, 720)
        extracted['hand_left_keypoints_2d'] = normalize_keypoints(extracted['hand_left_keypoints_2d'], 1280, 720)
        extracted['hand_right_keypoints_2d'] = normalize_keypoints(extracted['hand_right_keypoints_2d'], 1280, 720)

    return [
        extracted.get('pose_keypoints_2d', []),
        extracted.get('face_keypoints_2d', []),
        extracted.get('hand_left_keypoints_2d', []),
        extracted.get('hand_right_keypoints_2d', [])
    ]

def normalize_keypoints(keypoints, frame_width, frame_height):
    """
    Normalize keypoints by dividing x by frame_width and y by frame_height.
    Confidence values remain unchanged.
    
    Args:
        keypoints (list): List of numbers in format [x1, y1, conf1, x2, y2, conf2, ...]
        frame_width (int/float): Width of the frame
        frame_height (int/float): Height of the frame
    
    Returns:
        list: Normalized keypoints [x1/frame_width, y1/frame_height, conf1, ...]
    """
    normalized = []
    
    # Process keypoints in groups of 3 (x, y, confidence)
    for i in range(0, len(keypoints), 3):
        if i + 2 < len(keypoints):  # Make sure we have all 3 values
            x = keypoints[i]
            y = keypoints[i + 1]
            confidence = keypoints[i + 2]
            
            # Normalize x and y, keep confidence unchanged
            if frame_width > 0 and frame_height > 0:
                normalized_x = x / frame_width
                normalized_y = y / frame_height
            else:
                # Handle edge case of zero dimensions
                normalized_x = x
                normalized_y = y
            
            normalized.extend([normalized_x, normalized_y, confidence])
    
    return normalized

def collate_datasets(csv_path, json_folders_path, output_path=None):
    """
    Collate CSV dataset with JSON folders into a single dataset.
    
    Args:
        csv_path (str): Path to the CSV file with sentence_name and sentence columns
        json_folders_path (str): Path to the directory containing folders with JSON files
        output_path (str): Optional path to save the output CSV file
    
    Returns:
        pandas.DataFrame: Combined dataset
    """

    # Read the CSV file
    print("Reading CSV file...")
    df_csv = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    if 'SENTENCE_NAME' not in df_csv.columns or 'SENTENCE' not in df_csv.columns:
        raise ValueError("CSV must contain 'SENTENCE_NAME' and 'SENTENCE' columns")
    
    # Initialize the result dataframe
    result_data = []
    
    # Process each row in the CSV
    print("Processing folders and JSON files...")
    for idx, row in df_csv.iterrows():
        sentence_name = str(row['SENTENCE_NAME'])
        #print(f"Processing folder: {sentence_name}")
        sentence_text = row['SENTENCE'].strip('"')
        
        # Look for corresponding folder
        folder_path = Path(json_folders_path) / sentence_name
        
        if not folder_path.exists():
            print(f"Warning: Folder '{sentence_name}' not found. Skipping...")
            continue
        
        # Collect all JSON files in the folder and sort by frame number
        json_files = glob.glob(str(folder_path / "*.json"))
        #print(f"Found {len(json_files)} JSON files in folder '{sentence_name}'")
        
        def extract_frame_number(filename):
            """Extract the frame number from filename like 'xxx_000000000066_keypoints.json'"""
            import re
            # Look for a sequence of digits (usually 12 digits) in the filename
            match = re.search(r'_(\d{8,}?)_', os.path.basename(filename))
            if match:
                return int(match.group(1))
            else:
                # Fallback: if pattern not found, just use alphabetical order
                print(f"Warning: Failed to extract frame number from {filename}. Using alphabetical order...")
                return 0
        
        json_files = sorted(json_files, key=extract_frame_number) #actually goes through each file and sorts it while calling the extract frame number each time and using that as a key - essentially a collapsed lambda function
        
        if not json_files:
            print(f"Warning: No JSON files found in folder '{sentence_name}'. Adding empty list...")
            json_content_list = []
        else:
            # Read and extract specific keypoints from all JSON files
            json_content_list = []
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        
                        # Extract keypoints data in the specified order
                        extracted_data = extract_keypoints(json_data)

                        json_content_list.append(extracted_data)
                        
                except json.JSONDecodeError as e:
                    print(f"Error reading {json_file}: {e}")
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
        
        # Add to result
        result_data.append({
            'sentence_name': sentence_name,
            'sentence': sentence_text,
            'json_data': json_content_list
        })
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} rows...")
    
    # Create result dataframe
    result_df = pd.DataFrame(result_data)
    
    # Save to file if output path provided
    if output_path:
        print(f"Saving to {output_path}...")
        # Note: JSON data will be saved as string representation in CSV
        result_df.to_csv(output_path, index=False)
        print("Dataset saved successfully!")
    
    print(f"Combined dataset created with {len(result_df)} rows")
    return result_df

def preview_dataset(df, num_rows=5):
    """
    Preview the first few rows of the dataset
    """
    print(f"\nPreview of first {num_rows} rows:")
    print("-" * 80)
    
    for idx in range(min(num_rows, len(df))):
        row = df.iloc[idx]
        print(f"Row {idx + 1}:")
        print(f"  Sentence Name: {row['sentence_name']}")
        print(f"  Sentence: {row['sentence'][:100]}{'...' if len(str(row['sentence'])) > 100 else ''}")
        print(f"  JSON Data: {len(row['json_data'])} frames")
        if row['json_data']:
            first_frame = row['json_data'][0]
            print(f"    First frame keypoints:")
            print(f"      Pose points: {len(first_frame.get('pose_keypoints_2d', [])) // 3} keypoints")
            print(f"      Face points: {len(first_frame.get('face_keypoints_2d', [])) // 3} keypoints") 
            print(f"      Left hand points: {len(first_frame.get('hand_left_keypoints_2d', [])) // 3} keypoints")
            print(f"      Right hand points: {len(first_frame.get('hand_right_keypoints_2d', [])) // 3} keypoints")
        print()

# Example usage
if __name__ == "__main__":
    # Update these paths to match your file locations
    csv_file_path = "validation_csv/df_no-duplicates_v1.csv"  # Path to your CSV file
    json_folders_path = "validation_json"  # Path to directory containing the ID folders
    output_file_path = "validation_csv/combined_dataset_final_v1.csv"  # Where to save the result
    
    try:
        # Collate the datasets
        combined_df = collate_datasets(csv_file_path, json_folders_path, output_file_path)
        
        # Preview the result
        preview_dataset(combined_df)
        
        # Additional statistics
        print(f"\nDataset Statistics:")
        print(f"Total rows: {len(combined_df)}")
        print(Path.cwd())
        print(f"Average frames per sentence: {combined_df['json_data'].apply(len).mean():.2f}")
        print(f"Max frames in a single sentence: {combined_df['json_data'].apply(len).max()}")
        print(f"Min frames in a single sentence: {combined_df['json_data'].apply(len).min()}")
        
        # Keypoint statistics
        def get_keypoint_stats(df):
            stats = {'pose': [], 'face': [], 'hand_left': [], 'hand_right': []}
            for _, row in df.iterrows():
                for frame in row['json_data']:
                    stats['pose'].append(len(frame.get('pose_keypoints_2d', [])) // 3)
                    stats['face'].append(len(frame.get('face_keypoints_2d', [])) // 3)
                    stats['hand_left'].append(len(frame.get('hand_left_keypoints_2d', [])) // 3)
                    stats['hand_right'].append(len(frame.get('hand_right_keypoints_2d', [])) // 3)
            return stats
        
        if len(combined_df) > 0 and len(combined_df.iloc[0]['json_data']) > 0:
            keypoint_stats = get_keypoint_stats(combined_df)
            print(f"\nKeypoint Statistics (per frame):")
            print(f"Average pose keypoints: {sum(keypoint_stats['pose']) / len(keypoint_stats['pose']):.1f}")
            print(f"Average face keypoints: {sum(keypoint_stats['face']) / len(keypoint_stats['face']):.1f}")
            print(f"Average left hand keypoints: {sum(keypoint_stats['hand_left']) / len(keypoint_stats['hand_left']):.1f}")
            print(f"Average right hand keypoints: {sum(keypoint_stats['hand_right']) / len(keypoint_stats['hand_right']):.1f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease check:")
        print("1. CSV file path is correct and contains 'sentence_name' and 'sentence' columns")
        print("2. JSON folders directory path is correct")
        print("3. Folder names match the sentence_name values in the CSV")
        print("4. JSON files are numbered 0, 1, 2, ... n in each folder")