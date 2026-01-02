import cv2
import numpy as np
import json
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import glob
import os

# OpenPose 25 keypoint connections (correct skeleton)
# OpenPose uses 25 keypoints, not 17 like COCO
POSE_CONNECTIONS = [
    (1, 2),   # neck -> right_shoulder
    (1, 5),   # neck -> left_shoulder
    (2, 3),   # right_shoulder -> right_elbow
    (3, 4),   # right_elbow -> right_wrist
    (5, 6),   # left_shoulder -> left_elbow
    (6, 7),   # left_elbow -> left_wrist
    (1, 8),   # neck -> mid_hip
    (8, 9),   # mid_hip -> right_hip
    (9, 10),  # right_hip -> right_knee
    (10, 11), # right_knee -> right_ankle
    (8, 12),  # mid_hip -> left_hip
    (12, 13), # left_hip -> left_knee
    (13, 14), # left_knee -> left_ankle
    (1, 0),   # neck -> nose
    (0, 15),  # nose -> right_eye
    (15, 17), # right_eye -> right_ear
    (0, 16),  # nose -> left_eye
    (16, 18), # left_eye -> left_ear
    (14, 19), # left_ankle -> left_big_toe
    (19, 20), # left_big_toe -> left_small_toe
    (14, 21), # left_ankle -> left_heel
    (11, 22), # right_ankle -> right_big_toe
    (22, 23), # right_big_toe -> right_small_toe
    (11, 24), # right_ankle -> right_heel
]

# Face keypoint connections (simplified face outline and features)
FACE_CONNECTIONS = [
    # Face outline (jawline)
    *[(i, i+1) for i in range(0, 16)],
    # Eyebrows
    *[(i, i+1) for i in range(17, 21)],
    *[(i, i+1) for i in range(22, 26)],
    # Nose
    *[(i, i+1) for i in range(27, 30)],
    *[(i, i+1) for i in range(31, 35)],
    # Left eye
    *[(i, i+1) for i in range(36, 41)], (41, 36),
    # Right eye  
    *[(i, i+1) for i in range(42, 47)], (47, 42),
    # Mouth outline
    *[(i, i+1) for i in range(48, 59)], (59, 48),
    # Inner mouth
    *[(i, i+1) for i in range(60, 67)], (67, 60),
]

# Hand keypoint connections (hand skeleton)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger  
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# OpenPose 25 keypoint names
KEYPOINT_NAMES = [
    'nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
    'left_shoulder', 'left_elbow', 'left_wrist', 'mid_hip', 'right_hip',
    'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle',
    'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe',
    'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel'
]

def load_keypoints_from_json(json_file):
    """
    Load keypoints from OpenPose JSON file.
    Expected format: OpenPose output with "people" array containing pose_keypoints_2d
    """
    keypoints_by_frame = []
    
    # If it's a single JSON file
    if json_file.endswith('.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract pose keypoints for first person
        if data.get('people') and len(data['people']) > 0:
            pose_keypoints = data['people'][0]['pose_keypoints_2d']
            # Convert flat list to (25, 3) array - OpenPose uses 25 keypoints
            keypoints = np.array(pose_keypoints).reshape(-1, 3)
            keypoints_by_frame.append(keypoints)
    
    return keypoints_by_frame

def load_keypoints_from_json_sequence(json_folder):
    keypoints_by_frame = []
    json_files = sorted(glob.glob(os.path.join(json_folder, '*.json')))
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if not data['people']:
            # No detections — create empty placeholders
            frame_data = {
                'pose': np.zeros((25, 3)),       # OpenPose BODY_25 model
                'face': np.zeros((70, 3)),       # OpenPose face
                'hand_left': np.zeros((21, 3)),  # Left hand
                'hand_right': np.zeros((21, 3))  # Right hand
            }
        else:
            person = data['people'][0]  # Assuming single person
            pose_keypoints = np.array(person.get('pose_keypoints_2d', [])).reshape((-1, 3))
            face_keypoints = np.array(person.get('face_keypoints_2d', [])).reshape((-1, 3))
            hand_left_keypoints = np.array(person.get('hand_left_keypoints_2d', [])).reshape((-1, 3))
            hand_right_keypoints = np.array(person.get('hand_right_keypoints_2d', [])).reshape((-1, 3))

            frame_data = {
                'pose': pose_keypoints,
                'face': face_keypoints,
                'hand_left': hand_left_keypoints,
                'hand_right': hand_right_keypoints
            }
        
        keypoints_by_frame.append(frame_data)
    
    return keypoints_by_frame

def load_keypoints_from_numpy(npy_file):
    """
    Load keypoints from NumPy file.
    Expected shape: (num_frames, num_keypoints, 3) where 3 = [x, y, confidence]
    or (num_frames, num_keypoints, 2) where 2 = [x, y]
    """
    keypoints = np.load(npy_file)
    
    # If no confidence scores, add them (set to 1.0)
    if keypoints.shape[2] == 2:
        confidence = np.ones((keypoints.shape[0], keypoints.shape[1], 1))
        keypoints = np.concatenate([keypoints, confidence], axis=2)
    
    return [keypoints[i] for i in range(keypoints.shape[0])]

def draw_keypoints(frame, frame_data, confidence_threshold=0.3):
    """
    Draw pose, face, and hand keypoints on a frame.
    
    Args:
        frame: OpenCV image (BGR format)
        frame_data: Dict containing 'pose', 'face', 'hand_left', 'hand_right' keypoints
        confidence_threshold: minimum confidence to draw keypoint
    """
    h, w = frame.shape[:2]
    
    # Draw pose skeleton
    pose_keypoints = frame_data['pose']
    for connection in POSE_CONNECTIONS:
        kp1_idx, kp2_idx = connection
        
        if (kp1_idx < len(pose_keypoints) and kp2_idx < len(pose_keypoints) and
            pose_keypoints[kp1_idx][2] > confidence_threshold and
            pose_keypoints[kp2_idx][2] > confidence_threshold):
            
            pt1 = (int(pose_keypoints[kp1_idx][0]), int(pose_keypoints[kp1_idx][1]))
            pt2 = (int(pose_keypoints[kp2_idx][0]), int(pose_keypoints[kp2_idx][1]))
            
            # Color skeleton connections based on body part
            if kp1_idx in [2, 3, 4] or kp2_idx in [2, 3, 4]:  # Right arm
                color = (0, 255, 255)  # Yellow
            elif kp1_idx in [5, 6, 7] or kp2_idx in [5, 6, 7]:  # Left arm
                color = (0, 128, 255)  # Orange
            elif kp1_idx in [9, 10, 11] or kp2_idx in [9, 10, 11]:  # Right leg
                color = (255, 0, 255)  # Magenta
            elif kp1_idx in [12, 13, 14] or kp2_idx in [12, 13, 14]:  # Left leg
                color = (0, 255, 0)  # Green
            else:  # Torso, head
                color = (255, 255, 255)  # White
            
            cv2.line(frame, pt1, pt2, color, 2)
    
    # Draw pose keypoints
    for i, keypoint in enumerate(pose_keypoints):
        if keypoint[2] > confidence_threshold:
            x, y = int(keypoint[0]), int(keypoint[1])
            
            # Different colors for different body parts (like the reference image)
            if i == 0:  # Nose
                color = (255, 255, 255)  # White
            elif i == 1:  # Neck
                color = (255, 255, 255)  # White
            elif i in [2, 3, 4]:  # Right arm
                color = (0, 255, 255)  # Yellow
            elif i in [5, 6, 7]:  # Left arm  
                color = (0, 128, 255)  # Orange
            elif i in [8]:  # Mid hip
                color = (255, 255, 255)  # White
            elif i in [9, 10, 11]:  # Right leg
                color = (255, 0, 255)  # Magenta
            elif i in [12, 13, 14]:  # Left leg
                color = (0, 255, 0)  # Green
            elif i in [15, 16]:  # Eyes
                color = (255, 0, 255)  # Magenta
            elif i in [17, 18]:  # Ears
                color = (255, 255, 255)  # White
            else:  # Feet/toes
                color = (255, 255, 0)  # Cyan
            
            cv2.circle(frame, (x, y), 4, color, -1)
    
    # Draw face keypoints
    face_keypoints = frame_data['face']
    if len(face_keypoints) > 0:
        # Draw face connections7
        for connection in FACE_CONNECTIONS:
            kp1_idx, kp2_idx = connection
            if (kp1_idx < len(face_keypoints) and kp2_idx < len(face_keypoints) and
                face_keypoints[kp1_idx][2] > confidence_threshold and
                face_keypoints[kp2_idx][2] > confidence_threshold):
                
                pt1 = (int(face_keypoints[kp1_idx][0]), int(face_keypoints[kp1_idx][1]))
                pt2 = (int(face_keypoints[kp2_idx][0]), int(face_keypoints[kp2_idx][1]))
                cv2.line(frame, pt1, pt2, (255, 0, 255), 1)  # Magenta for face
        
        # Draw face keypoints
        for keypoint in face_keypoints:
            if keypoint[2] > confidence_threshold:
                x, y = int(keypoint[0]), int(keypoint[1])
                cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)  # Small magenta dots
    
    # Draw left hand keypoints
    hand_left = frame_data['hand_left']
    if len(hand_left) > 0:
        # Draw hand connections
        for connection in HAND_CONNECTIONS:
            kp1_idx, kp2_idx = connection
            if (kp1_idx < len(hand_left) and kp2_idx < len(hand_left) and
                hand_left[kp1_idx][2] > confidence_threshold and
                hand_left[kp2_idx][2] > confidence_threshold):
                
                pt1 = (int(hand_left[kp1_idx][0]), int(hand_left[kp1_idx][1]))
                pt2 = (int(hand_left[kp2_idx][0]), int(hand_left[kp2_idx][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)  # Yellow for left hand
        
        # Draw hand keypoints
        for keypoint in hand_left:
            if keypoint[2] > confidence_threshold:
                x, y = int(keypoint[0]), int(keypoint[1])
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)  # Yellow dots
    
    # Draw right hand keypoints
    hand_right = frame_data['hand_right']
    if len(hand_right) > 0:
        # Draw hand connections
        for connection in HAND_CONNECTIONS:
            kp1_idx, kp2_idx = connection
            if (kp1_idx < len(hand_right) and kp2_idx < len(hand_right) and
                hand_right[kp1_idx][2] > confidence_threshold and
                hand_right[kp2_idx][2] > confidence_threshold):
                
                pt1 = (int(hand_right[kp1_idx][0]), int(hand_right[kp1_idx][1]))
                pt2 = (int(hand_right[kp2_idx][0]), int(hand_right[kp2_idx][1]))
                cv2.line(frame, pt1, pt2, (255, 255, 0), 2)  # Cyan for right hand
        
        # Draw hand keypoints
        for keypoint in hand_right:
            if keypoint[2] > confidence_threshold:
                x, y = int(keypoint[0]), int(keypoint[1])
                cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)  # Cyan dots

def create_keypoints_video(keypoints_list, output_path, fps=24, frame_size=(1280, 720), 
                          background_color=(0, 0, 0), confidence_threshold=0.3,
                          apply_smoothing=True, smoothing_method='gaussian', smoothing_params=None,
                          interpolate_missing=True):
    """
    Create video from keypoints data.
    
    Args:
        keypoints_list: List of keypoints for each frame
        output_path: Output video file path
        fps: Frames per second
        frame_size: (width, height) of output video
        background_color: Background color (B, G, R)
        confidence_threshold: Minimum confidence to draw keypoint
        apply_smoothing: Whether to apply temporal smoothing
        smoothing_method: 'gaussian', 'savgol', 'moving_average', or 'exponential'
        smoothing_params: Dict of parameters for smoothing method
        interpolate_missing: Whether to interpolate missing keypoints
    """
    
    # Apply preprocessing if requested
    processed_keypoints = keypoints_list.copy()
    
    if interpolate_missing:
        print("Interpolating missing keypoints...")
        processed_keypoints = interpolate_combined(processed_keypoints)
    
    if apply_smoothing:
        print(f"Applying {smoothing_method} smoothing...")
        if smoothing_params is None:
            smoothing_params = {}
        processed_keypoints = smooth_keypoints(processed_keypoints, 
                                             method=smoothing_method, 
                                             **smoothing_params)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    print(f"Creating video with {len(processed_keypoints)} frames...")
    
    for frame_idx, frame_data in enumerate(processed_keypoints):
        # Create blank frame
        frame = np.full((frame_size[1], frame_size[0], 3), background_color, dtype=np.uint8)
        
        # Draw keypoints on frame
        if len(frame_data['pose']) > 0:
            draw_keypoints(frame, frame_data, confidence_threshold)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame to video
        out.write(frame)
        
        if frame_idx % 30 == 0:  # Progress indicator
            print(f"Processed {frame_idx}/{len(processed_keypoints)} frames")
    
    # Release video writer
    out.release()
    print(f"Video saved to: {output_path}")

def smooth_keypoints(keypoints_list, method='gaussian', **kwargs):
    """
    Smooth keypoints across frames to reduce jitter.
    
    Args:
        keypoints_list: List of frame_data dictionaries
        method: 'gaussian', 'savgol', 'moving_average', or 'exponential'
        **kwargs: Parameters specific to each method
    
    Returns:
        Smoothed keypoints list
    """
    if len(keypoints_list) < 3:
        return keypoints_list
    
    smoothed_list = []
    
    # Process each keypoint type separately
    for keypoint_type in ['pose', 'face', 'hand_left', 'hand_right']:
        # Extract this keypoint type from all frames
        keypoints_array = np.array([frame_data[keypoint_type] for frame_data in keypoints_list])
        smoothed_array = keypoints_array.copy()
        
        num_frames, num_keypoints, _ = keypoints_array.shape
        
        # Only smooth x, y coordinates, keep confidence scores
        for kp_idx in range(num_keypoints):
            for coord_idx in range(2):  # x, y coordinates only
                signal = keypoints_array[:, kp_idx, coord_idx]
                
                # Skip if all values are zero (missing keypoint)
                if np.all(signal == 0):
                    continue
                    
                if method == 'gaussian':
                    sigma = kwargs.get('sigma', 1.0)
                    smoothed_signal = gaussian_filter1d(signal, sigma=sigma)
                    
                elif method == 'savgol':
                    window_length = kwargs.get('window_length', min(7, num_frames if num_frames % 2 == 1 else num_frames-1))
                    if window_length < 3:
                        window_length = 3
                    polyorder = kwargs.get('polyorder', min(2, window_length-1))
                    smoothed_signal = savgol_filter(signal, window_length, polyorder)
                    
                elif method == 'moving_average':
                    window_size = kwargs.get('window_size', 5)
                    smoothed_signal = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
                    
                elif method == 'exponential':
                    alpha = kwargs.get('alpha', 0.3)
                    smoothed_signal = np.zeros_like(signal)
                    smoothed_signal[0] = signal[0]
                    for i in range(1, len(signal)):
                        smoothed_signal[i] = alpha * signal[i] + (1 - alpha) * smoothed_signal[i-1]
                        
                else:
                    smoothed_signal = signal  # No smoothing
                
                smoothed_array[:, kp_idx, coord_idx] = smoothed_signal
        
        # Store smoothed data for this keypoint type
        if keypoint_type == 'pose':
            pose_smoothed = smoothed_array
        elif keypoint_type == 'face':
            face_smoothed = smoothed_array
        elif keypoint_type == 'hand_left':
            hand_left_smoothed = smoothed_array
        elif keypoint_type == 'hand_right':
            hand_right_smoothed = smoothed_array
    
    # Reconstruct the frame_data dictionaries
    for i in range(num_frames):
        frame_data = {
            'pose': pose_smoothed[i],
            'face': face_smoothed[i],
            'hand_left': hand_left_smoothed[i],
            'hand_right': hand_right_smoothed[i]
        }
        smoothed_list.append(frame_data)
    
    return smoothed_list

def interpolate_combined(keypoints_list, confidence_threshold=0.1):
    """
    Interpolate missing keypoints using confidence and zero coordinates,
    then specifically interpolate zeros regardless of confidence.
    """
    if len(keypoints_list) < 3:
        return keypoints_list

    interpolated_list = []
    keypoint_types = ['pose', 'face', 'hand_left', 'hand_right']
    
    # Convert all frames to arrays for easier processing
    data_arrays = {kpt_type: np.array([frame[kpt_type] for frame in keypoints_list]) for kpt_type in keypoint_types}

    for kpt_type in keypoint_types:
        arr = data_arrays[kpt_type]
        num_frames, num_keypoints, _ = arr.shape

        for kp_idx in range(num_keypoints):
            for coord_idx in range(2):
                coords = arr[:, kp_idx, coord_idx]
                confidences = arr[:, kp_idx, 2]

                # 1) Interpolate missing based on confidence and coords != 0 (your original logic)
                valid_mask = (confidences > confidence_threshold) & (coords != 0)
                if np.sum(valid_mask) >= 2:
                    valid_frames = np.where(valid_mask)[0]
                    valid_coords = coords[valid_mask]
                    interpolated = np.interp(np.arange(num_frames), valid_frames, valid_coords)
                    missing_mask = ~valid_mask
                    arr[missing_mask, kp_idx, coord_idx] = interpolated[missing_mask]
                    arr[missing_mask, kp_idx, 2] = 0.5  # interpolated confidence
                
                # 2) Additionally interpolate points that are exactly zero, ignoring confidence
                zero_mask = (arr[:, kp_idx, coord_idx] == 0)
                if np.sum(~zero_mask) >= 2:
                    valid_frames_zero = np.where(~zero_mask)[0]
                    valid_coords_zero = arr[~zero_mask, kp_idx, coord_idx]
                    interpolated_zero = np.interp(np.where(zero_mask)[0], valid_frames_zero, valid_coords_zero)
                    arr[zero_mask, kp_idx, coord_idx] = interpolated_zero

        data_arrays[kpt_type] = arr

    # Rebuild frame dicts
    for i in range(num_frames):
        frame_data = {kpt_type: data_arrays[kpt_type][i] for kpt_type in keypoint_types}
        interpolated_list.append(frame_data)

    return interpolated_list

    """
    Create a demo video with synthetic keypoints data.
    This shows a simple walking motion.
    """
    num_frames = 60
    keypoints_list = []
    
    for frame in range(num_frames):
        # Create synthetic keypoints for a simple walking animation
        keypoints = np.zeros((17, 3))
        
        # Simple oscillating motion
        t = frame / num_frames * 2 * np.pi
        center_x, center_y = 320, 240
        
        # Head (nose, eyes, ears)
        keypoints[0] = [center_x, center_y - 100, 1.0]  # nose
        keypoints[1] = [center_x - 10, center_y - 110, 1.0]  # left eye
        keypoints[2] = [center_x + 10, center_y - 110, 1.0]  # right eye
        keypoints[3] = [center_x - 20, center_y - 105, 1.0]  # left ear
        keypoints[4] = [center_x + 20, center_y - 105, 1.0]  # right ear
        
        # Shoulders
        keypoints[5] = [center_x - 40, center_y - 60, 1.0]  # left shoulder
        keypoints[6] = [center_x + 40, center_y - 60, 1.0]  # right shoulder
        
        # Arms (simple swinging motion)
        arm_swing = np.sin(t * 2) * 20
        keypoints[7] = [center_x - 60, center_y - 20 + arm_swing, 1.0]  # left elbow
        keypoints[8] = [center_x + 60, center_y - 20 - arm_swing, 1.0]  # right elbow
        keypoints[9] = [center_x - 70, center_y + 10 + arm_swing, 1.0]  # left wrist
        keypoints[10] = [center_x + 70, center_y + 10 - arm_swing, 1.0]  # right wrist
        
        # Hips
        keypoints[11] = [center_x - 25, center_y + 20, 1.0]  # left hip
        keypoints[12] = [center_x + 25, center_y + 20, 1.0]  # right hip
        
        # Legs (walking motion)
        leg_swing = np.sin(t * 2) * 30
        keypoints[13] = [center_x - 25 - leg_swing/2, center_y + 80, 1.0]  # left knee
        keypoints[14] = [center_x + 25 + leg_swing/2, center_y + 80, 1.0]  # right knee
        keypoints[15] = [center_x - 25 - leg_swing, center_y + 140, 1.0]  # left ankle
        keypoints[16] = [center_x + 25 + leg_swing, center_y + 140, 1.0]  # right ankle
        
        keypoints_list.append(keypoints)
    
    return keypoints_list

# Example usage
if __name__ == "__main__":
    # MAIN USE CASE: Load from directory of OpenPose JSON files (125 files, one per frame)
    # Put all your JSON files in a folder and specify the path:
    keypoints_list = load_keypoints_from_json_sequence('keypoints_test_file_v2')
    
    # Alternative: If you want to specify individual files
    # json_files = ['frame_001.json', 'frame_002.json', ..., 'frame_125.json']
    # keypoints_list = load_keypoints_from_json_sequence(json_files)
    
    # Demo data (uncomment to test)
    # keypoints_list = demo_with_synthetic_data()
    
    # Create video with 24fps, smoothing, and interpolation
    create_keypoints_video(
        keypoints_list=keypoints_list,
        output_path='video_file_v2/pose_visualization_coordinates.mp4',
        fps=24,
        frame_size=(1280, 720),  # Higher resolution
        background_color=(30, 30, 30),  # Dark gray background
        confidence_threshold=0.1,
        apply_smoothing=True,
        smoothing_method='gaussian',  # Options: 'gaussian', 'savgol', 'moving_average', 'exponential'
        smoothing_params={'sigma': 1.5},  # Adjust smoothing strength
        interpolate_missing=True
    )
    
    print("Done!")