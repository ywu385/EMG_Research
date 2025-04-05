import numpy as np
import pandas as pd
import csv
from typing import Dict, List
from post_processing import *
from data_labeling import *

def standardize_label(label):
    """Standardize label names to handle variations"""
    if label == 'inwards':
        return 'inward'
    elif label == 'outwards':
        return 'outward'
    elif label == 'upwards':
        return 'upward'
    elif label == 'downwards':
        return 'downward'
    return label

def create_feature_dataset_flexible(
    label_dict, 
    feature_extractor, 
    labelers_dict=None, 
    default_labeler=None,
    window_size=250, 
    overlap=0.5, 
    chunk_size=100, 
    trim_percent=5
):
    """
    Create a feature dataset from normalized EMG signals with flexible feature extraction
    and activity labeling. Adapts to different labeler types.
    
    Args:
        label_dict: Dictionary mapping filenames to normalized EMG data
        feature_extractor: Feature extractor instance with extract_features method
        labelers_dict: Dictionary mapping labels to activity labeler instances
                       If None, default_labeler will be used for all
        default_labeler: Default labeler to use if label not in labelers_dict
        window_size: Size of window for feature extraction (in samples)
        overlap: Overlap between consecutive windows (as a fraction)
        chunk_size: Size of chunks to process at a time (in samples)
        trim_percent: Percentage to trim from beginning and end of each recording
        
    Returns:
        DataFrame containing extracted features, with name and label columns
    """
    feature_rows = []
    
    # Set default labeler if none provided
    if default_labeler is None and labelers_dict is None:
        # Use the old default labeler type for backward compatibility
        default_labeler = ImprovedActivityLabeler(
            rest_threshold_percentile=35,
            derivative_threshold_factor=0.5
        )
    
    # Initialize labelers dict if None
    if labelers_dict is None:
        labelers_dict = {}
    
    # Calculate stride based on window size and overlap
    stride = int(window_size * (1 - overlap))
    
    # Process each file
    for filename, data in label_dict.items():
        # Parse name and label from filename
        file_base = filename.split('.')[0]
        name_parts = file_base.split('_')
        
        # Extract name (first part) and label (last part if more than one part)
        name = name_parts[0]
        label = name_parts[-1] if len(name_parts) > 1 else "unknown"
        label = standardize_label(label)
            
        print(f'Processing: Name: {name}, Label: {label}')
        
        n_channels, n_samples = data.shape
        
        # Calculate trim amount in samples
        trim_samples = int(n_samples * (trim_percent / 100))
        
        # Trim the data (skip first and last X% of samples)
        if 2 * trim_samples < n_samples:  # Only trim if we have enough data
            trimmed_data = data[:, trim_samples:n_samples-trim_samples]
            print(f"  Trimmed {trim_percent}% ({trim_samples} samples) from beginning and end")
            print(f"  Original length: {n_samples}, Trimmed length: {trimmed_data.shape[1]}")
        else:
            # If recording is too short to trim, use all data
            trimmed_data = data
            print(f"  Warning: Recording too short to trim {trim_percent}% from both ends")
            print(f"  Using full recording length: {n_samples} samples")
        
        # Choose appropriate labeler for this label
        if label in labelers_dict:
            labeler = labelers_dict[label]
            print(f"  Using custom labeler for label: {label}")
        elif default_labeler is not None:
            labeler = default_labeler
            print(f"  Using default labeler for label: {label}")
        else:
            # Create a new default labeler using original class for backward compatibility
            labeler = ImprovedActivityLabeler(
                rest_threshold_percentile=35,
                derivative_threshold_factor=0.5
            )
            print(f"  Created new default labeler for label: {label}")
        
        # Initialize labeler with this data
        labeler.initialize_from_data(trimmed_data)
        
        # Special handling for labelers that have a visualize method
        # if hasattr(labeler, 'visualize'):
        #     fig, _ = labeler.visualize(trimmed_data, n_windows=100)
        #     plt.show()
        # else:
        #     # Fallback to old visualization method
        #     fig = labeler.visualize_thresholds(trimmed_data, n_windows=100)
        #     plt.show()
        
        # Special handling for labelers that support batch processing
        position_to_label = {}
        if hasattr(labeler, 'label_windows'):
            print("  Using batch labeling for better rest detection")
            # PRE-COMPUTE ALL WINDOWS AND LABELS:
            # Create all windows from the entire signal
            trimmed_n_samples = trimmed_data.shape[1]
            all_windows = []
            window_indices = []  # Remember where each window came from
            
            for i in range(0, trimmed_n_samples - window_size + 1, stride):
                window = trimmed_data[:, i:i+window_size]
                all_windows.append(window)
                window_indices.append(i)
            
            # Get labels for all windows at once
            all_labels = labeler.label_windows(all_windows)
            
            # Create a map from window start position to activity label
            position_to_label = {start_idx: label for start_idx, label in zip(window_indices, all_labels)}
            
            print(f"  Pre-computed {len(all_labels)} window labels: {all_labels.count('active')} active, {all_labels.count('rest')} rest")
        
        # Process trimmed data in chunks
        trimmed_n_samples = trimmed_data.shape[1]
        buffer = SignalBuffer(window_size=window_size, overlap=overlap)
        
        for i in range(0, trimmed_n_samples, chunk_size):
            chunk = trimmed_data[:, i:min(i+chunk_size, trimmed_n_samples)]
            windows = buffer.add_chunk(chunk)
            
            # Extract features from each complete window
            for window_idx, window in enumerate(windows):
                # For labelers that support batch processing:
                # Try to find a pre-computed label if available
                activity_label = None
                
                if position_to_label:
                    # Calculate the global position of this window
                    # This depends on how your SignalBuffer works
                    # If your SignalBuffer returns absolute positions, use those
                    # Otherwise, estimate the position:
                    approx_window_position = i + window_idx * stride
                    
                    # Find the closest pre-computed window position
                    closest_positions = sorted(position_to_label.keys(), 
                                             key=lambda x: abs(x - approx_window_position))
                    
                    # Use the pre-computed label if within a reasonable distance
                    if closest_positions and abs(closest_positions[0] - approx_window_position) <= stride:
                        activity_label = position_to_label[closest_positions[0]]
                
                # If no pre-computed label was found, use the standard method
                if activity_label is None:
                    activity_label = labeler.label_window(window)
                
                if activity_label == 'rest':
                    mixed_label = 'rest'
                else:
                    mixed_label = label
                
                features = {'name': name, 'label': label, 'activity': activity_label, 'mixed_label': mixed_label}
                
                # Extract features for each channel
                for channel in range(n_channels):
                    channel_features = feature_extractor.extract_features(window[channel])
                    # Add channel index to feature names
                    for feature_name, feature_value in channel_features.items():
                        features[f'{channel+1}_{feature_name}'] = feature_value
                
                feature_rows.append(features)

    # Convert to DataFrame
    df = pd.DataFrame(feature_rows)
    
    # Reorder columns to have name and label at the end
    cols = df.columns.tolist()
    cols = [col for col in cols if col not in ['name', 'label', 'activity', 'mixed_label']] + ['name', 'label', 'activity', 'mixed_label']
    df = df[cols]
    
    return df