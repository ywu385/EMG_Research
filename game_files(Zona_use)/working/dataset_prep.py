import pandas as pd
import numpy as np
import glob
from post_processing import *
from processors import *
from stream_processor_bit import *
from train_helpers import *
from post_processing import FeatureUtils, WaveletFeatureExtractor
from data_labeling import RestActiveLabeler

def main():
    # Set data path
    data_path = './emg_recordings/'
    
    # Find all files
    files = glob.glob(data_path + '*.txt')  # Make sure we only get txt files
    labeled_files = [file for file in files if 'random' not in file]
    
    print(f"Found {len(labeled_files)} labeled files to process")
    
    label_dict = {}
    for f in labeled_files:
        print(f'Converting {f}')
        pipeline = EMGPipeline()
        pipeline.add_processor(NotchFilter([60], sampling_rate=1000))
        pipeline.add_processor(DCRemover())
        pipeline.add_processor(AdaptiveMaxNormalizer())
        streamer = TXTStreamer(f)
        streamer.add_pipeline(pipeline)
        all_data = streamer.process_all()
        label_dict[f.split('/')[-1]] = all_data
    
    # Configure standard labeler
    standard_labeler = RestActiveLabeler(
        rest_threshold_percentile=60,
        local_minima_margin=0.2,
        max_slope_factor=0.2,
        max_rest_rms_percentile=60,
        rest_expansion=2
    )
    
    # Configure labelers dictionary
    labeler_dict = {
        'downward': standard_labeler,
        'upward': standard_labeler,
        # Add other gestures if needed
    }
    
    # Generate initial feature dataset
    feature_df = create_feature_dataset_flexible(
        label_dict,
        feature_extractor=FeatureUtils(),
        labelers_dict=labeler_dict,
        default_labeler=standard_labeler
    )
    
    # Process with different wavelets
    wavelet_types = ['sym4', 'sym5', 'db4']
    wavelet_dfs = []
    
    for wavelet in wavelet_types:
        print(f"Processing wavelet: {wavelet}")
        wave_extract = WaveletFeatureExtractor(wavelet=wavelet, levels=2)
        wavelet_df = create_feature_dataset_flexible(
            label_dict,
            feature_extractor=wave_extract,
            labelers_dict=labeler_dict,
            default_labeler=standard_labeler
        )
        wavelet_df['type'] = wavelet
        wavelet_dfs.append(wavelet_df)
    
    # Process and rename wavelet columns
    new_wavelets_df = []
    
    # Add the regular feature dataframe (non-wavelet)
    new_wavelets_df.append(feature_df)
    
    # Process each wavelet dataframe
    for w in wavelet_dfs:
        wtype = w['type'].unique()[0]
        
        # Make a copy to avoid modifying the original
        w_copy = w.drop(columns=['type', 'name', 'label', 'activity', 'mixed_label'])
        
        # Rename columns to include wavelet type
        w_cols = w_copy.columns
        col_map = {col: f"{col}_{wtype}" for col in w_cols}
        
        # Apply renaming and add to list
        w_renamed = w_copy.rename(columns=col_map)
        new_wavelets_df.append(w_renamed)
    
    # Create wide dataframe by joining all the dataframes
    print("Creating wide dataframe...")
    wide_df = feature_df  # Start with the feature dataframe
    
    # Join the wavelet dataframes
    for i, w in enumerate(new_wavelets_df[1:], 1):  # Skip the first one as it's already in wide_df
        # Join on index
        try:
            wide_df = wide_df.join(w)
            print(f"Joined wavelet dataframe {i} - shape: {wide_df.shape}")
        except Exception as e:
            print(f"Error joining wavelet dataframe {i}: {e}")
    
    # Save to CSV
    output_file = data_path + 'training_data.csv'
    print(f"Saving wide dataframe to {output_file}")
    wide_df.to_csv(output_file, index=False)
    print(f"Dataset preparation complete! Final shape: {wide_df.shape}")

if __name__ == '__main__':
    main()