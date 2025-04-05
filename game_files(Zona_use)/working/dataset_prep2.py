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
    print("Generating basic features dataset...")
    feature_df = create_feature_dataset_flexible(
        label_dict,
        feature_extractor=FeatureUtils(),
        labelers_dict=labeler_dict,
        default_labeler=standard_labeler
    )
    
    # Add a unique identifier to each row in the feature dataframe
    feature_df['data_id'] = range(len(feature_df))
    
    # Extract label columns to preserve them
    label_columns = ['name', 'label', 'activity', 'mixed_label', 'data_id']
    labels_df = feature_df[label_columns]
    
    # Process with different wavelets
    print("Processing wavelet features...")
    wavelet_types = ['sym4', 'sym5', 'db4']
    all_feature_dfs = [feature_df.drop(columns=label_columns[:-1])]  # Keep data_id
    
    for wavelet in wavelet_types:
        print(f"Processing wavelet: {wavelet}")
        wave_extract = WaveletFeatureExtractor(wavelet=wavelet, levels=2)
        wavelet_df = create_feature_dataset_flexible(
            label_dict,
            feature_extractor=wave_extract,
            labelers_dict=labeler_dict,
            default_labeler=standard_labeler
        )
        
        # Ensure wavelet dataframe has the same number of rows and order as feature_df
        if len(wavelet_df) != len(feature_df):
            print(f"Warning: Wavelet dataframe {wavelet} has different length than main feature dataframe")
            print(f"Feature DF: {len(feature_df)} rows, Wavelet DF: {len(wavelet_df)} rows")
            continue
        
        # Add the same data_id
        wavelet_df['data_id'] = range(len(wavelet_df))
        
        # Drop label columns, we'll add them back at the end
        wavelet_features = wavelet_df.drop(columns=['name', 'label', 'activity', 'mixed_label'])
        
        # Rename columns to include wavelet type (except data_id)
        columns_to_rename = [col for col in wavelet_features.columns if col != 'data_id']
        col_map = {col: f"{col}_{wavelet}" for col in columns_to_rename}
        wavelet_features = wavelet_features.rename(columns=col_map)
        
        # Add to our list of feature dataframes
        all_feature_dfs.append(wavelet_features)
    
    # Merge all feature dataframes using data_id as the key
    print("Merging all feature dataframes...")
    merged_df = all_feature_dfs[0]  # Start with the base features
    
    for i, df in enumerate(all_feature_dfs[1:], 1):
        try:
            merged_df = pd.merge(merged_df, df, on='data_id', how='inner')
            print(f"Merged wavelet dataframe {i} - shape: {merged_df.shape}")
        except Exception as e:
            print(f"Error merging wavelet dataframe {i}: {e}")
    
    # Add the labels back
    final_df = pd.merge(merged_df, labels_df, on='data_id', how='inner')
    print(f"Added labels back - shape: {final_df.shape}")
    
    # Drop the temporary id column
    final_df = final_df.drop(columns=['data_id'])
    
    # Save to CSV
    output_file = data_path + 'training_data.csv'
    print(f"Saving final dataframe to {output_file}")
    final_df.to_csv(output_file, index=False)
    
    # Print column information to verify labels are included
    print(f"Dataset preparation complete! Final shape: {final_df.shape}")
    print(f"Label columns in final dataset: {[col for col in final_df.columns if col in ['name', 'label', 'activity', 'mixed_label']]}")
    print(f"Total columns: {len(final_df.columns)}")

if __name__ == '__main__':
    main()