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
        # pipeline.add_processor(MaxNormalizer())
        # pipeline.add_processor(AdaptiveMaxNormalizer())
        emg_bandpass = RealTimeButterFilter(
                            cutoff=[20, 450],  # Target the 20-450 Hz frequency range for EMG
                            sampling_rate=1000,  # Assuming 1000 Hz sampling rate
                            filter_type='bandpass',
                            order=4  # 4th order provides good balance between sharpness and stability
                        )
        pipeline.add_processor(emg_bandpass)
        
        streamer = TXTStreamer(f)
        streamer.add_pipeline(pipeline)
        all_data = streamer.process_all()
        label_dict[f.split('/')[-1]] = all_data
    
    # Configure standard labeler
    standard_labeler = RestActiveLabeler(
        rest_threshold_percentile=60,
        local_minima_margin=0.2,
        max_slope_factor=0.2,
        max_rest_rms_percentile=50,
        rest_expansion=2
    )

    # Keep the order they took in terms of squares
    # 

    down_labeler = ImprovedActivityLabeler(rest_threshold_percentile=50,
                                       derivative_threshold_factor=0.8)

    up_labeler = ImprovedActivityLabeler(rest_threshold_percentile= 35,
                                     derivative_threshold_factor=0.8)
    

    # Configure labelers dictionary
    labeler_dict = {
        'downward': down_labeler,
        'upward': up_labeler,
        # Add other gestures if needed
    }

    # labeler_dict = {
    #     'downward':standard_labeler,
    #     'upward':standard_labeler
    # }
    
    # Generate all feature datasets
    print("Generating feature datasets in the correct order...")
    
    # First, process with different wavelets (in the same order as model extraction)
    # wavelet_types = ['sym4', 'sym5', 'db4']
    wavelet_types = ['sym5']
    # wavelet_types = []
    all_dfs_with_labels = []
    
    # Process each wavelet type first (in the same order as extract_features method)
    for wavelet in wavelet_types:
        print(f"Processing wavelet: {wavelet}")
        wave_extract = WaveletFeatureExtractor(wavelet=wavelet, levels=2)
        wavelet_df = create_feature_dataset_flexible(
            label_dict,
            feature_extractor=wave_extract,
            labelers_dict=labeler_dict,
            default_labeler=standard_labeler
        )
        all_dfs_with_labels.append((wavelet_df, f"wavelet_{wavelet}"))
    
    # Then process basic features (last in extraction order)
    print("Generating basic features dataset...")
    feature_df = create_feature_dataset_flexible(
        label_dict,
        feature_extractor=FeatureUtils(),
        labelers_dict=labeler_dict,
        default_labeler=standard_labeler
    )
    all_dfs_with_labels.append((feature_df, "basic"))
    
    # Add a unique identifier to each dataset to ensure alignment
    for i, (df, _) in enumerate(all_dfs_with_labels):
        df['data_id'] = range(len(df))
    
    # Extract and save labels from any dataframe (they should all have the same labels)
    label_columns = ['name', 'label', 'activity', 'mixed_label', 'data_id']
    labels_df = all_dfs_with_labels[0][0][label_columns].copy()
    
    # Prepare dataframes for merging
    feature_dfs = []
    
    # Process each dataframe to prepare for merging
    for df, df_type in all_dfs_with_labels:
        # Drop label columns except data_id
        features_only = df.drop(columns=[col for col in label_columns if col != 'data_id'])
        
        # For wavelet dataframes, rename columns to include wavelet type
        if df_type.startswith("wavelet_"):
            wavelet = df_type.split("_")[1]
            # Rename columns (except data_id)
            columns_to_rename = [col for col in features_only.columns if col != 'data_id']
            col_map = {col: f"{col}_{wavelet}" for col in columns_to_rename}
            features_only = features_only.rename(columns=col_map)
        
        feature_dfs.append(features_only)
    
    # Merge in the exact order needed for the model
    print("Merging all feature dataframes in the correct order...")
    
    # Start with an empty dataframe containing only the data_id
    final_features = pd.DataFrame({'data_id': range(len(feature_dfs[0]))})
    
    # Merge each feature dataframe in order
    for i, df in enumerate(feature_dfs):
        try:
            final_features = pd.merge(final_features, df, on='data_id', how='inner')
            df_type = all_dfs_with_labels[i][1]
            print(f"Merged {df_type} features - shape: {final_features.shape}")
        except Exception as e:
            print(f"Error merging dataframe {i}: {e}")
    
    # Add the labels back
    final_df = pd.merge(final_features, labels_df, on='data_id', how='inner')
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
    
    # Print the first 20 columns to verify order
    print(f"First 20 columns (to verify order): {list(final_df.columns)[:20]}")

    # Print class outputs
    print(final_df['mixed_label'].value_counts())

if __name__ == '__main__':
    main()