#%%
from post_processing import *
from processors import *
from stream_processor_bit import *
import numpy as np
from train_helpers import *
from post_processing import FeatureUtils, WaveletFeatureExtractor
from data_labeling import RestActiveLabeler
import glob
#%%

def main():

#%%
    data_path = './emg_recordings/'
    files = glob.glob(data_path+'*')

    labeled_files = [file for file in files if 'random' not in file]

    label_dict = {}
    for f in labeled_files:
        print(f'converting {f}')
        pipeline=EMGPipeline()
        pipeline.add_processor(DCRemover())
        pipeline.add_processor(AdaptiveMaxNormalizer())
        streamer = TXTStreamer(f)
        streamer.add_pipeline(pipeline)
        all = streamer.process_all()
        label_dict[f.split('/')[-1]] =all


    standard_labeler = RestActiveLabeler(rest_threshold_percentile=60,
                                     local_minima_margin=0.2,
                                     max_slope_factor= 0.2,
                                     max_rest_rms_percentile=60,
                                     rest_expansion=2
                                     )
    
    labeler_dict = {''
# '               inward': up_labeler,
                'downward':standard_labeler,
                'upward':standard_labeler,
                # 'inward':up_labeler
                }
    
    feature_df = create_feature_dataset_flexible(label_dict, 
                                            feature_extractor=FeatureUtils(),
                                            labelers_dict=labeler_dict,
                                            default_labeler= standard_labeler
                                            )

    wavelet_types = ['sym4','sym5','db4']
    wavelet_dfs = []

    for wavelet in wavelet_types:
        wave_extract = WaveletFeatureExtractor(wavelet=wavelet,levels=2)
        wavelet_df = create_feature_dataset_flexible(label_dict,
                                                    feature_extractor=wave_extract, 
                                                    labelers_dict=labeler_dict,
                                                    default_labeler=standard_labeler)
        wavelet_df['type'] = wavelet
        wavelet_dfs.append(wavelet_df)
        

    new_wavelets_df = []
    for w in wavelet_dfs:
        wtype = w['type'].unique()[0]
        w = w.drop(columns=['type',
                            'name',
                            'label',
                            'activity',
                            'mixed_label'])  # 'columns' not 'column'
        w_cols = w.columns  # w.columns is an attribute, not a method
        col_map = {}
        for col in w_cols:
            col_map[col] = col + '_' + wtype  # You need to add wtype as a string
        
        # You're missing the code to rename the columns and append to the list
        w = w.rename(columns=col_map)
        new_wavelets_df.append(w)

    new_wavelets_df.append(feature_df)

    wide_df = pd.DataFrame()
    for w in new_wavelets_df:
        # For the first dataframe, just use it as is
        if wide_df.empty:
            wide_df = w
        else:
            # For subsequent dataframes, join them to the existing wide_df
            # You'll need a common index to join on
            wide_df = wide_df.join(w)
    #%%
    wide_df.drop(columns=['label','activity','mixed_label','name'])
    wide_df.to_csv(data_path+'training_data.csv',index=False)

if __name__ == '__main__':
    main()