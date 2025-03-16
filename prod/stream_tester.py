#%%
from processors import *
from post_processing import *
from stream_processor_bit import *

import glob 
data_path = glob.glob('../data/newest data/*')
streamer = TXTStreamer(data_path[0])

#%%

pipeline = EMGPipeline()
# pipeline.add_processor(ZeroChannelRemover())
pipeline.add_processor(FiveChannels())   # Only use this if model is trained on 5 channels
pipeline.add_processor(NotchFilter([60],sampling_rate = 1000))
pipeline.add_processor(DCRemover())


streamer.add_pipeline(pipeline)

#%%
model_path = 'models/*'
models = glob.glob(model_path)

# import pickle
# with open(models[0], 'rb') as file:
#     model = pickle.load(file)

model_processor = ModelProcessor(
    model= models[0],
    window_size=250,  # 250ms window
    overlap=0.5,      # 50% overlap
    sampling_rate=1000
)
#%%


#%%

buffer = SignalBuffer(window_size = 250, overlap = 0.5)
intensity_processor = IntensityProcessor(scaling_factor=1.5) # Initialize 


for s in streamer.stream_processed(duration_seconds= 0.1):
    # print(f'processed shape: {s.shape}')
    windows = buffer.add_chunk(s)
    for w in windows:
        prediction = model_processor.process(w) #processes and outputs predictions
        intensity_metrics = intensity_processor.process(w)
