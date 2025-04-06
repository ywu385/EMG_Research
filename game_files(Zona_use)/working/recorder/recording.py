#%%
try:
    from stream_processor_bit import *
    from processors import *
    from revolution_api.bitalino import *
    from post_processing import *
    
    EMG_MODULES_AVAILABLE = True
    print("All EMG modules loaded successfully")
except ImportError as e:
    print(f"Error importing EMG modules: {e}")
    EMG_MODULES_AVAILABLE = False


print('Loading Bitalino Device')

mac_address = "/dev/tty.BITalino-3C-C2"
device = BITalino(mac_address)
device.battery(10)

streamer = BitaStreamer(device)

pipeline = EMGPipeline()
pipeline.add_processor(ZeroChannelRemover())
pipeline.add_processor(NotchFilter([60], sampling_rate =1000))
streamer.add_pipeline(pipeline)

for chunk in streamer.stream_processed():
    print(chunk)
