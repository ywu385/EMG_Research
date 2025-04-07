#%%
import traceback
import multiprocessing
import time


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

#%%
print('Loading Bitalino Device')

mac_address = "/dev/tty.BITalino-3C-C2"
device = BITalino(mac_address)
device.battery(10)

streamer = BitaStreamer(device)
#%%
import glob
data_path = 'data/*'
data_files = glob.glob(data_path)
#%%
streamer = TXTStreamer(data_files[0])

pipeline = EMGPipeline()
pipeline.add_processor(ZeroChannelRemover())
pipeline.add_processor(NotchFilter([60], sampling_rate =1000))
streamer.add_pipeline(pipeline)

counter = 0
for chunk in streamer.stream_processed():
    counter += 1
    print(f"Chunk {counter}: {chunk.shape}")


def run_data_flow(chunk_queue):
    print("Running data flow")

    while True:
        try:
            for chunk in streamer.stream_processed():
                if chunk_queue.full():
                    try:
                        chunk_queue.get_nowait()
                    except:
                        pass

                chunk_queue.put(chunk, block=False)
        except Exception as e:
            print(f"Error processing Bitalino data: {e}")
            traceback.print_exc()
            print("Will attempt to reconnect in 3 seconds...")
            time.sleep(3)  # Wait before retrying
