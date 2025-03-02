import numpy as np
import time
from typing import Generator, List, Tuple
from scipy import signal
import os

######################################################## Streamer object ######################################################################
import csv
import time


# Main processor that can chain processing steps
class EMGPipeline:
    def __init__(self):
        self.processors: List[SignalProcessor] = []
        self.initialized = False

    def add_processor(self, processor: SignalProcessor):
        self.processors.append(processor)
        return self

    def initialize(self, data: np.ndarray):
        for processor in self.processors:
            if hasattr(processor, 'initialize'):
                processor.initialize(data)
        self.initialized = True

    def process(self, data: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self.initialize(data)
        result = data
        for processor in self.processors:
            result = processor.process(result)
        return result
    

#%%
class BitaStreamer: 
    def __init__(self, device,sampling_rate = 1000, acqChannels = [0,1,2,3,4,5], nSamples = 10):
        """
        Device is preconfigured after Bitalino is instantiated.  
        (e.g. device = BITalino())
        """
        self.device = device
        self.sampling_rate = sampling_rate
        self.channels = acqChannels
        self.processor = EMGPipeline()
        self.nSamples = nSamples

    def add_pipeline(self, pipeline):
        self.processor = pipeline

    def add_processor(self, processor):
        self.processor.add_processor(processor)
        return self
        
    def get_sample(self):
        samples = self.device.read(self.nSamples)
        sample_output = []
        for s in samples:
            sample_output.extend(s.T)
        return sample_output
    
    def stream_processed(self, duration_seconds=15):
        start = time.time()
        end = time.time()
        
        self.device.start(self.sampling_rate, self.channels)
        
        try:
            while (end - start) < duration_seconds:
                samples = self.device.read(self.nSamples)
                # Convert to numpy array and transpose to match EMGProcessor format
                chunk = np.array(samples)[:, 2:].T  # Skipping first two columns
                processed_chunk = self.processor.process(chunk)
                yield processed_chunk
                end = time.time()
                
        finally:
            self.device.stop()
            self.device.close()
            
    def stream_raw(self, duration_seconds=15):
        start = time.time()
        end = time.time()
        
        self.device.start(self.sampling_rate, self.channels)
        
        try:
            while (end - start) < duration_seconds:
                samples = self.device.read(self.nSamples)
                chunk = np.array(samples)[:, 2:].T
                yield chunk
                end = time.time()
                
        finally:
            self.device.stop()
            self.device.close()


