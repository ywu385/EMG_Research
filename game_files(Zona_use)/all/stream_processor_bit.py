import numpy as np
import time
from typing import Generator, List, Tuple
from scipy import signal
import os
from processors import *

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
            # try:
            result = processor.process(result)
            # except Exception as e:
            #     print(f'{processor.name} failed for {e}')
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


######################################################## Used for streaming CSV, TXT files ######################################################################


class TXTStreamer:
    def __init__(self, filepath: str, sampling_rate: int = 1000):
        """
        Initialize the EMG streamer
        
        Args:
            filepath: Path to the EMG data file
            sampling_rate: Original sampling rate of the data (Hz)
        """
        self.filepath = filepath
        self.sampling_rate = sampling_rate
        self.data = self._load_data()
        self.processor = EMGPipeline() 
        self.name = self.process_name(filepath)

    def add_pipeline(self, pipeline):
        self.processor = pipeline
    
    def add_processor(self, processor):
        self.processor.add_processor(processor)
        return self
    
    @staticmethod
    def process_name(file_path):
        """Extract the base filename without extension"""
        return os.path.splitext(os.path.basename(file_path))[0]
        
    def _load_data(self) -> np.ndarray:
        """Load and parse the EMG data file"""
        data_lines = []
        with open(self.filepath, 'r') as f:
            # Skip header lines until we find EndOfHeader
            for line in f:
                if line.strip() == "# EndOfHeader":
                    break
                    
            # Read the actual data
            for line in f:
                # Split the line and convert to float, skip the sequence number (first column)
                values = [float(x.replace('*', '')) for x in line.strip().split()[1:]]
                # Only keep the non-zero columns (the active EMG channels)
                active_values = [v for v in values if any(values)]
                data_lines.append(active_values)
                
        return np.array(data_lines)
    
    def stream(self, duration_seconds: float = 1.0) -> Generator[np.ndarray, None, None]:
        """
        Stream the EMG data at real-time rate
        
        Args:
            duration_seconds: How many seconds of data to yield in each iteration
            
        Yields:
            numpy.ndarray: Array of shape (n_channels, n_samples) containing the EMG data
            where n_samples = duration_seconds * sampling_rate
        """
        samples_per_chunk = int(duration_seconds * self.sampling_rate)
        start_idx = 0
        
        while start_idx < len(self.data):
            chunk = self.data[start_idx:start_idx + samples_per_chunk].T
            yield chunk
            
            # Sleep for the specified duration to maintain real-time streaming
            time.sleep(duration_seconds)
            start_idx += samples_per_chunk
            
    def stream_continuous(self, duration_seconds: float = 1.0) -> Generator[np.ndarray, None, None]:
        """
        Stream the EMG data continuously, looping back to the start when reaching the end
        
        Args:
            duration_seconds: How many seconds of data to yield in each iteration
            
        Yields:
            numpy.ndarray: Array of shape (n_channels, n_samples) containing the EMG data
        """
        samples_per_chunk = int(duration_seconds * self.sampling_rate)
        start_idx = 0
        
        while True:
            # If we don't have enough data left, loop back to the start
            if start_idx + samples_per_chunk > len(self.data):
                start_idx = 0
                
            chunk = self.data[start_idx:start_idx + samples_per_chunk].T
            yield chunk
            
            time.sleep(duration_seconds)
            start_idx += samples_per_chunk


    def stream_processed(self, duration_seconds: float = 1.0):
        for chunk in self.stream_continuous(duration_seconds):
            processed_chunk = self.processor.process(chunk)
            yield processed_chunk

    def process_all(self) -> np.ndarray:
        """
        Process the entire EMG data file with the currently added processors,
        and return the resulting numpy array of shape (channels, samples).
        """
        # Make sure the EMGProcessor is initialized
        if not self.processor.initialized:
            self.processor.initialize(self.data.T)  
            # Note: If your processors assume (channels, samples), we pass data transposed if necessary.
            
        processed_data = self.processor.process(self.data.T)  # shape: (channels, samples)
        return processed_data
