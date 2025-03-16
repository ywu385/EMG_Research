from statistics import mode
import numpy as np
from processors import SignalProcessor
import pickle
from typing import List

######################################################## Buffering class ######################################################################
class SignalBuffer:
    """Efficiently buffers signal data for overlapping windows"""
    def __init__(self, window_size=250, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap
        self.stride = int(window_size * (1 - overlap))
        self.buffer = None
        
    def initialize(self, n_channels):
        # Initialize with just enough space for one window
        self.buffer = np.zeros((n_channels, self.window_size))
        self.buffer_filled = 0  # Track how many samples are in the buffer
        
    def add_chunk(self, chunk: np.ndarray) -> List[np.ndarray]:
        """
        Add a new chunk to the buffer and return complete windows
        
        Args:
            chunk: New data chunk of shape (n_channels, n_samples)
            
        Returns:
            List of complete windows that can be formed
        """
        n_channels, n_samples = chunk.shape
        
        # Initialize buffer if needed
        if self.buffer is None:
            self.initialize(n_channels)
        
        windows = []
        chunk_pos = 0
        
        # While we have data to process in the chunk
        while chunk_pos < n_samples:
            # Calculate how much space is left in the buffer
            space_left = self.window_size - self.buffer_filled
            
            # Calculate how many samples we can add from the chunk
            samples_to_add = min(space_left, n_samples - chunk_pos)
            
            # Add samples to buffer
            self.buffer[:, self.buffer_filled:self.buffer_filled + samples_to_add] = \
                chunk[:, chunk_pos:chunk_pos + samples_to_add]
            
            # Update positions
            self.buffer_filled += samples_to_add
            chunk_pos += samples_to_add
            
            # If buffer is full, create a window
            if self.buffer_filled == self.window_size:
                windows.append(self.buffer.copy())
                
                # Shift buffer by stride (keep the overlap portion)
                overlap_size = self.window_size - self.stride
                self.buffer[:, :overlap_size] = self.buffer[:, self.stride:]
                self.buffer_filled = overlap_size
        
        return windows


######################################################## New Util Feature processing ######################################################################
class FeatureUtils:
    @staticmethod
    def extract_features(signal):
        """
        Extract time-domain features from a signal array
        Args:
            signal (array-like): Input signal array
        Returns:
            dict: Dictionary containing computed features
        """
        # Convert to numpy array if not already
        signal = np.array(signal)
        # Root Mean Square (RMS)
        rms = np.sqrt(np.mean(signal**2))
        # Variance
        variance = np.var(signal)
        # Mean Absolute Value (MAV)
        mav = np.mean(np.abs(signal))
        # Slope Sign Change (SSC)
        diff = np.diff(signal)
        ssc = np.sum((diff[:-1] * diff[1:]) < 0)
        # Zero Crossing Rate (ZCR)
        zcr = np.sum(np.diff(np.signbit(signal).astype(int)) != 0)
        # Waveform Length (WL)
        wl = np.sum(np.abs(np.diff(signal)))
        return {
            'rms': rms,
            'variance': variance,
            'mav': mav,
            'ssc': ssc,
            'zcr': zcr,
            'wl': wl
        }

class ModelProcessor(SignalProcessor):
    def __init__(self, model, window_size=250, overlap=0.5, sampling_rate=1000, 
                 n_predictions=5, aggregate=True):
        """
        Args:
            model: Loaded ML model or path to model file
            window_size: Number of samples per window
            overlap: Overlap ratio between windows (0 to 1)
            sampling_rate: Sampling rate in Hz
            n_predictions: Number of recent predictions to consider for mode
            aggregate: Whether to return the mode of recent predictions
        """
        # Check if model is a string (file path)
        if isinstance(model, str):
            model = self.load_model(model)
            print("Model loaded via path")
        
        self.model = model
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.aggregate = aggregate
        self.n_predictions = n_predictions
        self.prediction_history = []
    
    @staticmethod
    def load_model(model_path):
        import os
        import pickle
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} not found, check path again")
        
        try:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
        
    def process(self, window: np.ndarray) -> np.ndarray:
        """Process a single window of EMG data and return prediction with smoothing"""
        features = []
        
        # Extract features from each channel
        for channel in window:
            features.extend(list(FeatureUtils.extract_features(channel).values()))
            
        # Make prediction
        pred = self.model.predict(np.array(features).reshape(1, -1))[0]
        
        # Add to prediction history
        self.prediction_history.append(pred)
        
        # Keep only the most recent predictions
        if len(self.prediction_history) > self.n_predictions:
            self.prediction_history.pop(0)
        
        # Return individual prediction or mode of recent predictions
        if self.aggregate and len(self.prediction_history) > 0:
            return mode(self.prediction_history)
        else:
            return pred

# class ModelProcessor(SignalProcessor):
#     def __init__(self, model, window_size=250, overlap=0.5, sampling_rate=1000, aggregate=True):
#         """
#         Args:
#             model: Loaded ML model or path to model file
#             window_size: Number of samples per window
#             overlap: Overlap ratio between windows (0 to 1)
#             sampling_rate: Sampling rate in Hz
#         """
#         # Check if model is a string (file path)
#         if isinstance(model, str):
#             model = self.load_model(model)
#             print("Model loaded via path")
        
#         self.model = model
#         self.window_size = window_size
#         self.stride = int(window_size * (1 - overlap))
#         self.sampling_rate = sampling_rate
#         self.aggregate = aggregate
    
#     @staticmethod
#     def load_model(model_path):
#         import os
#         import pickle
        
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model path {model_path} not found, check path again")
        
#         try:
#             with open(model_path, 'rb') as file:
#                 model = pickle.load(file)
#             return model
#         except Exception as e:
#             raise Exception(f"Error loading model: {str(e)}")
        
         
#     def process(self, data: np.ndarray) -> np.ndarray:
#         """Process EMG data with overlapping windows and return predictions"""
#         n_channels, n_samples = data.shape
#         predictions = []
        
#         # Process overlapping windows
#         for i in range(0, n_samples - self.window_size + 1, self.stride):
#             window = data[:, i:i+self.window_size]
#             features = []
            
#             # Extract features from each channel
#             for channel in window:
#                 features.extend(list(FeatureUtils.extract_features(channel).values()))
                
#             # Make prediction
#             pred = self.model.predict(np.array(features).reshape(1, -1))
#             predictions.append(pred[0])
#             print(f'Prediction of window {i}: {pred}')
        
#         if self.aggregate:
#             return mode(np.array(predictions))
#         else:
#             return np.array(predictions)
        

class IntensityProcessor:
    """Processes EMG signal windows and calculates intensity based on extracted features"""
    def __init__(self, scaling_factor=1.5):
        self.max_rms = None
        self.scaling_factor = scaling_factor
    
    def process(self, window: np.ndarray) -> dict:
        """
        Process EMG window and calculate intensity metrics
        
        Args:
            window: EMG data array of shape (channels, samples)
            
        Returns:
            Dictionary with intensity metrics
        """
        feature_values = []
        
        # Extract features from each channel
        for channel in window:
            features = FeatureUtils.extract_features(channel)
            feature_values.append(features)
        
        # Get RMS values from all channels
        rms_values = [features['rms'] for features in feature_values]
        current_max_rms = max(rms_values)
        
        # Initialize max_rms if this is first window
        if self.max_rms is None:
            self.max_rms = current_max_rms * self.scaling_factor
        
        # Update max RMS if we see a higher value
        if current_max_rms > self.max_rms:
            self.max_rms = current_max_rms
        
        # Calculate average RMS and normalize
        avg_rms = np.mean(rms_values)
        # normalized_rms = avg_rms / self.max_rms
        
        # Get MAV values from all channels
        # mav_values = [features['mav'] for features in feature_values]
        # avg_mav = np.mean(mav_values)
        
        return {
            'feature_values': feature_values,  # All extracted features
            'rms_values': rms_values,          # RMS for each channel
            'max_rms_ever': self.max_rms,      # Historical maximum RMS
            # 'normalized_rms': normalized_rms,  # Normalized average RMS
            'avg_rms': avg_rms,                # Average RMS across channels
            # 'avg_mav': avg_mav,                # Average MAV across channels
            'max_channel': np.argmax(rms_values)  # Most active channel
        }