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

class WideModelProcessor(SignalProcessor):
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
    
    def extract_features(self, window: np.ndarray) -> list:
        """
        Extract features from each channel in the window.
        This function encapsulates the feature extraction logic so that you
        can easily inspect or debug the features.
        """
        features = []
        # Loop through each channel and extract features using different methods.
        for channel in window:
            features.extend(list(WaveletFeatureExtractor(wavelet='sym4', levels=2).extract_features(channel).values()))
            features.extend(list(WaveletFeatureExtractor(wavelet='sym5',levels=2).extract_features(channel).values()))
            features.extend(list(WaveletFeatureExtractor(wavelet='db4',levels=2).extract_features(channel).values()))
            features.extend(list(FeatureUtils.extract_features(channel).values()))
        return features

    def process(self, window: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Process a single window of EMG data and return prediction with smoothing.
        Optionally, print out the features for debugging if debug=True.
        """
        # Extract features
        features = self.extract_features(window)
        # Store features for later inspection if needed
        self.last_features = features
        
        # Print features if in debug mode
        if debug:
            print("Extracted features:", features)
        
        # Make prediction
        pred = self.model.predict(np.array(features).reshape(1, -1))[0]

        pred = np.argmax(pred)
        
        # Add to prediction history
        self.prediction_history.append(pred)
        if len(self.prediction_history) > self.n_predictions:
            self.prediction_history.pop(0)
        
        # Return individual prediction or the mode of recent predictions
        if self.aggregate and self.prediction_history:
            return mode(self.prediction_history)
        else:
            return pred
    


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
    
######################################################## Wavelet Decomp ######################################################################

import pywt
from typing import Dict, List, Tuple

class WaveletProcessor:
    """Simplified wavelet processor for EMG signal analysis
       Used by Wavelet Feature Extractor to decompose wavelets
    """
    
    def __init__(self, wavelet='sym4', levels=2):
        """
        Initialize wavelet processor
        
        Args:
            wavelet: Wavelet type (default: 'db4')
            levels: Number of decomposition levels (default: 3)
        """
        self.wavelet = wavelet
        self.levels = levels
        
    def decompose(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform wavelet decomposition on signal
        
        Args:
            signal: 1D array of signal values
            
        Returns:
            Dictionary containing coefficients for each level
        """
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.levels)
        
        # Package coefficients with their level names
        results = {f'A{self.levels}': coeffs[0]}  # Final approximation
        for level in range(self.levels):
            results[f'D{level+1}'] = coeffs[level+1]
            
        return results
    
    def reconstruct_level(self, coeffs: Dict[str, np.ndarray], level: str) -> np.ndarray:
        """
        Reconstruct signal from specific decomposition level
        
        Args:
            coeffs: Coefficient dictionary from decompose()
            level: Level to reconstruct (e.g., 'D1', 'A3')
            
        Returns:
            Reconstructed signal for that level
        """
        # Create list of zeros for all levels except the one we want
        coeff_list = []
        for i in range(self.levels + 1):
            if i == 0 and f'A{self.levels}' == level:
                coeff_list.append(coeffs[f'A{self.levels}'])
            elif i > 0 and f'D{i}' == level:
                coeff_list.append(coeffs[f'D{i}'])
            else:
                coeff_list.append(np.zeros_like(
                    coeffs[f'A{self.levels}' if i == 0 else f'D{i}']))
                
        return pywt.waverec(coeff_list, self.wavelet)

######################################################## Feature Extractors ######################################################################

class WaveletFeatureExtractor:
    """Extract features from wavelet decomposed signals"""
    
    def __init__(self, wavelet='db4', levels=3):
        """
        Initialize wavelet feature extractor
        
        Args:
            wavelet: Wavelet type (default: 'db4')
            levels: Decomposition levels (default: 3)
        """
        self.wavelet_processor = WaveletProcessor(wavelet, levels)
        self.levels = levels
        
    def extract_features(self, signal: np.ndarray) -> Dict:
        """
        Extract features from wavelet decomposition of signal
        
        Args:
            signal: 1D array of signal values
            
        Returns:
            Dictionary of features
        """
        # Decompose signal
        coeffs = self.wavelet_processor.decompose(signal)
        
        features = {}
        
        # Extract features from approximation coefficients
        approx_key = f'A{self.levels}'
        approx_signal = self.wavelet_processor.reconstruct_level(coeffs, approx_key)
        approx_features = FeatureUtils.extract_features(approx_signal)
        
        # Add prefix to feature names
        for feature_name, feature_value in approx_features.items():
            features[f'{approx_key}_{feature_name}'] = feature_value
        
        # Extract features from detail coefficients
        for level in range(1, self.levels + 1):
            detail_key = f'D{level}'
            detail_signal = self.wavelet_processor.reconstruct_level(coeffs, detail_key)
            detail_features = FeatureUtils.extract_features(detail_signal)
            
            # Add prefix to feature names
            for feature_name, feature_value in detail_features.items():
                features[f'{detail_key}_{feature_name}'] = feature_value
        
        return features
    

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

