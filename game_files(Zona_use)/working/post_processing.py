# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from statistics import mode
import numpy as np
from processors import SignalProcessor
import pickle
from typing import List
import warnings

# Suppress the warning about feature names
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, 
                           module='sklearn.utils.validation')

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
                 n_predictions=5, aggregate=True, label_encoder = None):
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
        
        if label_encoder:
            self.label_encoder = label_encoder
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

        if self.label_encoder is not None:
            pred_index = np.argmax(pred)
            pred = self.label_encoder.inverse_transform([pred_index])[0]

        # Add to prediction history
        self.prediction_history.append(pred)
        if len(self.prediction_history) > self.n_predictions:
            self.prediction_history.pop(0)
        
        # Return individual prediction or the mode of recent predictions
        if self.aggregate and self.prediction_history:
            return mode(self.prediction_history)
        else:
            return pred

######################################################## LGBM with basic ######################################################################

class LGBMProcessor(SignalProcessor):
    def __init__(self, models, window_size=250, overlap=0.5, sampling_rate=1000, 
                 n_predictions=5, aggregate=True, debug=False):
        """
        Args:
            models: Trained LGBM models list for ensemble prediction
            window_size: Number of samples per window
            overlap: Overlap ratio between windows (0 to 1)
            sampling_rate: Sampling rate in Hz
            n_predictions: Number of recent predictions to consider for mode
            aggregate: Whether to return the mode of recent predictions
            debug: Enable verbose logging for debugging
        """
        self.models = models if isinstance(models, list) else [models]
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.aggregate = aggregate
        self.n_predictions = n_predictions
        self.prediction_history = []
        self.debug = debug
        self.latest_probabilities = None
        
        # Feature extractors - matching training pipeline
        self.feature_extractor = FeatureUtils()
        
        # Get expected feature order from first model (if possible)
        self.feature_names = self._get_feature_names()
        
        if self.debug:
            print(f"Initialized LGBMProcessor with {len(self.models)} models")
            if self.feature_names:
                print(f"Expecting {len(self.feature_names)} features in order")
        
    def _get_feature_names(self):
        """Try to get feature names from first model if available"""
        try:
            if hasattr(self.models[0], 'feature_name_'):
                return self.models[0].feature_name_
        except:
            pass
        return None
        
    def extract_features(self, window):
        """
        Extract features from a window of EMG data using exact column naming from training
        """
        # Ensure window is oriented as [channels, samples]
        if len(window.shape) == 1:
            window = window.reshape(1, -1)
        
        if window.shape[0] > window.shape[1]:
            window = window.T
            
        num_channels = window.shape[0]
        
        if self.debug:
            print(f"Extracting features from window with {num_channels} channels, shape: {window.shape}")
        
        # Define feature types in the exact order used in training
        feature_types = ['rms', 'variance', 'mav', 'ssc', 'zcr', 'wl']
        
        # Build feature dictionary with EXACT naming convention from training
        features_dict = {}
        
        for channel_idx in range(num_channels):
            channel_data = window[channel_idx]
            
            # Extract features for this channel
            channel_features = self.feature_extractor.extract_features(channel_data)
            
            # Use exact naming format from your training data
            # Format is "{channel_number}_{feature_type}" without "ch" prefix
            for feat_type, value in channel_features.items():
                col_name = f"{channel_idx+1}_{feat_type}"  # Changed from "ch{channel_idx+1}_"
                features_dict[col_name] = value
                
        if self.debug:
            print(f"Extracted {len(features_dict)} features")
            
        # If we have feature names from the model, ensure exact order
        if self.feature_names:
            ordered_features = []
            missing_features = []
            
            for feature in self.feature_names:
                if feature in features_dict:
                    ordered_features.append(features_dict[feature])
                else:
                    missing_features.append(feature)
                    # Use 0 as default for missing features
                    ordered_features.append(0)
                    
            if missing_features and self.debug:
                print(f"Warning: Missing {len(missing_features)} features: {missing_features[:5]}...")
                
            return ordered_features
        else:
            # Without feature names, return dictionary and hope order is preserved
            return list(features_dict.values())
            

    def bagged_predict(self, features):
        """Make ensemble prediction using all models"""
        # Reshape features to 2D if needed
        if len(np.array(features).shape) == 1:
            features_array = np.array(features).reshape(1, -1)
        else:
            features_array = np.array(features)
            
        # Get predictions from all models
        all_preds = []
        all_probs = []
        
        # Suppress the feature names warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, 
                                module='sklearn.utils.validation')
            
            for model in self.models:
                try:
                    preds = model.predict(features_array)
                    all_preds.append(preds[0])
                    
                    # Try to get probabilities if available
                    try:
                        probs = model.predict_proba(features_array)[0]
                        all_probs.append(probs)
                    except:
                        pass
                except Exception as e:
                    if self.debug:
                        print(f"Prediction error: {str(e)}")
                    
        # Store probabilities for debugging
        if all_probs:
            # Average the probabilities from all models
            self.latest_probabilities = np.mean(all_probs, axis=0)
            
        # Return most common prediction (mode)
        from collections import Counter
        prediction = Counter(all_preds).most_common(1)[0][0]
        return prediction
        
    def process(self, window):
        """
        Process window to make prediction
        
        Args:
            window: EMG data window
            
        Returns:
            Prediction label
        """
        try:
            # Extract features
            features = self.extract_features(window)
            
            # Make prediction
            prediction = self.bagged_predict(features)
            
            # Update prediction history
            self.prediction_history.append(prediction)
            if len(self.prediction_history) > self.n_predictions:
                self.prediction_history.pop(0)
                
            # Return mode of recent predictions if aggregating
            if self.aggregate and len(self.prediction_history) > 0:
                from collections import Counter
                return Counter(self.prediction_history).most_common(1)[0][0]
                
            return prediction
        except Exception as e:
            if self.debug:
                print(f"Processing error: {str(e)}")
            return None

######################################################## OLD STUFF ######################################################################
# class LGBMProcessor(SignalProcessor):
#     def __init__(self, models, window_size=250, overlap=0.5, sampling_rate=1000, 
#                  n_predictions=5, aggregate=True):
#         """
#         Args:
#             models: Loaded ML model or path to model file
#             window_size: Number of samples per window
#             overlap: Overlap ratio between windows (0 to 1)
#             sampling_rate: Sampling rate in Hz
#             n_predictions: Number of recent predictions to consider for mode
#             aggregate: Whether to return the mode of recent predictions
#         """
#         # Check if model is a string (file path)
#         if isinstance(models, str):
#             model_dict = self.load_model(models)
#             models = model_dict['models']
#             print("Model loaded via path")
        
#         self.models = models
#         self.window_size = window_size
#         self.sampling_rate = sampling_rate
#         self.aggregate = aggregate
#         self.n_predictions = n_predictions
#         self.prediction_history = []
        
#         # Only use basic features extractor
#         self.basic_extractor = FeatureUtils()
        
#         # For debugging
#         self.debug = False
    
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
        
#     def extract_features(self, window: np.ndarray) -> list:
#         """
#         Extract features from a window of EMG data,
#         ensuring feature order matches training data exactly.
        
#         Args:
#             window: EMG data with shape [channels, samples]
            
#         Returns:
#             List of feature values in consistent order matching training
#         """
#         # Ensure window is oriented as [channels, samples]
#         if len(window.shape) == 1:
#             # Single channel data
#             window = window.reshape(1, -1)
        
#         if window.shape[0] > window.shape[1]:  # More rows than columns
#             if self.debug:
#                 print("WARNING: Window appears to be transposed (more rows than columns)")
#                 print(f"Original shape: {window.shape}, transposing...")
#             window = window.T
#             if self.debug:
#                 print(f"New shape: {window.shape}")
        
#         num_channels = window.shape[0]
        
#         if self.debug:
#             print(f"Processing window with shape: {window.shape}, {num_channels} channels")
        
#         # Define the expected order of feature types based on your dataset
#         feature_types = ['rms', 'variance', 'mav', 'ssc', 'zcr', 'wl']
        
#         features = []
        
#         # Extract features by channel, then by feature type
#         for channel_idx in range(num_channels):
#             channel_data = window[channel_idx]
            
#             if self.debug:
#                 print(f"Processing channel {channel_idx+1} basic features")
            
#             # Extract basic features
#             basic_features = self.basic_extractor.extract_features(channel_data)
            
#             # Add features in the specific order that matches training
#             for feature_type in feature_types:
#                 if feature_type in basic_features:
#                     features.append(basic_features[feature_type])
#                 elif self.debug:
#                     print(f"Warning: Feature {feature_type} not found for channel {channel_idx+1}")
        
#         if self.debug:
#             print(f"Extracted {len(features)} features in channel-first order")
#             # Print expected feature names in the order they were added
#             expected_names = [f"{c+1}_{f}" for c in range(num_channels) for f in feature_types]
#             print(f"Expected feature order: {expected_names[:10]}...")
        
#         return features
    

#     def process(self, window: np.ndarray, debug: bool = False) -> np.ndarray:
#         """
#         Process a single window of EMG data and return prediction with smoothing.
#         """
#         self.debug = debug
        
#         # Extract features
#         features = self.extract_features(window)
        
#         # Store features for later inspection
#         self.last_features = features
        
#         # Print debug info if requested
#         if debug:
#             print(f"Extracted {len(features)} features")
#             print(f"First few features: {features[:3]}")
#             print(f"Last few features: {features[-3:]}")
        
#         # Make prediction with ensemble
#         pred = self.predict_bagged(np.array(features).reshape(1, -1))[0]
        
#         if debug:
#             print(f"Raw prediction: {pred}")

#         # Add to prediction history
#         self.prediction_history.append(pred)
#         if len(self.prediction_history) > self.n_predictions:
#             self.prediction_history.pop(0)
        
#         # Return individual prediction or the mode of recent predictions
#         if self.aggregate and len(self.prediction_history) > 0:
#             from collections import Counter
#             most_common = Counter(self.prediction_history).most_common(1)[0][0]
            
#             if debug:
#                 print(f"Prediction history: {self.prediction_history}")
#                 print(f"Aggregated prediction: {most_common}")
            
#             return most_common
#         else:
#             return pred
        
#     def predict_bagged(self, X):
#         """
#         Make predictions using an ensemble of models.
#         """
#         from collections import Counter
        
#         # Get predictions from all models
#         model_predictions = [model.predict(X) for model in self.models]
        
#         if self.debug:
#             print(f"Individual model predictions: {model_predictions}")
        
#         # Transpose the predictions
#         preds = np.array(model_predictions).T
        
#         # Get the most common prediction for each sample
#         result = np.array([Counter(row).most_common(1)[0][0] for row in preds])
        
#         return result

######################################################## LGBM ######################################################################
# class LGBMProcessor(SignalProcessor):
#     def __init__(self, models, window_size=250, overlap=0.5, sampling_rate=1000, 
#                  n_predictions=5, aggregate=True):
#         """
#         Args:
#             model: Loaded ML model or path to model file
#             window_size: Number of samples per window
#             overlap: Overlap ratio between windows (0 to 1)
#             sampling_rate: Sampling rate in Hz
#             n_predictions: Number of recent predictions to consider for mode
#             aggregate: Whether to return the mode of recent predictions
#         """
#         # Check if model is a string (file path)
#         if isinstance(models, str):
#             model_dict = self.load_model(models)
#             models = model_dict['models']
#             print("Model loaded via path")
        
#         self.models = models
#         self.window_size = window_size
#         self.sampling_rate = sampling_rate
#         self.aggregate = aggregate
#         self.n_predictions = n_predictions
#         self.prediction_history = []
#         self.wavelet_extractors = [
#             WaveletFeatureExtractor(wavelet='sym4', levels=2),
#             # WaveletFeatureExtractor(wavelet='sym5', levels=2),
#             # WaveletFeatureExtractor(wavelet='db4', levels=2)
#             ]
#         self.basic_extractor = FeatureUtils()
# class LGBMProcessor(SignalProcessor):
#     def __init__(self, models, window_size=250, overlap=0.5, sampling_rate=1000, 
#                  n_predictions=5, aggregate=True):
#         """
#         Args:
#             models: List of trained models or path to model file
#             window_size: Number of samples per window
#             overlap: Overlap ratio between windows (0 to 1)
#             sampling_rate: Sampling rate in Hz
#             n_predictions: Number of recent predictions to consider for mode
#             aggregate: Whether to return the mode of recent predictions
#         """
#         # Check if model is a string (file path)
#         if isinstance(models, str):
#             model_dict = self.load_model(models)
#             models = model_dict['models']
#             print("Model loaded via path")
        
#         self.models = models
#         self.window_size = window_size
#         self.sampling_rate = sampling_rate
#         self.aggregate = aggregate
#         self.n_predictions = n_predictions
#         self.prediction_history = []
        
#         # Initialize extractors (only using sym4 as in training)
#         # self.wavelet_extractor = WaveletFeatureExtractor(wavelet='sym4', levels=2)
#         self.wavelet_extractor = None
#         self.basic_extractor = FeatureUtils()
        
#         # Enable debug mode for detailed logging
#         self.debug = False
    
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
    
#     def extract_features(self, window: np.ndarray) -> list:
#         """
#         Extract features in the exact same order as used during training.
        
#         This matches the process in dataset_prep where:
#         1. First all features for sym4 wavelet are extracted for each channel
#         2. Then all basic features are extracted for each channel
#         """
#         all_features = []
        
#         num_channels = window.shape[1] if len(window.shape) > 1 else 1
        
#         if self.debug:
#             print(f"Processing window with shape: {window.shape}, detected {num_channels} channels")
        
#         # First extract wavelet features for all channels
#         if self.wavelet_extractor:
#             for channel_idx in range(num_channels):
#                 channel_data = window[:, channel_idx] if len(window.shape) > 1 else window
                
#                 # Extract wavelet features (sym4)
#                 wavelet_features = self.wavelet_extractor.extract_features(channel_data)
                
#                 if self.debug:
#                     print(f"Channel {channel_idx+1} wavelet features: {list(wavelet_features.keys())}")
                
#                 # Add each feature value to the list
#                 for feature_value in wavelet_features.values():
#                     all_features.append(feature_value)
        
#         # Then extract basic features for all channels
#         for channel_idx in range(num_channels):
#             channel_data = window[:, channel_idx] if len(window.shape) > 1 else window
            
#             # Extract basic features
#             basic_features = self.basic_extractor.extract_features(channel_data)
            
#             if self.debug:
#                 print(f"Channel {channel_idx+1} basic features: {list(basic_features.keys())}")
            
#             # Add each feature value to the list
#             for feature_value in basic_features.values():
#                 all_features.append(feature_value)
        
#         if self.debug:
#             print(f"Total features extracted: {len(all_features)}")
        
#         return all_features

#     def process(self, window: np.ndarray, debug: bool = False) -> np.ndarray:
#         """
#         Process a single window of EMG data and return prediction with smoothing.
#         """
#         self.debug = debug
        
#         # Extract features
#         features = self.extract_features(window)
#         # Store features for later inspection if needed
#         self.last_features = features
        
#         # Print features if in debug mode
#         if debug:
#             print(f"Extracted {len(features)} features")
#             print(f"First 3 features: {features[:3]}")
#             print(f"Last 3 features: {features[-3:]}")
        
#         # Make prediction with the ensemble
#         pred = self.predict_bagged(np.array(features).reshape(1, -1))[0]
        
#         if debug:
#             print(f"Raw prediction: {pred}")

#         # Add to prediction history
#         self.prediction_history.append(pred)
#         if len(self.prediction_history) > self.n_predictions:
#             self.prediction_history.pop(0)
        
#         # Return individual prediction or the mode of recent predictions
#         if self.aggregate and len(self.prediction_history) > 0:
#             from collections import Counter
#             most_common = Counter(self.prediction_history).most_common(1)[0][0]
            
#             if debug:
#                 print(f"Prediction history: {self.prediction_history}")
#                 print(f"Aggregated prediction: {most_common}")
            
#             return most_common
#         else:
#             return pred
        
#     def predict_bagged(self, X):
#         """
#         Make predictions using an ensemble of models.
#         """
#         from collections import Counter
        
#         # Get predictions from all models
#         model_predictions = [model.predict(X) for model in self.models]
        
#         if self.debug:
#             print(f"Individual model predictions: {model_predictions}")
        
#         # Transpose predictions for each sample
#         preds = np.array(model_predictions).T
        
#         # Get most common prediction for each sample
#         result = np.array([Counter(row).most_common(1)[0][0] for row in preds])
        
#         if self.debug:
#             print(f"Final bagged prediction: {result}")
        
#         return result


# class LGBMProcessor(SignalProcessor):
#     def __init__(self, models, window_size=250, overlap=0.5, sampling_rate=1000, 
#                  n_predictions=5, aggregate=True):
#         """
#         Args:
#             model: Loaded ML model or path to model file
#             window_size: Number of samples per window
#             overlap: Overlap ratio between windows (0 to 1)
#             sampling_rate: Sampling rate in Hz
#             n_predictions: Number of recent predictions to consider for mode
#             aggregate: Whether to return the mode of recent predictions
#         """
#         # Check if model is a string (file path)
#         if isinstance(models, str):
#             model_dict = self.load_model(models)
#             models = model_dict['models']
#             print("Model loaded via path")
        
#         self.models = models
#         self.window_size = window_size
#         self.sampling_rate = sampling_rate
#         self.aggregate = aggregate
#         self.n_predictions = n_predictions
#         self.prediction_history = []
        
#         # Use the same wavelet types as in your training script
#         self.wavelet_types = ['sym4']  # Make sure this matches your training
#         self.wavelet_extractors = {
#             wavelet: WaveletFeatureExtractor(wavelet=wavelet, levels=2)
#             for wavelet in self.wavelet_types
#         }
#         self.basic_extractor = FeatureUtils()

    
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
    

#     # def extract_features(self, window: np.ndarray) -> list:
#     #     """
#     #     Extract features from each channel in the window.
#     #     This function encapsulates the feature extraction logic so that you
#     #     can easily inspect or debug the features.
#     #     """
#     #     features = []
#     #     # Loop through each channel and extract features using different methods.
#     #     for channel in window:
#     #         features.extend(list(WaveletFeatureExtractor(wavelet='sym4', levels=2).extract_features(channel).values()))
#     #         features.extend(list(WaveletFeatureExtractor(wavelet='sym5',levels=2).extract_features(channel).values()))
#     #         features.extend(list(WaveletFeatureExtractor(wavelet='db4',levels=2).extract_features(channel).values()))
#     #         features.extend(list(FeatureUtils.extract_features(channel).values()))
#     #     return features
#     # def extract_features(self, window: np.ndarray) -> list:
#     #     features = []
#     #     for channel in window:
#     #         # Use the pre-created extractors
#     #         for extractor in self.wavelet_extractors:
#     #             features.extend(list(extractor.extract_features(channel).values()))
#     #         features.extend(list(self.basic_extractor.extract_features(channel).values()))
#     #     return features

#     def extract_features(self, window: np.ndarray) -> list:
#         """
#         Extract features in the exact same order as used during training.
#         This implementation mimics the dataset preparation process to ensure
#         feature order consistency.
#         """
#         # First extract all wavelet features for each channel
#         all_wavelet_features = []
#         for wavelet_type in self.wavelet_types:
#             wavelet_extractor = self.wavelet_extractors[wavelet_type]
#             for channel_idx, channel in enumerate(window):
#                 # Extract wavelet features
#                 channel_wavelet_features = wavelet_extractor.extract_features(channel)
#                 # Add each feature to the list (would be renamed in training)
#                 for feature_name, feature_value in channel_wavelet_features.items():
#                     all_wavelet_features.append(feature_value)
        
#         # Then extract all basic features for each channel
#         all_basic_features = []
#         for channel_idx, channel in enumerate(window):
#             # Extract basic features
#             channel_basic_features = self.basic_extractor.extract_features(channel)
#             # Add each feature to the list
#             for feature_name, feature_value in channel_basic_features.items():
#                 all_basic_features.append(feature_value)
        
#         # Combine all features in the same order as training
#         all_features = all_wavelet_features + all_basic_features
        
#         # Debug information - print feature count to verify it matches training
#         expected_count = len(window) * (len(self.wavelet_types) * 6 * 3 + 6)  # Adjust based on your feature counts
#         if len(all_features) != expected_count:
#             print(f"WARNING: Feature count mismatch. Expected {expected_count}, got {len(all_features)}")
        
#         return all_features

#     def process(self, window: np.ndarray, debug: bool = False) -> np.ndarray:
#         """
#         Process a single window of EMG data and return prediction with smoothing.
#         Optionally, print out the features for debugging if debug=True.
#         """
#         # Extract features
#         features = self.extract_features(window)
#         # Store features for later inspection if needed
#         self.last_features = features
        
#         # Print features if in debug mode
#         if debug:
#             print("Extracted features:", features)
        
#         # Make prediction
#         pred = self.predict_bagged(np.array(features).reshape(1, -1))[0]

#         # Add to prediction history
#         self.prediction_history.append(pred)
#         if len(self.prediction_history) > self.n_predictions:
#             self.prediction_history.pop(0)
        
#         # Return individual prediction or the mode of recent predictions
#         if self.aggregate and self.prediction_history:
#             return mode(self.prediction_history)
#         else:
#             return pred
        
#     def predict_bagged(self, X):
#         models = self.models
#         model_predictions = [model.predict(X) for model in models]
#         print(f"Individual model predictions: {model_predictions}")
        
#         preds = np.array(model_predictions).T
#         print(f"Transposed predictions shape: {preds.shape}")
        
#         result = np.array([Counter(row).most_common(1)[0][0] for row in preds])
#         print(f"Final aggregated prediction: {result}")
        
#         return result
        
#     # def predict_bagged(self, X):
#     #     models = self.models
#     #     all_preds = [model.predict(X) for model in models]
#     #     return np.array([
#     #         Counter(col).most_common(1)[0][0] for col in zip(*all_preds)
#     #     ])
    
#     # def predict_bagged(self, X):
#     #     models = self.models
#     #     all_preds = [model.predict(X) for model in models]
        
#     #     # Initialize latest_probabilities attribute
#     #     self.latest_probabilities = None
        
#     #     # Try to get probabilities if available
#     #     try:
#     #         if hasattr(models[0], 'predict_proba'):
#     #             # Average probabilities from all models
#     #             all_probs = [model.predict_proba(X)[0] for model in models]
#     #             self.latest_probabilities = np.mean(all_probs, axis=0)
#     #     except Exception as e:
#     #         # Silently fail if probabilities aren't available
#     #         pass
        
#     #     return np.array([
#     #         Counter(col).most_common(1)[0][0] for col in zip(*all_preds)
#     #     ])
    
#     def process_with_metadata(self, window: np.ndarray, debug: bool = False) -> dict:
#         """
#         Process a window and return a rich prediction object with metadata.
#         """
#         # Extract features
#         features = self.extract_features(window)
#         self.last_features = features
        
#         if debug:
#             print("Extracted features:", features)
        
#         # Make prediction using your bagged prediction method
#         X = np.array(features).reshape(1, -1)
#         pred = self.predict_bagged(X)[0]
        
#         # Add to prediction history (keep this consistent with your original method)
#         self.prediction_history.append(pred)
#         if len(self.prediction_history) > self.n_predictions:
#             self.prediction_history.pop(0)
        
#         # If aggregation is enabled, get the most common prediction
#         if self.aggregate and self.prediction_history:
#             final_prediction = mode(self.prediction_history)
#         else:
#             final_prediction = pred
        
#         # Create result dictionary with all metadata
#         result = {
#             'label': final_prediction,
#             'raw_prediction': pred,
#             'prediction_history': self.prediction_history.copy(),
#             'confidence': 0.0,
#             'probabilities': {}
#         }
        
#         # Add probabilities if available
#         if hasattr(self, 'latest_probabilities') and self.latest_probabilities is not None:
#             classes = [str(i) for i in range(len(self.latest_probabilities))]
#             result['probabilities'] = {c: p for c, p in zip(classes, self.latest_probabilities)}
#             result['confidence'] = max(self.latest_probabilities)
        
#         return result

class IntensityProcessor:
    """
    Processes EMG signal windows and calculates intensity based on extracted features
    with improved stability across channels
    """
    def __init__(self, scaling_factor=1.5, smoothing_factor=0.8):
        self.max_rms_per_channel = None  # Track max RMS separately for each channel
        self.current_active_channels = None  # Track which channels are active
        self.smoothed_rms = None  # For smoothing RMS values over time
        self.scaling_factor = scaling_factor
        self.smoothing_factor = smoothing_factor  # Higher = more smoothing (0-1)
        
    def process(self, window: np.ndarray) -> dict:
        """
        Process EMG window and calculate intensity metrics with channel tracking
        
        Args:
            window: EMG data array of shape (channels, samples)
        
        Returns:
            Dictionary with intensity metrics
        """
        num_channels = len(window)
        feature_values = []
        
        # Initialize channel tracking if this is the first window
        if self.max_rms_per_channel is None:
            self.max_rms_per_channel = np.zeros(num_channels)
            self.smoothed_rms = np.zeros(num_channels)
        
        # Extract features from each channel
        for channel_idx, channel in enumerate(window):
            features = FeatureUtils.extract_features(channel)
            feature_values.append(features)
        
        # Get RMS values from all channels
        rms_values = np.array([features['rms'] for features in feature_values])
        
        # Apply temporal smoothing to RMS values
        if self.smoothed_rms is not None:
            self.smoothed_rms = (self.smoothing_factor * self.smoothed_rms + 
                                (1 - self.smoothing_factor) * rms_values)
        else:
            self.smoothed_rms = rms_values.copy()
        
        # Update max RMS for each channel separately
        for i, rms in enumerate(rms_values):
            if rms > self.max_rms_per_channel[i]:
                self.max_rms_per_channel[i] = rms * self.scaling_factor
        
        # Find active channels (those with significant activity)
        active_threshold = np.mean(rms_values) * 0.5  # Threshold for considering a channel active
        active_channels = np.where(rms_values > active_threshold)[0]
        
        # If no channels are active, use all channels
        if len(active_channels) == 0:
            active_channels = np.arange(num_channels)
        
        # Calculate metrics using only active channels
        avg_rms = np.mean(rms_values[active_channels])
        max_channel = np.argmax(rms_values)
        
        # Calculate normalized RMS for each channel (with respect to its own historical max)
        normalized_rms_values = np.zeros_like(rms_values)
        for i, rms in enumerate(rms_values):
            if self.max_rms_per_channel[i] > 0:
                normalized_rms_values[i] = rms / self.max_rms_per_channel[i]
        
        # Get overall normalized RMS (using active channels only)
        if len(active_channels) > 0:
            overall_normalized_rms = np.mean(normalized_rms_values[active_channels])
        else:
            overall_normalized_rms = 0
            
        # Get max RMS ever seen across all channels
        max_rms_ever = np.max(self.max_rms_per_channel)
        
        return {
            'feature_values': feature_values,         # All extracted features
            'rms_values': rms_values.tolist(),        # RMS for each channel
            'smoothed_rms': self.smoothed_rms.tolist(), # Smoothed RMS values
            'max_rms_per_channel': self.max_rms_per_channel.tolist(), # Max RMS per channel
            'max_rms_ever': max_rms_ever,             # Max RMS ever seen across all channels
            'avg_rms': avg_rms,                       # Average RMS across active channels
            'normalized_rms_values': normalized_rms_values.tolist(), # Normalized RMS per channel
            'overall_normalized_rms': overall_normalized_rms, # Overall normalized RMS
            'max_channel': int(max_channel),          # Most active channel
            'active_channels': active_channels.tolist() # List of currently active channels
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pickle
from collections import Counter

class BaggedRF:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        all_preds = [model.predict(X) for model in self.models]
        return np.array([
            Counter(col).most_common(1)[0][0] for col in zip(*all_preds)
        ])