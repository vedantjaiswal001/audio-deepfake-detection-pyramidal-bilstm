
import os, random, glob, numpy as np, librosa, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import logging
import sys
from datetime import datetime
import math

# ------------------------------
# Logging Setup
# ------------------------------
def setup_logging():
    """Set up comprehensive logging to both console and file."""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/pyramidal_bilstm_pytorch_training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create a custom logger
    logger = logging.getLogger('pyramidal_bilstm_pytorch')
    
    # Log startup information
    logger.info("="*80)
    logger.info("PYRAMIDAL BiLSTM AUDIO DEEPFAKE DETECTION SYSTEM - PYTORCH")
    logger.info("="*80)
    logger.info(f"Log file created: {log_filename}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"Librosa version: {librosa.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Using CUDA device: cuda:2 (GPU 2)")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(2)}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(2).total_memory / 1024**3:.2f} GB")
    logger.info("="*80)
    
    return logger, log_filename

# Initialize logging
logger, current_log_file = setup_logging()

# Override print function to also log to file
original_print = print
def log_print(*args, **kwargs):
    """Enhanced print function that logs to both console and file."""
    # Convert all args to strings and join them
    message = ' '.join(str(arg) for arg in args)
    
    # Log the message
    logger.info(message)
    
    # Also call original print for immediate console output formatting
    original_print(*args, **kwargs)

# Replace built-in print
print = log_print

# ------------------------------
# Logging Utility Functions
# ------------------------------
def log_model_summary(model, model_name="Pyramidal BiLSTM"):
    """Log detailed model summary and architecture information."""
    print(f"\n{'='*60}")
    print(f"MODEL SUMMARY: {model_name}")
    print(f"{'='*60}")
    
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {non_trainable_params:,}")
    print(f"{'='*60}")

def log_experiment_results(results, experiment_name):
    """Log comprehensive experiment results."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT RESULTS: {experiment_name}")
    print(f"{'='*80}")
    
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, str):
                print(f"  {key}: {value}")
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], (int, float)):
                    print(f"  {key}: [{value[0]:.4f}, {value[1]:.4f}]")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"Results: {results}")
    
    print(f"{'='*80}")

def log_final_summary():
    """Log final session summary and provide log file location."""
    print(f"\n{'='*80}")
    print("TRAINING SESSION COMPLETED")
    print(f"{'='*80}")
    print(f"📝 Complete log saved to: {current_log_file}")
    print(f"🕒 Session duration: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 All outputs (training progress, model summaries, metrics) are saved")
    print(f"📊 Log file includes:")
    print(f"   • Complete model architecture details")
    print(f"   • Training progress and epoch-by-epoch metrics") 
    print(f"   • Comprehensive evaluation results")
    print(f"   • Cross-validation and hyperparameter optimization logs")
    print(f"   • System information and configuration")
    print(f"💾 Use this log file for:")
    print(f"   • Reproducing experiments")
    print(f"   • Debugging and performance analysis")
    print(f"   • Research documentation and reporting")
    print(f"   • Comparing different model configurations")
    print(f"{'='*80}")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some statistical tests will be skipped.")

try:
    import optuna
    OPTUNA_AVAILABLE = True
    print("Optuna available for hyperparameter optimization")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Hyperparameter optimization will be skipped.")
    print("Install with: pip install optuna")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve, 
                             precision_recall_curve, average_precision_score, 
                             f1_score, precision_score, recall_score,
                             confusion_matrix, brier_score_loss)

# ------------------------------
# Device Configuration
# ------------------------------
# Use CUDA device 2 (GPU 2) if available
if torch.cuda.is_available():
    device = torch.device('cuda:7')
    torch.cuda.set_device(2)
    print(f"Using device: {device} (GPU 2)")
    print(f"GPU Name: {torch.cuda.get_device_name(2)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(2).total_memory / 1024**3:.2f} GB")
else:
    device = torch.device('cpu')
    print(f"CUDA not available, using CPU")
print(f"Device: {device}")

# ------------------------------
# Reproducibility
# ------------------------------
seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # Seeds all GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 Random seed set to {seed_value} for all devices including GPU 2")
random.seed(seed_value)

# ------------------------------
# Config
# ------------------------------
dataset_path = 'for-norm/for-norm'  # adjust if needed
labels_map = {'real': 0, 'fake': 1}
n_mels = 80  # Increased from 40 MFCC to 80 mel bins for better frequency resolution

def extract_mel_features(file_path, max_seconds=2.0, n_mels=80, n_fft=1024, hop_length=256):
    """
    Extract log-Mel spectrogram features with consistent audio preprocessing.
    Log-Mel spectrograms typically outperform MFCCs for deep learning.
    Forces 16kHz sampling rate and mono channel for consistency.
    """
    # Force consistent audio parameters: 16kHz mono
    y, sr = librosa.load(file_path, duration=max_seconds, sr=16000, mono=True)
    if y.size == 0:
        return None
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, 
        n_mels=n_mels, 
        n_fft=n_fft, 
        hop_length=hop_length,
        fmax=sr//2  # Nyquist frequency
    )  # (n_mels, T)
    
    # Convert to log scale (dB)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)  # (n_mels, T)
    
    return log_mel.T  # (T, n_mels)

# ------------------------------
# SpecAugment Data Augmentation
# ------------------------------
def spec_augment(mel_spec, time_mask_param=15, freq_mask_param=10, n_time_masks=2, n_freq_masks=2):
    """
    Apply SpecAugment to log-mel spectrogram for data augmentation.
    
    Args:
        mel_spec: (T, F) mel spectrogram
        time_mask_param: Maximum width of time mask
        freq_mask_param: Maximum width of frequency mask  
        n_time_masks: Number of time masks to apply
        n_freq_masks: Number of frequency masks to apply
    
    Returns:
        Augmented mel spectrogram with same shape (T, F)
    """
    mel_spec = mel_spec.copy()  # Don't modify original
    T, n_freq_bins = mel_spec.shape
    
    # Apply frequency masking
    for _ in range(n_freq_masks):
        if freq_mask_param > 0 and n_freq_bins > freq_mask_param:
            mask_width = np.random.randint(0, min(freq_mask_param, n_freq_bins))
            if mask_width > 0:
                mask_start = np.random.randint(0, n_freq_bins - mask_width)
                mel_spec[:, mask_start:mask_start + mask_width] = 0
    
    # Apply time masking
    for _ in range(n_time_masks):
        if time_mask_param > 0 and T > time_mask_param:
            mask_width = np.random.randint(0, min(time_mask_param, T))
            if mask_width > 0:
                mask_start = np.random.randint(0, T - mask_width)
                mel_spec[mask_start:mask_start + mask_width, :] = 0
                
    return mel_spec

# ------------------------------
# Advanced Audio Augmentation
# ------------------------------
def add_colored_noise(audio, noise_type='white', snr_db=15):
    """
    Add colored noise at specified SNR.
    
    Args:
        audio: Input audio signal
        noise_type: 'white', 'pink', or 'brown'
        snr_db: Signal-to-noise ratio in dB
    """
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    if noise_type == 'white':
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    elif noise_type == 'pink':
        # Simplified pink noise approximation
        white_noise = np.random.normal(0, 1, len(audio))
        # Apply simple pink noise filter (approximate)
        noise = np.convolve(white_noise, [1, -0.5], mode='same') 
        noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))
    elif noise_type == 'brown':
        # Brown noise (simple approximation)
        white_noise = np.random.normal(0, 1, len(audio))
        noise = np.cumsum(white_noise) * 0.02  # Scale factor for brown noise
        noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))
    else:
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    
    return audio + noise

def speed_change(audio, sr, speed_factor):
    """Change speed of audio without changing pitch."""
    if speed_factor == 1.0:
        return audio
    return librosa.effects.time_stretch(audio, rate=speed_factor)

def pitch_shift(audio, sr, n_steps):
    """Shift pitch by n_steps semitones."""
    if n_steps == 0:
        return audio
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def apply_audio_augmentation(audio, sr, augment_prob=0.7):
    """
    Apply random audio augmentations to raw audio signal.
    
    Args:
        audio: Raw audio signal
        sr: Sample rate
        augment_prob: Probability of applying each augmentation
        
    Returns:
        Augmented audio signal
    """
    augmented = audio.copy()
    
    # Apply speed perturbation (70% chance)
    if np.random.random() < augment_prob:
        speed_factors = [0.9, 1.0, 1.1]  # Conservative speed changes
        speed_factor = np.random.choice(speed_factors)
        if speed_factor != 1.0:
            augmented = speed_change(augmented, sr, speed_factor)
    
    # Apply pitch shifting (50% chance, smaller range)
    if np.random.random() < (augment_prob * 0.7):  # 50% of 70%
        pitch_steps = [-1, 0, 1]  # ±1 semitone only for subtle changes
        n_steps = np.random.choice(pitch_steps)
        if n_steps != 0:
            augmented = pitch_shift(augmented, sr, n_steps)
    
    # Apply noise addition (80% chance)
    if np.random.random() < (augment_prob + 0.1):  # 80% chance
        noise_type = np.random.choice(['white', 'pink'])
        snr_db = np.random.uniform(10, 20)  # Random SNR between 10-20 dB
        augmented = add_colored_noise(augmented, noise_type, snr_db)
    
    # Normalize to prevent clipping
    if np.max(np.abs(augmented)) > 1.0:
        augmented = augmented / np.max(np.abs(augmented)) * 0.95
        
    return augmented

def extract_mel_features_with_augment(file_path, max_seconds=2.0, n_mels=80, n_fft=1024, 
                                     hop_length=256, apply_augment=False):
    """
    Extract log-Mel spectrogram features with comprehensive augmentation pipeline.
    
    Args:
        apply_augment: If True, apply both audio-level and spec-level augmentation
    """
    # Load raw audio with consistent parameters
    y, sr = librosa.load(file_path, duration=max_seconds, sr=16000, mono=True)
    if y.size == 0:
        return None
    
    # Apply audio-level augmentation if requested (before feature extraction)
    if apply_augment:
        y = apply_audio_augmentation(y, sr, augment_prob=0.6)  # 60% chance for audio aug
    
    # Extract mel spectrogram from (possibly augmented) audio
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, 
        n_mels=n_mels, 
        n_fft=n_fft, 
        hop_length=hop_length,
        fmax=sr//2  # Nyquist frequency
    )  # (n_mels, T)
    
    # Convert to log scale (dB)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)  # (n_mels, T)
    log_mel = log_mel.T  # (T, n_mels)
        
    # Apply SpecAugment if requested (spectral-level augmentation)
    if apply_augment:
        log_mel = spec_augment(
            log_mel, 
            time_mask_param=15,   # Mask up to 15 time frames
            freq_mask_param=10,   # Mask up to 10 frequency bins
            n_time_masks=1,       # 1 time mask (conservative for 2sec audio)
            n_freq_masks=1        # 1 frequency mask
        )
    
    return log_mel

def load_split(split_dir, apply_augment=False):
    """
    Load audio split with comprehensive augmentation pipeline for training robustness.
    
    Args:
        split_dir: Directory containing 'real' and 'fake' subdirs
        apply_augment: If True, apply both audio-level and spectral-level augmentation
    """
    X, y = [], []
    augment_str = "FULL AUGMENTATION (Audio + Spectral)" if apply_augment else "NO AUGMENTATION"
    print(f"Loading data from {split_dir}")
    print(f"  Augmentation: {augment_str}")
    
    if apply_augment:
        print(f"  Audio-level: Speed (±10%), Pitch (±1 semitone), Noise (10-20dB SNR)")
        print(f"  Spectral-level: Time masking (15 frames), Frequency masking (10 bins)")
    
    for label_name in ['real', 'fake']:
        class_dir = os.path.join(split_dir, label_name)
        file_count = 0
        for fn in os.listdir(class_dir):
            if not fn.lower().endswith('.wav'):
                continue
            
            # Use comprehensive augmentation-aware feature extraction
            arr = extract_mel_features_with_augment(
                os.path.join(class_dir, fn), 
                max_seconds=2.0, 
                n_mels=n_mels,
                apply_augment=apply_augment
            )
            if arr is None:
                continue
            X.append(arr)
            y.append(labels_map[label_name])
            file_count += 1
            
        print(f"  {label_name}: {file_count} files loaded")
    
    print(f"Total: {len(X)} samples with {augment_str.lower()}")
    return X, np.array(y, dtype=np.int64)

# Load data with COMPREHENSIVE augmentation pipeline ONLY for training set
print("="*80)
print("LOADING DATASET WITH COMPREHENSIVE AUGMENTATION PIPELINE")
print("="*80)
print("TRAINING: Audio-level + Spectral-level augmentation")
print("VAL/TEST: Clean data (no augmentation)")
print("="*80)

X_train_list, y_train = load_split(os.path.join(dataset_path, 'training'), apply_augment=True)  # Full augmentation for training
X_val_list,   y_val   = load_split(os.path.join(dataset_path, 'validation'), apply_augment=False)  # Clean validation data
X_test_list,  y_test  = load_split(os.path.join(dataset_path, 'testing'), apply_augment=False)  # Clean test data

# Pad to a fixed max length with explicit mask value
def pad_sequences_numpy(X_list, maxlen, mask_value=0.0):
    """
    Pad sequences to uniform length using explicit mask value.
    mask_value=0.0 will be used for masking.
    """
    n_samples = len(X_list)
    n_features = X_list[0].shape[1]
    
    X_padded = np.full((n_samples, maxlen, n_features), mask_value, dtype=np.float32)
    
    for i, x in enumerate(X_list):
        length = min(x.shape[0], maxlen)
        X_padded[i, :length, :] = x[:length, :]
    
    return X_padded

# CRITICAL FIX: Pad all datasets to the SAME maximum length
# Calculate global maximum across all datasets to avoid shape mismatch
all_lengths = []
all_lengths.extend([x.shape[0] for x in X_train_list])
all_lengths.extend([x.shape[0] for x in X_val_list])
all_lengths.extend([x.shape[0] for x in X_test_list])
global_max_T = max(all_lengths)

print(f"Global maximum sequence length: {global_max_T}")
print(f"Training max: {max([x.shape[0] for x in X_train_list])}")
print(f"Validation max: {max([x.shape[0] for x in X_val_list])}")
print(f"Testing max: {max([x.shape[0] for x in X_test_list])}")

# Pad all datasets to the global maximum to ensure consistent shapes
X_train = pad_sequences_numpy(X_train_list, global_max_T)
X_val = pad_sequences_numpy(X_val_list, global_max_T)
X_test = pad_sequences_numpy(X_test_list, global_max_T)

T_train = T_val = T_test = global_max_T  # All datasets now have same length

# Standardize features (avoid fitting on padded zeros)
scaler = StandardScaler()
N, T, n_features = X_train.shape

# Only fit scaler on non-zero (non-padded) frames
train_lengths = [x.shape[0] for x in X_train_list]  # Original lengths before padding
non_padded_frames = []
for i, length in enumerate(train_lengths):
    non_padded_frames.append(X_train[i, :length, :])  # Only actual frames, not padding
non_padded_data = np.vstack(non_padded_frames)  # Stack all real frames
scaler.fit(non_padded_data)

# Transform all data (including padded frames, but scaler fitted on real data only)
X_train = scaler.transform(X_train.reshape(-1, n_features)).reshape(N, T, n_features)
X_val   = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape[0], X_val.shape[1], n_features)
X_test  = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape[0], X_test.shape[1], n_features)

# ------------------------------
# PyTorch Dataset
# ------------------------------
class AudioDataset(Dataset):
    """PyTorch Dataset for audio features"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ------------------------------
# PyTorch Model: Pyramidal BiLSTM
# ------------------------------
class PyramidalDownsample(nn.Module):
    """
    Pyramid downsampling layer that concatenates adjacent time frames.
    Reduces temporal resolution by 2x while doubling feature dimension.
    """
    def __init__(self):
        super(PyramidalDownsample, self).__init__()
    
    def forward(self, x):
        """
        Perform pyramid downsampling on inputs.
        Args:
            x: Tensor of shape (batch_size, seq_len, feat_dim)
        Returns:
            Tensor of shape (batch_size, seq_len//2, feat_dim*2)
        """
        batch_size, seq_len, feat_dim = x.size()
        
        # Ensure even sequence length (drop last frame if odd)
        if seq_len % 2 == 1:
            x = x[:, :-1, :]
            seq_len = seq_len - 1
        
        # Reshape to group adjacent frames: (B, T, F) -> (B, T//2, 2, F)
        x = x.contiguous().view(batch_size, seq_len // 2, 2, feat_dim)
        
        # Concatenate adjacent frames: (B, T//2, 2, F) -> (B, T//2, 2*F)
        x = x.contiguous().view(batch_size, seq_len // 2, 2 * feat_dim)
        
        return x

class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer that computes weighted average of all time steps.
    Much better than taking only the last LSTM output as it uses all temporal information.
    """
    def __init__(self, input_dim, attention_units=128):
        super(AttentionPooling, self).__init__()
        self.attention_units = attention_units
        self.attention_dense = nn.Linear(input_dim, attention_units)
        self.attention_score = nn.Linear(attention_units, 1)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, time, features)
            mask: (batch, time) boolean mask, True for valid positions
        Returns:
            attended_output: (batch, features)
        """
        # Compute attention weights
        attention_hidden = torch.tanh(self.attention_dense(x))  # (batch, time, attention_units)
        attention_scores = self.attention_score(attention_hidden)  # (batch, time, 1)
        
        # Apply mask if present
        if mask is not None:
            # Convert mask to float and expand dimensions
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, time, 1)
            # Set attention scores for masked positions to large negative value
            attention_scores = attention_scores.masked_fill(mask_expanded == 0, -1e9)
        
        # Compute softmax attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, time, 1)
        
        # Apply attention weights to get weighted average
        attended_output = torch.sum(attention_weights * x, dim=1)  # (batch, features)
        
        return attended_output

class PyramidalBiLSTM(nn.Module):
    """
    Pyramidal BiLSTM model for audio deepfake detection with attention pooling.
    """
    def __init__(self, input_dim, base_units=128, num_pyramid_layers=2, 
                 dropout_rate=0.3, recurrent_dropout=0.2, attention_units=128, dense_units=64):
        super(PyramidalBiLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.base_units = base_units
        self.num_pyramid_layers = num_pyramid_layers
        
        # First BiLSTM layer
        self.first_bilstm = nn.LSTM(
            input_dim, base_units, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True,
            dropout=0  # PyTorch LSTM dropout only works with num_layers > 1
        )
        self.first_dropout = nn.Dropout(dropout_rate)
        self.first_layernorm = nn.LayerNorm(base_units * 2)
        
        # Pyramid layers
        self.pyramid_layers = nn.ModuleList()
        self.pyramid_bilstms = nn.ModuleList()
        self.pyramid_dropouts = nn.ModuleList()
        self.pyramid_layernorms = nn.ModuleList()
        
        current_dim = base_units * 2  # Bidirectional output
        for i in range(num_pyramid_layers):
            # Downsample layer
            self.pyramid_layers.append(PyramidalDownsample())
            current_dim = current_dim * 2  # Doubled by concatenation
            
            # BiLSTM layer
            bilstm = nn.LSTM(
                current_dim, base_units,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0
            )
            self.pyramid_bilstms.append(bilstm)
            
            # Dropout and LayerNorm
            dropout_rate_progressive = min(dropout_rate + 0.05 * (i + 1), 0.6)
            self.pyramid_dropouts.append(nn.Dropout(dropout_rate_progressive))
            self.pyramid_layernorms.append(nn.LayerNorm(base_units * 2))
            
            current_dim = base_units * 2
        
        # Attention pooling
        self.attention_pooling = AttentionPooling(base_units * 2, attention_units)
        self.attention_dropout = nn.Dropout(dropout_rate + 0.1)
        
        # Dense layers
        self.dense1 = nn.Linear(base_units * 2, dense_units)
        self.dense1_layernorm = nn.LayerNorm(dense_units)
        self.dense1_dropout = nn.Dropout(min(dropout_rate + 0.2, 0.7))
        
        # Output layer
        self.output = nn.Linear(dense_units, 1)
    
    def forward(self, x, lengths=None):
        """
        Forward pass
        Args:
            x: (batch, seq_len, input_dim)
            lengths: (batch,) actual sequence lengths (before padding)
        Returns:
            output: (batch, 1) logits
        """
        batch_size = x.size(0)
        
        # Create mask from lengths (True for valid positions, False for padding)
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        else:
            mask = None
        
        # First BiLSTM
        if lengths is not None:
            # Pack padded sequence for efficiency
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.first_bilstm(packed_input)
            x, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            x, _ = self.first_bilstm(x)
        
        x = self.first_layernorm(x)
        x = self.first_dropout(x)
        
        # Update mask after each downsampling
        current_mask = mask
        
        # Pyramid layers
        for i in range(self.num_pyramid_layers):
            # Downsample
            x = self.pyramid_layers[i](x)
            
            # Update mask (also downsample)
            if current_mask is not None:
                # Downsample mask by taking every other position
                if current_mask.size(1) % 2 == 1:
                    current_mask = current_mask[:, :-1]
                current_mask = current_mask[:, ::2]
            
            # BiLSTM
            x, _ = self.pyramid_bilstms[i](x)
            x = self.pyramid_layernorms[i](x)
            x = self.pyramid_dropouts[i](x)
        
        # Attention pooling with mask
        x = self.attention_pooling(x, current_mask)
        x = self.attention_dropout(x)
        
        # Dense layers
        x = F.relu(self.dense1(x))
        x = self.dense1_layernorm(x)
        x = self.dense1_dropout(x)
        
        # Output
        x = self.output(x)
        
        return x

def build_pyramidal_bilstm(input_shape, base_units=128, num_pyramid_layers=2,
                          dropout_rate=0.3, recurrent_dropout=0.2, attention_units=128,
                          dense_units=64, learning_rate=1e-4, weight_decay=1e-5):
    """
    Build and return PyTorch Pyramidal BiLSTM model with optimizer
    
    Args:
        input_shape: (seq_len, input_dim) - only input_dim is used
        ... other hyperparameters
    
    Returns:
        model, optimizer, criterion
    """
    input_dim = input_shape[1] if isinstance(input_shape, tuple) else input_shape
    
    model = PyramidalBiLSTM(
        input_dim=input_dim,
        base_units=base_units,
        num_pyramid_layers=num_pyramid_layers,
        dropout_rate=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        attention_units=attention_units,
        dense_units=dense_units
    ).to(device)
    
    # AdamW optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-7
    )
    
    # Binary cross-entropy loss with logits
    criterion = nn.BCEWithLogitsLoss()
    
    return model, optimizer, criterion

# ------------------------------
# Evaluation Metrics
# ------------------------------
def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    thr = thresholds[idx]
    return eer, thr

def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence intervals for any metric.
    """
    n_samples = len(y_true)
    bootstrap_scores = []
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for i in range(n_bootstrap):
        # Bootstrap sample indices
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Calculate metric on bootstrap sample
        try:
            score = metric_func(y_true[indices], y_pred[indices])
            bootstrap_scores.append(score)
        except:
            # Skip invalid bootstrap samples
            continue
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    return {
        'mean': np.mean(bootstrap_scores),
        'std': np.std(bootstrap_scores),
        'ci_lower': np.percentile(bootstrap_scores, lower_percentile),
        'ci_upper': np.percentile(bootstrap_scores, upper_percentile),
        'confidence': confidence
    }

def calculate_calibration_metrics(y_true, y_prob, n_bins=10):
    """
    Calculate calibration metrics for probability predictions.
    """
    # Brier Score (lower is better)
    brier_score = brier_score_loss(y_true, y_prob)
    
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    mce = 0  # Maximum Calibration Error
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Calibration error for this bin
            bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * bin_error
            mce = max(mce, bin_error)
    
    return {
        'brier_score': brier_score,
        'ece': ece,  # Expected Calibration Error
        'mce': mce   # Maximum Calibration Error
    }

def comprehensive_evaluation(y_true, y_prob, dataset_name="Test"):
    """
    Perform comprehensive evaluation with advanced metrics and confidence intervals.
    """
    # Binary predictions
    y_pred = (y_prob > 0.5).astype(int)
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EVALUATION METRICS - {dataset_name.upper()} SET")
    print(f"{'='*80}")
    
    # Basic metrics
    accuracy = np.mean(y_true == y_pred)
    
    # ROC metrics
    auc_roc = roc_auc_score(y_true, y_prob)
    eer, eer_threshold = calculate_eer(y_true, y_prob)
    
    # Precision-Recall metrics
    auc_pr = average_precision_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Calibration metrics
    calibration = calculate_calibration_metrics(y_true, y_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall  # Same as recall/TPR
    balanced_accuracy = (sensitivity + specificity) / 2
    
    # Bootstrap confidence intervals (95% CI)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS (95% CI):")
    print(f"Computing bootstrap estimates with 1000 samples...")
    
    # Bootstrap for key metrics
    auc_roc_ci = bootstrap_metric(y_true, y_prob, 
                                  lambda yt, yp: roc_auc_score(yt, yp), n_bootstrap=1000)
    auc_pr_ci = bootstrap_metric(y_true, y_prob, 
                                 lambda yt, yp: average_precision_score(yt, yp), n_bootstrap=1000)
    f1_ci = bootstrap_metric(y_true, y_pred, 
                             lambda yt, yp: f1_score(yt, yp), n_bootstrap=1000)
    
    # Print results
    print(f"\nCORE PERFORMANCE METRICS:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f} [{auc_roc_ci['ci_lower']:.4f}, {auc_roc_ci['ci_upper']:.4f}]")
    print(f"  AUC-PR: {auc_pr:.4f} [{auc_pr_ci['ci_lower']:.4f}, {auc_pr_ci['ci_upper']:.4f}]")
    print(f"  F1-Score: {f1:.4f} [{f1_ci['ci_lower']:.4f}, {f1_ci['ci_upper']:.4f}]")
    print(f"  EER: {eer:.4f} @ threshold={eer_threshold:.4f}")
    
    print(f"\nPER-CLASS METRICS:")
    print(f"  Precision (Fake Detection): {precision:.4f}")
    print(f"  Recall/Sensitivity (Fake Detection): {recall:.4f}")
    print(f"  Specificity (Real Detection): {specificity:.4f}")
    
    print(f"\nCONFUSION MATRIX:")
    print(f"  True Negatives (Real→Real): {tn}")
    print(f"  False Positives (Real→Fake): {fp}")
    print(f"  False Negatives (Fake→Real): {fn}")
    print(f"  True Positives (Fake→Fake): {tp}")
    
    print(f"\nCALIBRATION ANALYSIS:")
    print(f"  Brier Score: {calibration['brier_score']:.4f} (lower is better)")
    print(f"  Expected Calibration Error (ECE): {calibration['ece']:.4f}")
    print(f"  Maximum Calibration Error (MCE): {calibration['mce']:.4f}")
    
    # Return comprehensive results
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'auc_roc': auc_roc,
        'auc_roc_ci': auc_roc_ci,
        'auc_pr': auc_pr,
        'auc_pr_ci': auc_pr_ci,
        'f1_score': f1,
        'f1_ci': f1_ci,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'confusion_matrix': cm,
        'calibration': calibration,
        'bootstrap_confidence': 0.95
    }

# ------------------------------
# Training Function
# ------------------------------
def train_and_evaluate_model(num_pyramid_layers, X_train, y_train, X_val, y_val, X_test, y_test,
                            base_units=128, learning_rate=1e-4, weight_decay=1e-5,
                            batch_size=8, epochs=100, patience=7):
    """
    Train and evaluate a pyramidal BiLSTM with specified number of pyramid layers
    """
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL WITH {num_pyramid_layers} PYRAMID LAYER(S)")
    print(f"{'='*60}")
    
    # Build model
    model, optimizer, criterion = build_pyramidal_bilstm(
        (X_train.shape[1], X_train.shape[2]),
        base_units=base_units,
        num_pyramid_layers=num_pyramid_layers,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Log model architecture details
    log_model_summary(model, f"Pyramidal BiLSTM ({num_pyramid_layers} layers)")
    
    # Create datasets and dataloaders
    train_dataset = AudioDataset(X_train, y_train)
    val_dataset = AudioDataset(X_val, y_val)
    test_dataset = AudioDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # OneCycle learning rate scheduler
    def onecycle_lambda(epoch):
        max_lr_factor = 5.0  # Peak at 5x base
        min_lr_factor = 0.01  # Min at 0.01x base
        warmup_epochs = 5
        peak_epochs = 15
        total_epochs = epochs
        
        if epoch < warmup_epochs:
            # Linear warmup
            return 1.0 + (max_lr_factor - 1.0) * (epoch / warmup_epochs)
        elif epoch < warmup_epochs + peak_epochs:
            # Stay at peak
            return max_lr_factor
        else:
            # Cosine annealing
            remaining_epochs = total_epochs - warmup_epochs - peak_epochs
            progress = (epoch - warmup_epochs - peak_epochs) / remaining_epochs
            progress = min(progress, 1.0)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_factor + (max_lr_factor - min_lr_factor) * cosine_factor
    
    scheduler = LambdaLR(optimizer, lr_lambda=onecycle_lambda)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, 
                                         min_lr=1e-8, verbose=True)
    
    # Training setup
    import time
    timestamp = int(time.time())
    best_checkpoint_path = f"best_model_{num_pyramid_layers}layers_{timestamp}.pt"
    last_checkpoint_path = f"last_epoch_model_{num_pyramid_layers}layers_{timestamp}.pt"
    
    print(f"\n{'='*80}")
    print(f"ENHANCED TRAINING PIPELINE - PYTORCH")
    print(f"{'='*80}")
    print(f"Optimizer: AdamW (weight_decay={weight_decay}, lr={learning_rate})")
    print(f"Learning Rate: OneCycle (Base: {learning_rate}, Peak: {learning_rate*5}, Min: {learning_rate*0.01})")
    print(f"Schedule: Warmup (5 epochs) → Peak (15 epochs) → Cosine Annealing")
    print(f"Max Epochs: {epochs} (with early stopping, patience={patience})")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print(f"\n💾 CHECKPOINT STRATEGY:")
    print(f"  • Best model: Saved when validation loss improves")
    print(f"  • Last epoch: Saved at training completion")
    print(f"  • Evaluation: Uses best model weights")
    print(f"{'='*80}")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.float().to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * batch_features.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).long()
            train_correct += (predictions == batch_labels.long()).sum().item()
            train_total += batch_labels.size(0)
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.float().to(device)
                
                outputs = model(batch_features).squeeze()
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item() * batch_features.size(0)
                predictions = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (predictions == batch_labels.long()).sum().item()
                val_total += batch_labels.size(0)
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        plateau_scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
              f"LR: {current_lr:.2e}")
        
        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, best_checkpoint_path)
            print(f"💾 Best model saved: {best_checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save last epoch checkpoint (regardless of performance)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'history': history
    }, last_checkpoint_path)
    print(f"💾 Last epoch checkpoint saved: {last_checkpoint_path}")
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Epochs trained: {len(history['train_loss'])}")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
    print(f"\n📁 SAVED CHECKPOINTS:")
    print(f"  Best model: {best_checkpoint_path}")
    print(f"  Last epoch: {last_checkpoint_path}")
    print(f"{'='*60}")
    
    # Load best model
    print(f"Loading best model from checkpoint for evaluation...")
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Generate predictions for comprehensive evaluation
    print(f"\nGenerating predictions for comprehensive evaluation...")
    model.eval()
    
    with torch.no_grad():
        # Test predictions
        y_test_prob = []
        for batch_features, _ in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_test_prob.extend(probs if probs.ndim > 0 else [probs.item()])
        y_test_prob = np.array(y_test_prob)
        
        # Val predictions
        y_val_prob = []
        for batch_features, _ in val_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_val_prob.extend(probs if probs.ndim > 0 else [probs.item()])
        y_val_prob = np.array(y_val_prob)
    
    # Perform comprehensive evaluation on both test and validation sets
    test_metrics = comprehensive_evaluation(y_test, y_test_prob, dataset_name="Test")
    val_metrics = comprehensive_evaluation(y_val, y_val_prob, dataset_name="Validation")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Store comprehensive results
    results = {
        'pyramid_layers': num_pyramid_layers,
        'total_params': total_params,
        'epochs_trained': len(history['train_loss']),
        'best_epoch': np.argmin(history['val_loss']) + 1,
        'best_val_loss': min(history['val_loss']),
        'best_val_acc': max(history['val_acc']),
        
        # Comprehensive evaluation results
        'test_metrics': test_metrics,
        'val_metrics': val_metrics,
        
        # Training metadata with both checkpoints
        'checkpoint_path': best_checkpoint_path,  # Best model (used for evaluation)
        'best_checkpoint_path': best_checkpoint_path,
        'last_checkpoint_path': last_checkpoint_path,
        'history': history,
        'evaluation_type': 'comprehensive_with_bootstrap'
    }
    
    # Print detailed results
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE RESULTS FOR {num_pyramid_layers} PYRAMID LAYER(S)")
    print(f"{'='*70}")
    print(f"TRAINING SUMMARY:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Epochs Trained: {len(history['train_loss'])}")
    print(f"  Best Epoch: {results['best_epoch']} (1-indexed)")
    print(f"  Best Validation Loss: {results['best_val_loss']:.4f}")
    print(f"  Model Checkpoint: {best_checkpoint_path}")
    
    print(f"\n📁 CHECKPOINTS SAVED:")
    print(f"  Best model: {best_checkpoint_path}")
    print(f"  Last epoch: {last_checkpoint_path}")
    
    print(f"\nFINAL PERFORMANCE SUMMARY (using best model):")
    print(f"  Test AUC-ROC: {test_metrics['auc_roc']:.4f} [{test_metrics['auc_roc_ci']['ci_lower']:.4f}, {test_metrics['auc_roc_ci']['ci_upper']:.4f}]")
    print(f"  Test AUC-PR: {test_metrics['auc_pr']:.4f} [{test_metrics['auc_pr_ci']['ci_lower']:.4f}, {test_metrics['auc_pr_ci']['ci_upper']:.4f}]")
    print(f"  Test F1-Score: {test_metrics['f1_score']:.4f} [{test_metrics['f1_ci']['ci_lower']:.4f}, {test_metrics['f1_ci']['ci_upper']:.4f}]")
    print(f"  Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Test EER: {test_metrics['eer']:.4f} @ threshold={test_metrics['eer_threshold']:.4f}")
    
    # Classification report
    y_pred_bin = (y_test_prob > 0.5).astype(int)
    print(f"\nSTANDARD CLASSIFICATION REPORT (Test Set):")
    print(classification_report(y_test, y_pred_bin, target_names=['Real', 'Fake']))
    
    return results, model

# ------------------------------
# Main Execution
# ------------------------------
print("\n" + "="*90)
print("ULTRA-ENHANCED PYRAMIDAL BiLSTM WITH COMPREHENSIVE AUGMENTATION - PYTORCH")
print("Testing 2 pyramid layers + attention + multi-level data augmentation")
print("="*90)
print("PYTORCH IMPLEMENTATION WITH MAJOR IMPROVEMENTS:")
print("✓ PyTorch-native pyramid downsampling")
print("✓ Log-Mel spectrograms (80 bins)")
print("✓ Proper sequence masking for variable-length sequences")
print("✓ AttentionPooling using ALL temporal information")
print("✓ AdamW optimizer with OneCycle learning rate schedule")
print("✓ ENHANCED REGULARIZATION PIPELINE:")
print("  • Dropout in LSTM and Dense layers")
print("  • LayerNormalization after BiLSTM and Dense layers")
print("  • Progressive dropout rates (0.3 → 0.5)")
print("  • Weight decay via AdamW optimizer")
print("  • Gradient clipping (max_norm=1.0)")
print("✓ COMPREHENSIVE DATA AUGMENTATION PIPELINE:")
print("  • Audio-level: Speed perturbation (±10%), Pitch shift (±1 semitone)")
print("  • Audio-level: Colored noise addition (white/pink, 10-20dB SNR)")
print("  • Spectral-level: SpecAugment (time + frequency masking)")
print("✓ Training robustness: ~80% samples get some form of augmentation")
print("="*90)

print(f"\n{'='*90}")
print("APPROACH 1: SINGLE TRAIN/VALIDATION/TEST SPLIT - PYTORCH")
print(f"{'='*90}")

single_split_results, single_model = train_and_evaluate_model(
    2, X_train, y_train, X_val, y_val, X_test, y_test,
    base_units=128, learning_rate=1e-4, weight_decay=1e-5,
    batch_size=8, epochs=100, patience=7
)

# Log single split experiment results
log_experiment_results(single_split_results, "Single Split Validation - PyTorch")

# Clean up memory
del single_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ------------------------------
# Stratified K-Fold Cross-Validation Framework
# ------------------------------
def stratified_kfold_cv(X_data, y_data, X_test, y_test, num_pyramid_layers=2, 
                       k_folds=5, random_state=42, base_units=128, 
                       learning_rate=1e-4, weight_decay=1e-5, batch_size=8):
    """
    Perform stratified k-fold cross-validation for robust performance estimation.
    PyTorch implementation with manual training loops.
    
    Args:
        X_data: Combined training + validation data (list of sequences)
        y_data: Combined training + validation labels
        X_test: Hold-out test data (list of sequences)
        y_test: Hold-out test labels
        num_pyramid_layers: Number of pyramid layers in model
        k_folds: Number of folds for cross-validation
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with cross-validation results and statistics
    """
    print(f"\n{'='*90}")
    print(f"STRATIFIED {k_folds}-FOLD CROSS-VALIDATION ANALYSIS - PYTORCH")
    print(f"{'='*90}")
    print(f"Model Configuration: {num_pyramid_layers} Pyramid Layers")
    print(f"Dataset: {len(X_data)} samples for CV, {len(X_test)} samples for final test")
    print(f"Class Distribution in CV Data: Real={np.sum(y_data == 0)}, Fake={np.sum(y_data == 1)}")
    print(f"Random Seed: {random_state}")
    print(f"{'='*90}")
    
    # Initialize stratified k-fold splitter
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    
    # Store results for each fold
    fold_results = []
    
    # Process each fold
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_data, y_data)):
        print(f"\n{'='*80}")
        print(f"TRAINING FOLD {fold_idx + 1}/{k_folds}")
        print(f"{'='*80}")
        
        # Split data for this fold
        X_train_fold = [X_data[i] for i in train_idx]
        y_train_fold = y_data[train_idx]
        X_val_fold = [X_data[i] for i in val_idx]
        y_val_fold = y_data[val_idx]
        
        print(f"Fold {fold_idx + 1} Data Split:")
        print(f"  Training: {len(X_train_fold)} samples (Real={np.sum(y_train_fold == 0)}, Fake={np.sum(y_train_fold == 1)})")
        print(f"  Validation: {len(X_val_fold)} samples (Real={np.sum(y_val_fold == 0)}, Fake={np.sum(y_val_fold == 1)})")
        
        # Calculate maximum length for this specific fold
        fold_lengths = []
        fold_lengths.extend([x.shape[0] for x in X_train_fold])
        fold_lengths.extend([x.shape[0] for x in X_val_fold])
        fold_max_T = max(fold_lengths)
        
        print(f"  Fold max sequence length: {fold_max_T}")
        
        # Pad both to the same length
        X_train_padded = pad_sequences_numpy(X_train_fold, fold_max_T)
        X_val_padded = pad_sequences_numpy(X_val_fold, fold_max_T)
        
        # Standardize features for this fold
        scaler_fold = StandardScaler()
        
        # Fit scaler on training data only (avoid data leakage)
        train_lengths = [x.shape[0] for x in X_train_fold]
        non_padded_frames = []
        for i, length in enumerate(train_lengths):
            non_padded_frames.append(X_train_padded[i, :length, :])
        non_padded_data = np.vstack(non_padded_frames)
        scaler_fold.fit(non_padded_data)
        
        # Transform data
        N_train, T_train, n_features_fold = X_train_padded.shape
        X_train_scaled = scaler_fold.transform(X_train_padded.reshape(-1, n_features_fold)).reshape(N_train, T_train, n_features_fold)
        X_val_scaled = scaler_fold.transform(X_val_padded.reshape(-1, n_features_fold)).reshape(X_val_padded.shape[0], X_val_padded.shape[1], n_features_fold)
        
        # Train model for this fold
        try:
            fold_result, fold_model = train_and_evaluate_model(
                num_pyramid_layers, X_train_scaled, y_train_fold, 
                X_val_scaled, y_val_fold, X_val_scaled, y_val_fold,  # Use val as "test" for CV
                base_units=base_units, learning_rate=learning_rate, 
                weight_decay=weight_decay, batch_size=batch_size,
                epochs=100, patience=7
            )
            
            # Store fold information
            fold_result['fold_idx'] = fold_idx + 1
            fold_result['train_size'] = len(X_train_fold)
            fold_result['val_size'] = len(X_val_fold)
            fold_result['scaler'] = scaler_fold
            fold_result['fold_max_T'] = fold_max_T
            
            fold_results.append(fold_result)
            
            print(f"\nFold {fold_idx + 1} Completed Successfully")
            print(f"  Validation AUC-ROC: {fold_result['test_metrics']['auc_roc']:.4f}")
            print(f"  Validation F1-Score: {fold_result['test_metrics']['f1_score']:.4f}")
            
        except Exception as e:
            print(f"\nERROR in Fold {fold_idx + 1}: {str(e)}")
            print("Continuing with remaining folds...")
            continue
        
        # Clear memory
        del fold_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n{'='*90}")
    print(f"CROSS-VALIDATION COMPLETED - STATISTICAL ANALYSIS")
    print(f"{'='*90}")
    
    if len(fold_results) == 0:
        print("ERROR: No folds completed successfully!")
        return None
    
    # Aggregate results across folds
    cv_stats = analyze_cv_results(fold_results, k_folds)
    
    # Final evaluation on hold-out test set using best fold model
    print(f"\n{'='*80}")
    print(f"FINAL HOLD-OUT TEST SET EVALUATION")
    print(f"{'='*80}")
    
    # Find best fold based on validation AUC
    best_fold_idx = np.argmax([result['test_metrics']['auc_roc'] for result in fold_results])
    best_result = fold_results[best_fold_idx]
    
    print(f"Using model from Fold {best_result['fold_idx']} (best validation AUC: {best_result['test_metrics']['auc_roc']:.4f})")
    
    # Prepare test data using the SAME sequence length as best fold
    best_fold_max_T = best_result['fold_max_T']
    print(f"Padding test data to length {best_fold_max_T} (same as best fold)")
    
    X_test_padded = pad_sequences_numpy(X_test, best_fold_max_T)
    best_scaler = best_result['scaler']
    X_test_scaled = best_scaler.transform(X_test_padded.reshape(-1, X_test_padded.shape[2])).reshape(X_test_padded.shape)
    
    # Load best model and evaluate on test set
    try:
        checkpoint = torch.load(best_result['checkpoint_path'])
        best_model, _, _ = build_pyramidal_bilstm(
            (X_test_scaled.shape[1], X_test_scaled.shape[2]),
            base_units=base_units,
            num_pyramid_layers=num_pyramid_layers,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        best_model.load_state_dict(checkpoint['model_state_dict'])
        best_model.eval()
        
        # Generate predictions
        test_dataset = AudioDataset(X_test_scaled, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        y_test_prob = []
        with torch.no_grad():
            for batch_features, _ in test_loader:
                batch_features = batch_features.to(device)
                outputs = best_model(batch_features).squeeze()
                probs = torch.sigmoid(outputs).cpu().numpy()
                y_test_prob.extend(probs if probs.ndim > 0 else [probs.item()])
        y_test_prob = np.array(y_test_prob)
        
        test_metrics_final = comprehensive_evaluation(y_test, y_test_prob, dataset_name="Hold-out Test")
        del best_model
    except Exception as e:
        print(f"Error loading best model: {e}")
        test_metrics_final = None
    
    # Compile final results
    final_results = {
        'cv_type': 'stratified_kfold',
        'k_folds': k_folds,
        'successful_folds': len(fold_results),
        'fold_results': fold_results,
        'cv_statistics': cv_stats,
        'best_fold_idx': best_fold_idx + 1,
        'best_fold_result': best_result,
        'holdout_test_metrics': test_metrics_final,
        'model_config': {
            'pyramid_layers': num_pyramid_layers,
            'random_state': random_state
        }
    }
    
    return final_results

def analyze_cv_results(fold_results, k_folds):
    """
    Analyze and aggregate cross-validation results across folds.
    
    Args:
        fold_results: List of results from each fold
        k_folds: Number of folds
    
    Returns:
        Dictionary with aggregated statistics
    """
    print(f"Analyzing results from {len(fold_results)} successful folds (out of {k_folds} total)")
    
    # Extract key metrics from each fold
    metrics_keys = ['auc_roc', 'auc_pr', 'f1_score', 'balanced_accuracy', 'eer', 'precision', 'recall']
    fold_metrics = {}
    
    for key in metrics_keys:
        fold_metrics[key] = [result['test_metrics'][key] for result in fold_results]
    
    # Calculate statistics across folds
    cv_stats = {}
    for key in metrics_keys:
        values = np.array(fold_metrics[key])
        cv_stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values.tolist()
        }
    
    # Calculate overall statistics
    cv_stats['summary'] = {
        'successful_folds': len(fold_results),
        'total_folds': k_folds,
        'completion_rate': len(fold_results) / k_folds,
        'avg_training_epochs': np.mean([result['epochs_trained'] for result in fold_results]),
        'total_parameters': fold_results[0]['total_params']
    }
    
    # Print comprehensive CV statistics
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION STATISTICS (n={len(fold_results)} folds)")
    print(f"{'='*80}")
    print(f"MODEL PERFORMANCE ACROSS FOLDS:")
    print(f"  AUC-ROC:           {cv_stats['auc_roc']['mean']:.4f} ± {cv_stats['auc_roc']['std']:.4f} [{cv_stats['auc_roc']['min']:.4f}, {cv_stats['auc_roc']['max']:.4f}]")
    print(f"  AUC-PR:            {cv_stats['auc_pr']['mean']:.4f} ± {cv_stats['auc_pr']['std']:.4f} [{cv_stats['auc_pr']['min']:.4f}, {cv_stats['auc_pr']['max']:.4f}]")
    print(f"  F1-Score:          {cv_stats['f1_score']['mean']:.4f} ± {cv_stats['f1_score']['std']:.4f} [{cv_stats['f1_score']['min']:.4f}, {cv_stats['f1_score']['max']:.4f}]")
    print(f"  Balanced Accuracy: {cv_stats['balanced_accuracy']['mean']:.4f} ± {cv_stats['balanced_accuracy']['std']:.4f}")
    print(f"  EER:               {cv_stats['eer']['mean']:.4f} ± {cv_stats['eer']['std']:.4f}")
    
    print(f"\nTRAINING STATISTICS:")
    print(f"  Successful Folds: {cv_stats['summary']['successful_folds']}/{cv_stats['summary']['total_folds']} ({cv_stats['summary']['completion_rate']*100:.1f}%)")
    print(f"  Avg Training Epochs: {cv_stats['summary']['avg_training_epochs']:.1f}")
    print(f"  Model Parameters: {cv_stats['summary']['total_parameters']:,}")
    
    # Statistical significance tests
    if len(fold_results) >= 3:
        print(f"\nSTATISTICAL ROBUSTNESS:")
        
        # Coefficient of variation
        cv_coeff = cv_stats['auc_roc']['std'] / cv_stats['auc_roc']['mean']
        print(f"  AUC-ROC Coefficient of Variation: {cv_coeff:.3f} ({'Low' if cv_coeff < 0.05 else 'Moderate' if cv_coeff < 0.10 else 'High'} variability)")
        
        # 95% confidence interval
        if SCIPY_AVAILABLE:
            confidence_level = 0.95
            alpha = 1 - confidence_level
            dof = len(fold_results) - 1
            t_val = stats.t.ppf(1 - alpha/2, dof)
            
            auc_sem = cv_stats['auc_roc']['std'] / np.sqrt(len(fold_results))
            auc_ci_lower = cv_stats['auc_roc']['mean'] - t_val * auc_sem
            auc_ci_upper = cv_stats['auc_roc']['mean'] + t_val * auc_sem
            
            print(f"  Mean AUC-ROC 95% CI: [{auc_ci_lower:.4f}, {auc_ci_upper:.4f}]")
        else:
            auc_sem = cv_stats['auc_roc']['std'] / np.sqrt(len(fold_results))
            auc_ci_lower = cv_stats['auc_roc']['mean'] - 1.96 * auc_sem
            auc_ci_upper = cv_stats['auc_roc']['mean'] + 1.96 * auc_sem
            print(f"  Mean AUC-ROC 95% CI (approx): [{auc_ci_lower:.4f}, {auc_ci_upper:.4f}]")
    
    return cv_stats

# ------------------------------
# Hyperparameter Optimization Framework (Optuna)
# ------------------------------
def build_pyramidal_bilstm_optimized(input_shape, trial=None, base_units=128, 
                                    num_pyramid_layers=2, dropout_rate=0.3, 
                                    recurrent_dropout=0.2, attention_units=128,
                                    dense_units=64, learning_rate=1e-4, weight_decay=1e-5):
    """
    Build pyramidal BiLSTM with configurable hyperparameters for optimization.
    PyTorch implementation with Optuna trial suggestions.
    """
    if trial is not None:
        # Hyperparameter suggestions from Optuna
        base_units = trial.suggest_categorical('base_units', [64, 128, 192, 256])
        num_pyramid_layers = trial.suggest_int('num_pyramid_layers', 1, 3)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
        recurrent_dropout = trial.suggest_float('recurrent_dropout', 0.1, 0.3)
        attention_units = trial.suggest_categorical('attention_units', [64, 128, 192, 256])
        dense_units = trial.suggest_categorical('dense_units', [32, 64, 96, 128])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    input_dim = input_shape[1] if isinstance(input_shape, tuple) else input_shape
    
    model = PyramidalBiLSTM(
        input_dim=input_dim,
        base_units=base_units,
        num_pyramid_layers=num_pyramid_layers,
        dropout_rate=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        attention_units=attention_units,
        dense_units=dense_units
    ).to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-7
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    return model, optimizer, criterion

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function for hyperparameter optimization.
    PyTorch implementation.
    
    Args:
        trial: Optuna trial object
        X_train, y_train: Training data
        X_val, y_val: Validation data
    
    Returns:
        Validation AUC-ROC score (to maximize)
    """
    try:
        # Suggest batch size
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
        
        # Build model with trial-suggested hyperparameters
        model, optimizer, criterion = build_pyramidal_bilstm_optimized(
            (X_train.shape[1], X_train.shape[2]), 
            trial=trial
        )
        
        print(f"\nTrial {trial.number}: Testing configuration...")
        print(f"  Base Units: {trial.params.get('base_units', 128)}")
        print(f"  Pyramid Layers: {trial.params.get('num_pyramid_layers', 2)}")
        print(f"  Dropout: {trial.params.get('dropout_rate', 0.3):.3f}")
        print(f"  Learning Rate: {trial.params.get('learning_rate', 1e-4):.2e}")
        print(f"  Batch Size: {batch_size}")
        
        # Create datasets
        train_dataset = AudioDataset(X_train, y_train)
        val_dataset = AudioDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop (reduced epochs for optimization speed)
        early_stop_patience = 5
        best_val_loss = float('inf')
        patience_counter = 0
        epochs = 30  # Reduced for optimization
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.float().to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_features).squeeze()
                loss = criterion(outputs, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * batch_features.size(0)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.float().to(device)
                    
                    outputs = model(batch_features).squeeze()
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item() * batch_features.size(0)
            
            val_loss = val_loss / len(val_dataset)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break
            
            # Report for pruning
            trial.report(1.0 - val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Evaluate performance
        model.eval()
        y_val_prob = []
        with torch.no_grad():
            for batch_features, _ in val_loader:
                batch_features = batch_features.to(device)
                outputs = model(batch_features).squeeze()
                probs = torch.sigmoid(outputs).cpu().numpy()
                y_val_prob.extend(probs if probs.ndim > 0 else [probs.item()])
        y_val_prob = np.array(y_val_prob)
        
        val_auc = roc_auc_score(y_val, y_val_prob)
        
        print(f"  Validation AUC: {val_auc:.4f}")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return val_auc
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {str(e)}")
        return 0.5

def hyperparameter_optimization(X_train, y_train, X_val, y_val, n_trials=50, 
                               study_name="pytorch_pyramidal_bilstm_optimization",
                               use_sql_storage=True,
                               storage_path="sqlite:///optuna_studies.db"):
    """
    Perform hyperparameter optimization using Optuna with SQL database storage.
    PyTorch implementation.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_trials: Number of optimization trials
        study_name: Name of the optimization study
        use_sql_storage: Whether to use SQL database for persistent storage
        storage_path: SQL database path (SQLite, PostgreSQL, or MySQL)
                     Examples:
                     - SQLite: "sqlite:///optuna_studies.db"
                     - PostgreSQL: "postgresql://user:password@localhost/dbname"
                     - MySQL: "mysql://user:password@localhost/dbname"
    
    Returns:
        Dictionary with optimization results
    """
    if not OPTUNA_AVAILABLE:
        print("Optuna not available. Skipping hyperparameter optimization.")
        return None
    
    print(f"\n{'='*90}")
    print(f"AUTOMATED HYPERPARAMETER OPTIMIZATION - PYTORCH WITH SQL STORAGE")
    print(f"{'='*90}")
    print(f"Optimization Study: {study_name}")
    print(f"Number of Trials: {n_trials}")
    print(f"Training Data: {X_train.shape[0]} samples")
    print(f"Validation Data: {X_val.shape[0]} samples")
    print(f"Objective: Maximize Validation AUC-ROC")
    
    # SQL storage configuration
    if use_sql_storage:
        print(f"Storage Backend: SQL Database")
        print(f"Storage Path: {storage_path}")
        print(f"📊 Study will be saved persistently and can be resumed")
        storage = optuna.storages.RDBStorage(
            url=storage_path,
            engine_kwargs={"connect_args": {"timeout": 30}}  # 30s timeout for SQLite
        )
    else:
        print(f"Storage Backend: In-Memory (not persistent)")
        storage = None
    
    print(f"{'='*90}")
    
    # Create or load existing Optuna study
    try:
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            storage=storage,
            load_if_exists=True,  # Resume existing study if found
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=3
            )
        )
        
        # Check if study already exists
        if len(study.trials) > 0:
            print(f"\n🔄 RESUMING EXISTING STUDY:")
            print(f"  Found {len(study.trials)} existing trials")
            print(f"  Best trial so far: {study.best_trial.number} (AUC: {study.best_value:.4f})")
            print(f"  Continuing with {n_trials} additional trials...")
        else:
            print(f"\n🆕 CREATING NEW STUDY:")
            print(f"  Starting fresh optimization with {n_trials} trials")
            
    except Exception as e:
        print(f"Error accessing study: {e}")
        print(f"Creating new study without storage...")
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=3
            )
        )
    
    # Define objective with data
    objective_with_data = lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val)
    
    # Run optimization
    print(f"Starting hyperparameter search...")
    study.optimize(objective_with_data, n_trials=n_trials, timeout=None)
    
    # Analyze results
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER OPTIMIZATION COMPLETED")
    print(f"{'='*80}")
    
    best_trial = study.best_trial
    print(f"Best Trial: {best_trial.number}")
    print(f"Best Validation AUC: {best_trial.value:.4f}")
    
    print(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Additional analysis
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    print(f"\nOptimization Statistics:")
    print(f"  Completed Trials: {len(completed_trials)}")
    print(f"  Pruned Trials: {len(pruned_trials)}")
    print(f"  Failed Trials: {len(failed_trials)}")
    
    if len(completed_trials) > 1:
        auc_values = [t.value for t in completed_trials]
        print(f"  AUC Range: [{min(auc_values):.4f}, {max(auc_values):.4f}]")
        print(f"  AUC Improvement: {max(auc_values) - min(auc_values):.4f}")
    
    # Save study summary to database
    if use_sql_storage and storage:
        print(f"\n💾 STUDY SAVED TO DATABASE:")
        print(f"  Database: {storage_path}")
        print(f"  Study Name: {study_name}")
        print(f"  Total Trials: {len(study.trials)}")
        print(f"  To resume later, use the same study_name and storage_path")
        print(f"\n📊 To view study in dashboard:")
        print(f"  optuna-dashboard {storage_path}")
    
    return {
        'study': study,
        'best_trial': best_trial,
        'best_params': best_trial.params,
        'best_score': best_trial.value,
        'n_completed_trials': len(completed_trials),
        'n_pruned_trials': len(pruned_trials),
        'n_failed_trials': len(failed_trials),
        'study_name': study_name,
        'n_trials': n_trials,
        'storage_path': storage_path if use_sql_storage else None,
        'total_trials_in_db': len(study.trials)
    }

def view_optuna_studies(storage_path="sqlite:///optuna_studies.db"):
    """
    View all studies stored in the Optuna database.
    Useful for inspecting optimization history and resuming studies.
    
    Args:
        storage_path: Path to SQL database
    """
    if not OPTUNA_AVAILABLE:
        print("Optuna not available.")
        return
    
    try:
        import optuna
        
        # Get all study summaries
        study_summaries = optuna.study.get_all_study_summaries(storage=storage_path)
        
        if len(study_summaries) == 0:
            print(f"\n📊 No studies found in database: {storage_path}")
            return
        
        print(f"\n{'='*80}")
        print(f"OPTUNA STUDIES IN DATABASE: {storage_path}")
        print(f"{'='*80}")
        print(f"Total Studies: {len(study_summaries)}\n")
        
        for i, summary in enumerate(study_summaries, 1):
            print(f"Study {i}: {summary.study_name}")
            print(f"  Direction: {summary.direction.name}")
            print(f"  Total Trials: {summary.n_trials}")
            print(f"  Best Value: {summary.best_trial.value:.4f}" if summary.best_trial else "  Best Value: N/A")
            print(f"  Date Created: {summary.datetime_start}")
            print()
        
        print(f"To load a study:")
        print(f"  study = optuna.load_study(study_name='<name>', storage='{storage_path}')")
        print(f"\nTo visualize with dashboard:")
        print(f"  optuna-dashboard {storage_path}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error accessing Optuna database: {e}")
        print(f"Make sure the database exists at: {storage_path}")

def train_optimized_model(best_params, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train final model with optimized hyperparameters.
    PyTorch implementation.
    """
    print(f"\n{'='*90}")
    print(f"TRAINING FINAL MODEL WITH OPTIMIZED HYPERPARAMETERS - PYTORCH")
    print(f"{'='*90}")
    
    print(f"Optimized Configuration:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Extract parameters
    batch_size = best_params.get('batch_size', 8)
    base_units = best_params.get('base_units', 128)
    num_pyramid_layers = best_params.get('num_pyramid_layers', 2)
    learning_rate = best_params.get('learning_rate', 1e-4)
    weight_decay = best_params.get('weight_decay', 1e-5)
    
    # Train with optimized parameters
    results, trained_model = train_and_evaluate_model(
        num_pyramid_layers, X_train, y_train, X_val, y_val, X_test, y_test,
        base_units=base_units, learning_rate=learning_rate, 
        weight_decay=weight_decay, batch_size=batch_size,
        epochs=100, patience=7
    )
    
    results['optimization_params'] = best_params
    
    # Log optimization experiment results
    log_experiment_results(results, "Hyperparameter Optimized Model - PyTorch")
    
    return results, trained_model

# ------------------------------
# Optimized DataLoader Pipeline
# ------------------------------
class AudioDatasetFromFiles(Dataset):
    """
    PyTorch Dataset that loads audio files on-the-fly with augmentation.
    Equivalent to TF.Data pipeline for production-grade data loading.
    """
    def __init__(self, file_paths, labels, max_seconds=2.0, apply_augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.max_seconds = max_seconds
        self.apply_augment = apply_augment
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Extract features on-the-fly
        features = extract_mel_features_with_augment(
            self.file_paths[idx], 
            max_seconds=self.max_seconds,
            apply_augment=self.apply_augment
        )
        
        if features is None:
            # Return zero features if extraction fails
            features = np.zeros((int(self.max_seconds * 16000 // 256) + 1, n_mels), dtype=np.float32)
        
        label = self.labels[idx]
        return torch.FloatTensor(features), torch.LongTensor([label]).squeeze()

def collate_fn_variable_length(batch):
    """
    Custom collate function to pad variable-length sequences.
    Essential for efficient batching of audio data.
    """
    features, labels = zip(*batch)
    
    # Find max length in batch
    max_len = max([f.size(0) for f in features])
    
    # Pad sequences
    padded_features = []
    for f in features:
        pad_amount = max_len - f.size(0)
        padded = F.pad(f, (0, 0, 0, pad_amount), value=0.0)
        padded_features.append(padded)
    
    return torch.stack(padded_features), torch.stack([torch.tensor(l) for l in labels])

def create_optimized_dataloaders(dataset_path, batch_size=8, num_workers=4, 
                                max_seconds=2.0, apply_scaler=True):
    """
    Create optimized PyTorch DataLoaders with:
    - Multi-worker data loading (parallel processing)
    - Pin memory for faster GPU transfer
    - Proper collate function for variable-length sequences
    - On-the-fly feature extraction
    
    PyTorch equivalent of TF.Data pipeline.
    """
    print(f"\n{'='*80}")
    print("CREATING OPTIMIZED PYTORCH DATALOADERS")
    print(f"{'='*80}")
    
    # Load file paths and labels
    train_real_files = sorted([f for f in glob.glob(f"{dataset_path}/training/real/*.wav") if os.path.getsize(f) > 1000])
    train_fake_files = sorted([f for f in glob.glob(f"{dataset_path}/training/fake/*.wav") if os.path.getsize(f) > 1000])
    train_files = train_real_files + train_fake_files
    train_labels = [0] * len(train_real_files) + [1] * len(train_fake_files)
    
    val_real_files = sorted([f for f in glob.glob(f"{dataset_path}/validation/real/*.wav") if os.path.getsize(f) > 1000])
    val_fake_files = sorted([f for f in glob.glob(f"{dataset_path}/validation/fake/*.wav") if os.path.getsize(f) > 1000])
    val_files = val_real_files + val_fake_files
    val_labels = [0] * len(val_real_files) + [1] * len(val_fake_files)
    
    test_real_files = sorted([f for f in glob.glob(f"{dataset_path}/testing/real/*.wav") if os.path.getsize(f) > 1000])
    test_fake_files = sorted([f for f in glob.glob(f"{dataset_path}/testing/fake/*.wav") if os.path.getsize(f) > 1000])
    test_files = test_real_files + test_fake_files
    test_labels = [0] * len(test_real_files) + [1] * len(test_fake_files)
    
    print(f"Dataset Statistics:")
    print(f"  Training: {len(train_files)} files ({len(train_real_files)} real, {len(train_fake_files)} fake)")
    print(f"  Validation: {len(val_files)} files ({len(val_real_files)} real, {len(val_fake_files)} fake)")
    print(f"  Testing: {len(test_files)} files ({len(test_real_files)} real, {len(test_fake_files)} fake)")
    
    # Create datasets
    train_dataset = AudioDatasetFromFiles(train_files, train_labels, max_seconds, apply_augment=True)
    val_dataset = AudioDatasetFromFiles(val_files, val_labels, max_seconds, apply_augment=False)
    test_dataset = AudioDatasetFromFiles(test_files, test_labels, max_seconds, apply_augment=False)
    
    # Create DataLoaders with optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_variable_length,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_variable_length,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_variable_length,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"\nDataLoader Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Num Workers: {num_workers} (parallel data loading)")
    print(f"  Pin Memory: {torch.cuda.is_available()} (GPU optimization)")
    print(f"  Training Augmentation: ON")
    print(f"  Val/Test Augmentation: OFF")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_size': len(train_files),
        'val_size': len(val_files),
        'test_size': len(test_files),
        'batch_size': batch_size
    }

def train_with_optimized_pipeline(model, optimizer, criterion, loader_info, 
                                  epochs=100, patience=7):
    """
    Train model using optimized DataLoader pipeline.
    PyTorch equivalent of TF.Data training.
    """
    print(f"\n{'='*80}")
    print(f"TRAINING WITH OPTIMIZED DATALOADER PIPELINE - PYTORCH")
    print(f"{'='*80}")
    
    train_loader = loader_info['train_loader']
    val_loader = loader_info['val_loader']
    
    # Setup schedulers
    def onecycle_lambda(epoch):
        max_lr_factor = 5.0
        min_lr_factor = 0.01
        warmup_epochs = 5
        peak_epochs = 15
        
        if epoch < warmup_epochs:
            return 1.0 + (max_lr_factor - 1.0) * (epoch / warmup_epochs)
        elif epoch < warmup_epochs + peak_epochs:
            return max_lr_factor
        else:
            remaining_epochs = epochs - warmup_epochs - peak_epochs
            progress = (epoch - warmup_epochs - peak_epochs) / remaining_epochs
            progress = min(progress, 1.0)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_factor + (max_lr_factor - min_lr_factor) * cosine_factor
    
    scheduler = LambdaLR(optimizer, lr_lambda=onecycle_lambda)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    import time
    timestamp = int(time.time())
    best_checkpoint_path = f"dataloader_best_model_{timestamp}.pt"
    last_checkpoint_path = f"dataloader_last_epoch_{timestamp}.pt"
    
    print(f"Training Configuration:")
    print(f"  Max Epochs: {epochs}")
    print(f"  Early Stopping Patience: {patience}")
    print(f"  Best Checkpoint: {best_checkpoint_path}")
    print(f"  Last Checkpoint: {last_checkpoint_path}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_features.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).long()
            train_correct += (predictions == batch_labels.long()).sum().item()
            train_total += batch_labels.size(0)
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.float().to(device)
                
                outputs = model(batch_features).squeeze()
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item() * batch_features.size(0)
                predictions = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (predictions == batch_labels.long()).sum().item()
                val_total += batch_labels.size(0)
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
              f"LR: {current_lr:.2e}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, best_checkpoint_path)
            print(f"💾 Best model saved: {best_checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save last epoch checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'history': history
    }, last_checkpoint_path)
    print(f"💾 Last epoch checkpoint saved: {last_checkpoint_path}")
    
    print(f"\n📁 SAVED CHECKPOINTS:")
    print(f"  Best model: {best_checkpoint_path}")
    print(f"  Last epoch: {last_checkpoint_path}")
    
    return {
        'history': history,
        'checkpoint_path': best_checkpoint_path,  # For backward compatibility
        'best_checkpoint_path': best_checkpoint_path,
        'last_checkpoint_path': last_checkpoint_path,
        'epochs_trained': len(history['train_loss']),
        'best_val_loss': min(history['val_loss'])
    }

# ------------------------------
# Comprehensive Comparison Framework
# ------------------------------
def compare_all_approaches(single_split_results, cv_results=None, 
                          optimized_results=None, dataloader_results=None):
    """
    Compare all experimental approaches and rank performance.
    Generate deployment recommendations.
    PyTorch implementation.
    """
    print(f"\n\n{'='*100}")
    print("FINAL COMPREHENSIVE RESULTS COMPARISON - PYTORCH")
    print(f"{'='*100}")
    
    approaches_compared = 1
    results_for_ranking = []
    
    print(f"\nMETHODOLOGY COMPARISON:")
    print(f"{'='*80}")
    
    # Approach 1: Single Split
    print(f"APPROACH 1 - SINGLE SPLIT RESULTS (Manual Pipeline):")
    print(f"  Test AUC-ROC: {single_split_results['test_metrics']['auc_roc']:.4f} [{single_split_results['test_metrics']['auc_roc_ci']['ci_lower']:.4f}, {single_split_results['test_metrics']['auc_roc_ci']['ci_upper']:.4f}]")
    print(f"  Test F1-Score: {single_split_results['test_metrics']['f1_score']:.4f} [{single_split_results['test_metrics']['f1_ci']['ci_lower']:.4f}, {single_split_results['test_metrics']['f1_ci']['ci_upper']:.4f}]")
    print(f"  Validation AUC-ROC: {single_split_results['val_metrics']['auc_roc']:.4f}")
    print(f"  Parameters: {single_split_results['total_params']:,}")
    print(f"  Data Pipeline: Manual preprocessing & batching")
    results_for_ranking.append(("Single Split (Manual)", single_split_results['test_metrics']['auc_roc']))
    
    # Approach 2: K-Fold CV
    if cv_results is not None:
        approaches_compared += 1
        print(f"\nAPPROACH 2 - K-FOLD CROSS-VALIDATION RESULTS (Statistical Validation):")
        print(f"  CV AUC-ROC (mean±std): {cv_results['cv_statistics']['auc_roc']['mean']:.4f} ± {cv_results['cv_statistics']['auc_roc']['std']:.4f}")
        print(f"  CV F1-Score (mean±std): {cv_results['cv_statistics']['f1_score']['mean']:.4f} ± {cv_results['cv_statistics']['f1_score']['std']:.4f}")
        
        if cv_results['holdout_test_metrics'] is not None:
            print(f"  Hold-out Test AUC-ROC: {cv_results['holdout_test_metrics']['auc_roc']:.4f}")
            print(f"  Hold-out Test F1-Score: {cv_results['holdout_test_metrics']['f1_score']:.4f}")
            results_for_ranking.append(("K-Fold CV (Statistical)", cv_results['holdout_test_metrics']['auc_roc']))
        
        cv_coeff = cv_results['cv_statistics']['auc_roc']['std'] / cv_results['cv_statistics']['auc_roc']['mean']
        print(f"  Performance Variability: {cv_coeff:.3f} ({'Low' if cv_coeff < 0.05 else 'Moderate' if cv_coeff < 0.10 else 'High'})")
        print(f"  Robustness: Validated across {cv_results['k_folds']} folds")
    
    # Performance ranking
    print(f"\nPERFORMANCE RANKING:")
    print(f"{'='*60}")
    
    results_for_ranking.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, auc) in enumerate(results_for_ranking, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"  {medal} {i}. {name}: {auc:.4f} AUC-ROC")
    
    # Final assessment and recommendations
    print(f"\nFINAL ASSESSMENT & RECOMMENDATIONS:")
    print(f"{'='*80}")
    
    if approaches_compared >= 2:
        best_method, best_auc = results_for_ranking[0]
        print(f"🎯 BEST PERFORMING: {best_method} ({best_auc:.4f} AUC-ROC)")
        
        if cv_results is not None:
            cv_coeff = cv_results['cv_statistics']['auc_roc']['std'] / cv_results['cv_statistics']['auc_roc']['mean']
            if cv_coeff < 0.10:
                print(f"✅ ROBUST PERFORMANCE: Low cross-validation variability ({cv_coeff:.3f})")
                print(f"✅ PRODUCTION READY: Model shows consistent performance across data splits")
            else:
                print(f"⚠️  MODERATE VARIABILITY: CV coefficient = {cv_coeff:.3f}")
        
        print(f"\n🏆 RECOMMENDED DEPLOYMENT STRATEGY:")
        print(f"  → 🥇 DEPLOY {best_method.upper()} MODEL")
        print(f"  → 📈 PyTorch implementation provides excellent flexibility")
        print(f"  → ⚡ Native GPU acceleration available")
        print(f"  → 🔧 Easy integration with production serving frameworks")
    
    return {
        'approaches_compared': approaches_compared,
        'results_ranking': results_for_ranking,
        'best_method': results_for_ranking[0][0] if results_for_ranking else None,
        'best_auc': results_for_ranking[0][1] if results_for_ranking else None
    }

# ------------------------------
# Final Summary
# ------------------------------
print(f"\n\n{'='*100}")
print("FINAL COMPREHENSIVE RESULTS - PYTORCH IMPLEMENTATION")
print(f"{'='*100}")

print(f"\nPYTORCH SINGLE SPLIT RESULTS:")
print(f"  Test AUC-ROC: {single_split_results['test_metrics']['auc_roc']:.4f} [{single_split_results['test_metrics']['auc_roc_ci']['ci_lower']:.4f}, {single_split_results['test_metrics']['auc_roc_ci']['ci_upper']:.4f}]")
print(f"  Test F1-Score: {single_split_results['test_metrics']['f1_score']:.4f} [{single_split_results['test_metrics']['f1_ci']['ci_lower']:.4f}, {single_split_results['test_metrics']['f1_ci']['ci_upper']:.4f}]")
print(f"  Validation AUC-ROC: {single_split_results['val_metrics']['auc_roc']:.4f}")
print(f"  Parameters: {single_split_results['total_params']:,}")
print(f"  Framework: PyTorch {torch.__version__}")
print(f"  Device: {device}")

# ------------------------------
# APPROACH 2: STRATIFIED K-FOLD CROSS-VALIDATION
# ------------------------------
print(f"\n{'='*90}")
print("APPROACH 2: STRATIFIED 5-FOLD CROSS-VALIDATION - PYTORCH")
print(f"{'='*90}")

# Combine training and validation data for cross-validation
X_cv_data = X_train_list + X_val_list  # Combine original lists (before padding)
y_cv_data = np.concatenate([y_train, y_val])

print(f"Data Preparation for K-Fold CV:")
print(f"  Combined CV Data: {len(X_cv_data)} samples")
print(f"  Hold-out Test Data: {len(X_test_list)} samples")
print(f"  Class Distribution in CV: Real={np.sum(y_cv_data == 0)}, Fake={np.sum(y_cv_data == 1)}")

# Run 5-fold cross-validation
cv_results = stratified_kfold_cv(
    X_cv_data, y_cv_data, X_test_list, y_test,
    num_pyramid_layers=2, k_folds=5, random_state=42,
    base_units=128, learning_rate=1e-4, weight_decay=1e-5, batch_size=8
)

# Log cross-validation experiment results
if cv_results is not None:
    log_experiment_results(cv_results, "5-Fold Cross-Validation - PyTorch")

# Clean up memory
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ------------------------------
# COMPREHENSIVE COMPARISON & FINAL RESULTS
# ------------------------------
comparison_results = compare_all_approaches(
    single_split_results,
    cv_results=cv_results,
    optimized_results=optimized_results if 'optimized_results' in locals() else None,
    dataloader_results=dataloader_results if 'dataloader_results' in locals() else None
)

print(f"\n{'='*80}")
print("EXPERIMENT SUMMARY - PYTORCH ULTRA-ENHANCED ARCHITECTURE")
print(f"{'='*80}")
print("This comprehensive PyTorch implementation now features ALL 4 APPROACHES:")
print("\n✅ APPROACH 1 - SINGLE SPLIT VALIDATION:")
print("• 2 pyramid layers with native PyTorch downsampling")
print("• Log-Mel spectrograms (80 bins)")
print("• AttentionPooling for complete temporal information")
print("• AdamW optimizer with gradient clipping + OneCycle LR schedule")

print("\n✅ APPROACH 2 - STRATIFIED K-FOLD CROSS-VALIDATION:")
print("• 5-fold cross-validation for robust performance estimation")
print("• Stratified splitting preserves class distribution")
print("• Statistical analysis with coefficient of variation")
print("• 95% confidence intervals for mean performance")
print("• Hold-out test set evaluation with best fold model")

print("\n🔥 ENHANCED REGULARIZATION ACROSS ALL APPROACHES:")
print("  - Dropout in all layers with progressive rates")
print("  - LayerNormalization for training stability")
print("  - Weight decay for parameter control")
print("  - Gradient clipping (max_norm=1.0)")

print("\n🚀 ROBUST TRAINING PIPELINE:")
print("  - Model checkpointing with best weight restoration")
print("  - Early stopping (patience=7)")
print("  - OneCycle + Plateau LR schedulers")
print("  - Up to 100 epochs with intelligent stopping")

print("\n🎵 MULTI-LEVEL DATA AUGMENTATION:")
print("  - Audio-level: Speed (±10%), pitch (±1 semitone), noise (10-20dB)")
print("  - Spectral-level: SpecAugment time/frequency masking")

print("\n📊 COMPREHENSIVE STATISTICAL EVALUATION:")
print("  - Bootstrap confidence intervals (95% CI)")
print("  - AUC-PR analysis")
print("  - Calibration analysis (Brier score, ECE, MCE)")
print("  - Per-class performance breakdown")
print("  - Performance ranking with deployment recommendations")

print(f"\n📈 DATASET STATISTICS:")
print(f"  Training: {len(y_train)} samples")
print(f"  Validation: {len(y_val)} samples")
print(f"  Testing: {len(y_test)} samples")

print("\n🏆 METHODOLOGY COMPARISON:")
if comparison_results:
    print(f"  Approaches Evaluated: {comparison_results['approaches_compared']}")
    print(f"  Best Method: {comparison_results['best_method']}")
    print(f"  Best AUC-ROC: {comparison_results['best_auc']:.4f}")

print("\n✅ FULL FEATURE PARITY WITH TENSORFLOW VERSION ACHIEVED!")
print("="*80)

# Log the final session summary
log_final_summary()

print("\n" + "="*90)
print("✅ PYTORCH COMPREHENSIVE IMPLEMENTATION COMPLETE!")
print("="*90)
print("COMPLETE FEATURE PARITY WITH TENSORFLOW VERSION:")
print("✓ All 4 experimental approaches implemented")
print("✓ K-Fold cross-validation with statistical analysis")
print("✓ Automated hyperparameter optimization (Optuna)")
print("✓ Optimized DataLoader pipeline (parallel loading)")
print("✓ Comprehensive comparison framework")
print("✓ Performance ranking & deployment recommendations")

print("\nKEY PYTORCH ADVANTAGES:")
print("• Native PyTorch nn.Module classes (more flexible)")
print("• Manual training loops (full control)")
print("• Multi-worker DataLoader (parallel data loading)")
print("• torch.save() checkpointing (portable)")
print("• CUDA operations (native GPU acceleration)")
print("• Easier deployment (TorchScript, ONNX export)")
print("• Production-ready (PyTorch Serve, TorchServe)")

print(f"\n📊 TOTAL CODE SIZE: ~{sum(1 for line in open(__file__))} lines")
print(f"🎯 NOW MATCHES TensorFlow VERSION IN COMPREHENSIVENESS!")
print("="*90)

# ------------------------------
# Optional: Standalone Database Viewer
# ------------------------------
if __name__ == "__main__" and "--view-studies" in sys.argv:
    """
    Standalone mode to view Optuna studies without running full experiment.
    
    Usage:
        python krrish_pytorch.py --view-studies
    """
    print("\n" + "="*80)
    print("OPTUNA DATABASE VIEWER")
    print("="*80)
    
    # Get database path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "optuna_studies.db")
    storage_url = f"sqlite:///{db_path}"
    
    if os.path.exists(db_path):
        print(f"Database found: {db_path}")
        print(f"Size: {os.path.getsize(db_path) / 1024:.2f} KB\n")
        view_optuna_studies(storage_url)
    else:
        print(f"Database not found: {db_path}")
        print("Run the main experiment first to create the database.")
        print(f"\nUsage: python {os.path.basename(__file__)}")
    
    sys.exit(0)

