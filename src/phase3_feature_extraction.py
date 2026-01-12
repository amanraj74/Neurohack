import numpy as np
from scipy import signal
from scipy.integrate import simpson
from scipy.stats import skew, kurtosis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm


class FeatureExtractor:
    """
    Phase 3: ULTIMATE Feature Extraction with BASELINE CORRECTION (15 points)
    
    🏆 TOP-CLASS VERSION with baseline correction + maximum features
    ✅ NOW INCLUDES: Trial_Power - Baseline_Power for subject-specific correction
    """
    
    def __init__(self, sampling_rate=128, channel_names=None, output_dir='outputs/features', 
                 use_baseline_correction=True):
        self.sampling_rate = sampling_rate
        self.channel_names = channel_names or ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
                                                'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        self.output_dir = output_dir
        self.use_baseline_correction = use_baseline_correction
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        # ⭐ EXPANDED BANDS - 5 bands for comprehensive coverage
        self.bands = {
            'delta': (0.5, 4),    # Deep sleep, unconscious processing
            'theta': (4, 8),      # Emotional processing, meditation
            'alpha': (8, 13),     # Calm alertness, relaxation
            'beta': (13, 30),     # Active thinking, focus
            'gamma': (30, 45)     # High-level cognition
        }
        
        # DREAMER baseline period (first 61 seconds)
        self.baseline_duration = 61  # seconds
        
        print("="*80)
        print("PHASE 3: ULTIMATE FEATURE EXTRACTION WITH BASELINE CORRECTION (15 POINTS)")
        print("="*80)
        print(f"⭐ Using {len(self.bands)} frequency bands for maximum accuracy")
        if self.use_baseline_correction:
            print(f"✅ BASELINE CORRECTION: Enabled (first {self.baseline_duration}s = rest period)")
            print(f"   Formula: Corrected_Feature = Trial_Power - Baseline_Power")
        else:
            print(f"⚠️  BASELINE CORRECTION: Disabled (using full trial)")
    
    def extract_baseline_and_trial_segments(self, eeg_data):
        """Split EEG into baseline and trial segments"""
        baseline_samples = int(self.baseline_duration * self.sampling_rate)
        
        if eeg_data.shape[1] <= baseline_samples:
            print(f"   ⚠️  WARNING: Signal too short for baseline extraction. Using full signal.")
            return eeg_data, eeg_data
        
        baseline_segment = eeg_data[:, :baseline_samples]
        trial_segment = eeg_data[:, baseline_samples:]
        
        return baseline_segment, trial_segment
    
    def compute_psd_welch(self, eeg_data, nperseg=None):
        """Compute Power Spectral Density using Welch's method"""
        if nperseg is None:
            nperseg = min(2 * self.sampling_rate, eeg_data.shape[1] // 2)
        
        n_channels = eeg_data.shape[0]
        psd_list = []
        freqs = None
        
        for ch_idx in range(n_channels):
            f, pxx = signal.welch(
                eeg_data[ch_idx, :],
                fs=self.sampling_rate,
                nperseg=nperseg,
                noverlap=nperseg // 2,
                scaling='density'
            )
            psd_list.append(pxx)
            if freqs is None:
                freqs = f
        
        psd = np.array(psd_list)
        return freqs, psd
    
    def compute_baseline_corrected_psd(self, eeg_data):
        """Compute baseline-corrected PSD (Trial - Baseline)"""
        baseline_segment, trial_segment = self.extract_baseline_and_trial_segments(eeg_data)
        freqs_baseline, psd_baseline = self.compute_psd_welch(baseline_segment)
        freqs_trial, psd_trial = self.compute_psd_welch(trial_segment)
        
        # Baseline correction: Trial - Baseline
        psd_corrected = psd_trial - psd_baseline
        return freqs_trial, psd_trial, psd_baseline, psd_corrected
    
    def extract_band_power(self, freqs, psd, band_range):
        """Extract power in a specific frequency band"""
        idx_band = np.logical_and(freqs >= band_range[0], freqs <= band_range[1])
        
        # Handle negative values from baseline correction for integration
        psd_clipped = np.copy(psd)
        psd_clipped[:, idx_band] = np.maximum(psd_clipped[:, idx_band], 1e-12)
        
        band_power = simpson(psd_clipped[:, idx_band], x=freqs[idx_band], axis=1)
        return band_power

    def extract_statistical_features(self, eeg_data):
        """Statistical features from time domain"""
        features = {}
        for ch_idx, ch_name in enumerate(self.channel_names):
            ch_data = eeg_data[ch_idx, :]
            features[f'{ch_name}_mean'] = np.mean(ch_data)
            features[f'{ch_name}_std'] = np.std(ch_data)
            features[f'{ch_name}_skewness'] = skew(ch_data)
            features[f'{ch_name}_kurtosis'] = kurtosis(ch_data)
            features[f'{ch_name}_ptp'] = np.ptp(ch_data)
            features[f'{ch_name}_max'] = np.max(ch_data)
            features[f'{ch_name}_min'] = np.min(ch_data)
        return features

    def extract_asymmetry_features(self, freqs, psd):
        """
        ⭐ Hemispheric Asymmetry (CRITICAL for valence)
        
        ✅ NOW WITH PROPER HANDLING OF BASELINE-CORRECTED VALUES
        """
        features = {}
        
        # Define electrode pairs (left-right)
        pairs = [
            ('F3', 'F4'),   # Frontal (MOST IMPORTANT)
            ('FC5', 'FC6'), # Frontal-Central
            ('T7', 'T8'),   # Temporal
            ('P7', 'P8'),   # Parietal
            ('O1', 'O2'),   # Occipital
            ('AF3', 'AF4')  # Anterior Frontal
        ]
        
        for band_name, band_range in self.bands.items():
            for left_ch, right_ch in pairs:
                if left_ch in self.channel_names and right_ch in self.channel_names:
                    left_idx = self.channel_names.index(left_ch)
                    right_idx = self.channel_names.index(right_ch)
                    
                    left_power = self.extract_band_power(freqs, psd[[left_idx]], band_range)[0]
                    right_power = self.extract_band_power(freqs, psd[[right_idx]], band_range)[0]
                    
                    # ✅ HANDLE NEGATIVE VALUES FROM BASELINE CORRECTION
                    # Use absolute values for log asymmetry
                    left_power_abs = abs(left_power) + 1e-10
                    right_power_abs = abs(right_power) + 1e-10
                    
                    # Log asymmetry (standard in research)
                    asymmetry = np.log(right_power_abs) - np.log(left_power_abs)
                    
                    # Check for NaN
                    if not np.isnan(asymmetry) and not np.isinf(asymmetry):
                        features[f'{left_ch}_{right_ch}_{band_name}_asymmetry'] = asymmetry
                    else:
                        features[f'{left_ch}_{right_ch}_{band_name}_asymmetry'] = 0.0
        
        return features

    def extract_band_ratios(self, features):
        """Band Power Ratios (PROVEN emotion indicators)"""
        ratios = {}
        delta_mean = abs(features.get('delta_power_mean', 1))
        theta_mean = abs(features.get('theta_power_mean', 1))
        alpha_mean = abs(features.get('alpha_power_mean', 1))
        beta_mean = abs(features.get('beta_power_mean', 1))
        gamma_mean = abs(features.get('gamma_power_mean', 1))
        
        eps = 1e-10
        ratios['beta_alpha_ratio'] = beta_mean / (alpha_mean + eps)
        ratios['theta_beta_ratio'] = theta_mean / (beta_mean + eps)
        ratios['theta_alpha_ratio'] = theta_mean / (alpha_mean + eps)
        ratios['alpha_delta_ratio'] = alpha_mean / (delta_mean + eps)
        ratios['gamma_beta_ratio'] = gamma_mean / (beta_mean + eps)
        ratios['high_low_ratio'] = (beta_mean + gamma_mean) / (delta_mean + theta_mean + eps)
        ratios['arousal_index'] = (beta_mean + gamma_mean) / (alpha_mean + theta_mean + eps)
        
        return ratios

    def extract_regional_features(self, features):
        """Regional brain activity features"""
        regional_features = {}
        regions = {
            'frontal': ['AF3', 'F7', 'F3', 'AF4', 'F4', 'F8'],
            'central': ['FC5', 'FC6'],
            'temporal': ['T7', 'T8'],
            'parietal': ['P7', 'P8'],
            'occipital': ['O1', 'O2']
        }
        
        for band_name in self.bands.keys():
            for region_name, channels in regions.items():
                region_powers = []
                for ch in channels:
                    if ch in self.channel_names:
                        key = f'{ch}_{band_name}_power'
                        if key in features:
                            region_powers.append(abs(features[key]))
                
                if len(region_powers) > 0:
                    regional_features[f'{region_name}_{band_name}_mean'] = np.mean(region_powers)
                    regional_features[f'{region_name}_{band_name}_std'] = np.std(region_powers)
        
        return regional_features

    def visualize_baseline_correction(self, freqs, psd_baseline, psd_trial, psd_corrected, 
                                      participant_id, trial_id):
        """Visualize baseline correction effect"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        baseline_avg = np.mean(psd_baseline, axis=0)
        trial_avg = np.mean(psd_trial, axis=0)
        corrected_avg = np.mean(psd_corrected, axis=0)
        
        axes[0].semilogy(freqs, baseline_avg, color='blue', linewidth=2)
        axes[0].set_title('Baseline PSD\n(Rest - First 61s)')
        
        axes[1].semilogy(freqs, trial_avg, color='red', linewidth=2)
        axes[1].set_title('Trial PSD\n(Emotion - After 61s)')
        
        axes[2].plot(freqs, corrected_avg, color='green', linewidth=2)
        axes[2].axhline(y=0, color='black', linestyle='--')
        axes[2].set_title('Baseline-Corrected PSD\n(Trial - Baseline)')
        
        plt.suptitle(f'Baseline Correction Process - P{participant_id:02d}_T{trial_id:02d}')
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'plots', f'baseline_correction_P{participant_id:02d}_T{trial_id:02d}.png')
        plt.savefig(filepath, dpi=300)
        plt.close()

    def visualize_psd(self, freqs, psd, participant_id, trial_id, title_suffix=""):
        """Create PSD visualization"""
        fig, ax = plt.subplots(figsize=(14, 8))
        psd_mean = np.mean(psd, axis=0)
        psd_mean_pos = np.maximum(psd_mean, 1e-12)
        ax.plot(freqs, 10 * np.log10(psd_mean_pos), 'b-', linewidth=2)
        ax.set_title(f'Power Spectral Density{title_suffix} - P{participant_id:02d}_T{trial_id:02d}')
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'plots', f'psd{title_suffix.replace(" ", "_")}_P{participant_id:02d}_T{trial_id:02d}.png')
        plt.savefig(filepath, dpi=300)
        plt.close()

    def visualize_band_powers(self, features, participant_id, trial_id):
        """Create bar plot of band powers"""
        fig, axes = plt.subplots(1, 5, figsize=(24, 5))
        for idx, band_name in enumerate(self.bands.keys()):
            ax = axes[idx]
            band_powers = [abs(features[f'{ch}_{band_name}_power']) for ch in self.channel_names]
            ax.bar(range(len(self.channel_names)), band_powers)
            ax.set_title(f'{band_name.capitalize()}')
            ax.set_xticks(range(len(self.channel_names)))
            ax.set_xticklabels(self.channel_names, rotation=45)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'plots', f'band_powers_P{participant_id:02d}_T{trial_id:02d}.png')
        plt.savefig(filepath, dpi=300)
        plt.close()

    def extract_trial_features(self, eeg_data, participant_id=1, trial_id=1, visualize=False):
        """Extract ALL features from a single trial"""
        if self.use_baseline_correction:
            freqs, psd_trial, psd_baseline, psd_corrected = self.compute_baseline_corrected_psd(eeg_data)
            psd = psd_corrected
            if visualize:
                self.visualize_baseline_correction(freqs, psd_baseline, psd_trial, psd_corrected, participant_id, trial_id)
        else:
            freqs, psd = self.compute_psd_welch(eeg_data)
        
        features = {}
        for band_name, band_range in self.bands.items():
            band_power = self.extract_band_power(freqs, psd, band_range)
            for ch_idx, ch_name in enumerate(self.channel_names):
                features[f'{ch_name}_{band_name}_power'] = band_power[ch_idx]
            features[f'{band_name}_power_mean'] = np.mean(abs(band_power))
            features[f'{band_name}_power_std'] = np.std(band_power)
            features[f'{band_name}_power_max'] = np.max(abs(band_power))
            features[f'{band_name}_power_min'] = np.min(abs(band_power))
            features[f'{band_name}_power_median'] = np.median(abs(band_power))
        
        if self.use_baseline_correction:
            _, trial_segment = self.extract_baseline_and_trial_segments(eeg_data)
            stat_features = self.extract_statistical_features(trial_segment)
        else:
            stat_features = self.extract_statistical_features(eeg_data)
        features.update(stat_features)
        
        features.update(self.extract_asymmetry_features(freqs, psd))
        features.update(self.extract_band_ratios(features))
        features.update(self.extract_regional_features(features))
        
        features['participant_id'] = participant_id
        features['trial_id'] = trial_id
        
        if visualize:
            self.visualize_psd(freqs, psd, participant_id, trial_id)
            self.visualize_band_powers(features, participant_id, trial_id)
        
        return features, freqs, psd

    def extract_features_all_trials(self, all_eeg_data, emotion_ratings):
        """Extract features from all trials"""
        all_features = []
        total_trials = sum(len(p_trials) for p_trials in all_eeg_data)
        
        with tqdm(total=total_trials, desc="Extracting features") as pbar:
            for p_idx, participant_trials in enumerate(all_eeg_data):
                for t_idx, eeg_trial in enumerate(participant_trials):
                    visualize = (p_idx == 0 and t_idx == 0)
                    feat, _, _ = self.extract_trial_features(eeg_trial, p_idx + 1, t_idx + 1, visualize)
                    all_features.append(feat)
                    pbar.update(1)
        
        features_df = pd.DataFrame(all_features)
        features_df = features_df.merge(emotion_ratings, on=['participant_id', 'trial_id'], how='left')
        
        filename_suffix = "_baseline_corrected" if self.use_baseline_correction else ""
        features_df.to_csv(os.path.join(self.output_dir, f'extracted_features{filename_suffix}.csv'), index=False)
        self.create_feature_summary(features_df)
        return features_df

    def create_feature_summary(self, features_df):
        """Create summary statistics and correlation heatmap"""
        feature_cols = [col for col in features_df.columns if col not in ['participant_id', 'trial_id', 'valence', 'arousal', 'dominance']]
        important_features = [col for col in feature_cols if any(x in col for x in ['_mean', 'asymmetry', 'ratio', 'arousal_index'])][:25]
        
        if len(important_features) > 0:
            plt.figure(figsize=(14, 12))
            sns.heatmap(features_df[important_features + ['valence', 'arousal']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Feature-Emotion Correlations')
            plt.savefig(os.path.join(self.output_dir, 'plots', 'feature_emotion_correlation.png'))
            plt.close()


if __name__ == "__main__":
    from src.data_loader import DREAMERDataLoader
    dataset_path = r'D:\hackathon\neurohack\data\DREAMER.mat'
    loader = DREAMERDataLoader(dataset_path)
    data = loader.extract_all_data()
    
    extractor = FeatureExtractor(
        sampling_rate=data['sampling_rate'],
        channel_names=data['channel_names'],
        use_baseline_correction=True
    )
    
    features_df = extractor.extract_features_all_trials(data['eeg_signals'], data['emotion_ratings'])
    print(f"\n✅ Extraction complete! Shape: {features_df.shape}")