import scipy.io
import numpy as np
import pandas as pd
import os

class DREAMERDataLoader:
    """
    Loads and structures the DREAMER dataset for emotion recognition
    
    DREAMER Dataset Structure:
    - 23 participants
    - 18 film clips per participant
    - 14 EEG channels + 2 ECG channels
    - Sampling rate: 128 Hz
    - ScoreValence, ScoreArousal, ScoreDominance ratings (1-5 scale)
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.participants = []
        self.eeg_channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
                                   'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        self.ecg_channel_names = ['ECG_L', 'ECG_R']
        self.sampling_rate = 128
        
    def load(self):
        """Load the DREAMER dataset from .mat file"""
        print("="*80)
        print("LOADING DREAMER DATASET")
        print("="*80)
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at: {self.dataset_path}")
        
        file_size_mb = os.path.getsize(self.dataset_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB")
        print("Loading data (this may take a minute)...")
        
        # Load .mat file
        mat_data = scipy.io.loadmat(self.dataset_path)
        
        # Extract DREAMER structure
        self.data = mat_data['DREAMER'][0, 0]
        
        # Extract participant data
        participants_data = self.data['Data'][0]
        
        # Get dataset info
        sampling_rate = self.data['EEG_SamplingRate'][0, 0]
        num_subjects = self.data['noOfSubjects'][0, 0]
        num_videos = self.data['noOfVideoSequences'][0, 0]
        
        print(f"✓ Loaded {len(participants_data)} participants")
        print(f"  Sampling Rate: {sampling_rate} Hz")
        print(f"  Videos per participant: {num_videos}")
        
        self.sampling_rate = int(sampling_rate)
        
        return participants_data
    
    def extract_all_data(self):
        """
        Extract all EEG signals and emotion ratings into structured format
        
        Returns:
        --------
        dict containing:
            - eeg_signals: list of (n_trials, n_channels, n_timepoints) arrays per participant
            - emotion_ratings: DataFrame with valence, arousal, dominance ratings
            - metadata: trial and participant information
        """
        participants_data = self.load()
        
        all_eeg_signals = []
        all_emotion_ratings = []
        all_metadata = []
        
        print("\nExtracting EEG signals and emotion ratings...")
        
        for p_idx, participant in enumerate(participants_data):
            print(f"  Processing Participant {p_idx + 1}/{len(participants_data)}...", end='\r')
            
            # Extract EEG structure for this participant
            eeg_struct = participant['EEG'][0, 0]
            
            # Get baseline and stimuli data
            baseline = eeg_struct['baseline'][0, 0]  # Baseline recording
            stimuli = eeg_struct['stimuli'][0, 0]    # All trial recordings: shape (18, 1)
            
            # Extract emotion ratings from participant level
            valence_data = participant['ScoreValence'][0, 0]
            arousal_data = participant['ScoreArousal'][0, 0]
            dominance_data = participant['ScoreDominance'][0, 0]
            
            # Flatten if needed
            if valence_data.ndim > 1:
                valence_data = valence_data.flatten()
                arousal_data = arousal_data.flatten()
                dominance_data = dominance_data.flatten()
            
            # Store participant's data
            participant_trials = []
            
            # Get number of trials - stimuli.shape[0] = 18 trials
            n_trials = stimuli.shape[0]
            
            for trial_idx in range(n_trials):
                # Get trial data: stimuli[trial_idx, 0] gives (n_timepoints, n_channels)
                # Shape: (25472, 14) = (timepoints, channels)
                trial_data = stimuli[trial_idx, 0]
                
                # Transpose to get (channels, timepoints) format
                # This gives us (14, 25472) = (channels, timepoints)
                trial_data_transposed = trial_data.T
                
                # Now we have proper format: (14 channels, 25472 timepoints)
                eeg_trial_data = trial_data_transposed  # All 14 EEG channels
                
                participant_trials.append(eeg_trial_data)
                
                # Store emotion ratings
                all_emotion_ratings.append({
                    'participant_id': p_idx + 1,
                    'trial_id': trial_idx + 1,
                    'valence': float(valence_data[trial_idx]),
                    'arousal': float(arousal_data[trial_idx]),
                    'dominance': float(dominance_data[trial_idx])
                })
                
                # Store metadata
                all_metadata.append({
                    'participant_id': p_idx + 1,
                    'trial_id': trial_idx + 1,
                    'stimulus_id': f'P{p_idx+1:02d}_T{trial_idx+1:02d}',
                    'n_channels': eeg_trial_data.shape[0],
                    'n_timepoints': eeg_trial_data.shape[1],
                    'duration_sec': eeg_trial_data.shape[1] / self.sampling_rate
                })
            
            all_eeg_signals.append(participant_trials)
        
        print(f"\n✓ Extracted data from {len(participants_data)} participants")
        
        # Create emotion ratings DataFrame
        emotion_df = pd.DataFrame(all_emotion_ratings)
        metadata_df = pd.DataFrame(all_metadata)
        
        # Summary statistics
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        print(f"Total participants: {len(participants_data)}")
        print(f"Trials per participant: {n_trials}")
        print(f"Total trials: {len(emotion_df)}")
        print(f"EEG channels: {len(self.eeg_channel_names)}")
        print(f"Sampling rate: {self.sampling_rate} Hz")
        print(f"\nEmotion Ratings Statistics:")
        print(emotion_df[['valence', 'arousal', 'dominance']].describe())
        
        return {
            'eeg_signals': all_eeg_signals,
            'emotion_ratings': emotion_df,
            'metadata': metadata_df,
            'channel_names': self.eeg_channel_names,
            'sampling_rate': self.sampling_rate
        }
    
    def get_single_participant(self, participant_id):
        """Get data for a single participant"""
        all_data = self.extract_all_data()
        
        participant_eeg = all_data['eeg_signals'][participant_id - 1]
        participant_ratings = all_data['emotion_ratings'][
            all_data['emotion_ratings']['participant_id'] == participant_id
        ]
        
        return {
            'eeg_signals': participant_eeg,
            'emotion_ratings': participant_ratings,
            'channel_names': self.eeg_channel_names,
            'sampling_rate': self.sampling_rate
        }


if __name__ == "__main__":
    dataset_path = r'D:\hackathon\neurohack\data\DREAMER.mat'
    
    loader = DREAMERDataLoader(dataset_path)
    data = loader.extract_all_data()
    
    print("\n✓ Data loader test successful!")
    print(f"Loaded {len(data['eeg_signals'])} participants")
    print(f"Total trials: {len(data['emotion_ratings'])}")
    
    print("\nFirst 5 emotion ratings:")
    print(data['emotion_ratings'].head())
    
    print("\nSample EEG shape (Participant 1, Trial 1):")
    print(f"  Shape: {data['eeg_signals'][0][0].shape} (channels x timepoints)")
    print(f"  Duration: {data['eeg_signals'][0][0].shape[1] / 128:.1f} seconds")
