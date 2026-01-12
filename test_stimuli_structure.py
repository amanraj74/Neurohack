import scipy.io
import numpy as np

dataset_path = r'D:\hackathon\neurohack\data\DREAMER.mat'
data = scipy.io.loadmat(dataset_path)
dreamer = data['DREAMER'][0, 0]
subjects_data = dreamer['Data'][0]

# Get first participant
first_subject = subjects_data[0]
eeg_struct = first_subject['EEG'][0, 0]
stimuli = eeg_struct['stimuli'][0, 0]

print("Stimuli shape:", stimuli.shape)
print("Stimuli dtype:", stimuli.dtype)
print("\nExploring structure:")

# Try different access patterns
print("\n1. stimuli[0, 0] shape:", stimuli[0, 0].shape if stimuli.shape[1] > 0 else "N/A")

if stimuli.shape[1] > 0:
    first_trial = stimuli[0, 0]
    print("2. First trial shape:", first_trial.shape)
    print("3. First trial dtype:", first_trial.dtype)
    
    # If it's 2D, show dimensions
    if first_trial.ndim == 2:
        print(f"4. First trial is 2D: {first_trial.shape[0]} channels x {first_trial.shape[1]} timepoints")
    
# Check how many trials
print("\n5. Number of trials (stimuli.shape[1]):", stimuli.shape[1])

# Try accessing multiple trials
for i in range(min(3, stimuli.shape[1])):
    trial = stimuli[0, i]
    print(f"   Trial {i}: shape = {trial.shape}")
