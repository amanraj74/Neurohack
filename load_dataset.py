import scipy.io
import numpy as np
import pandas as pd
import os

# Your dataset path
dataset_path = r'D:\hackathon\neurohack\DREAMER.mat'

print("Loading DREAMER dataset...")
data = scipy.io.loadmat(dataset_path)

# Extract the main DREAMER structure
dreamer = data['DREAMER'][0, 0]

print("="*80)
print("DREAMER DATASET STRUCTURE")
print("="*80)

# Show all fields in the dataset
print("\nMain fields available:")
for field_name in dreamer.dtype.names:
    field_data = dreamer[field_name]
    print(f"  - {field_name}: shape {field_data.shape}, type {type(field_data)}")

# Access the data structure
subjects_data = dreamer['Data'][0]
num_subjects = len(subjects_data)

print(f"\n✓ Total Participants: {num_subjects}")

# Explore first subject
first_subject = subjects_data[0]
print(f"\nFirst participant contains:")
for field_name in first_subject.dtype.names:
    field_data = first_subject[field_name]
    print(f"  - {field_name}: shape {field_data.shape}")

# Access actual EEG data from first subject
eeg_data = first_subject['EEG'][0, 0]
print(f"\nEEG structure:")
for field_name in eeg_data.dtype.names:
    field_data = eeg_data[field_name]
    print(f"  - {field_name}: shape {field_data.shape}")

# Check if there are emotion ratings
if 'valence' in eeg_data.dtype.names:
    valence = eeg_data['valence'][0]
    arousal = eeg_data['arousal'][0]
    dominance = eeg_data['dominance'][0]
    
    print(f"\n✓ Emotion Ratings Found:")
    print(f"  - Valence: {valence.shape} (1-5 scale)")
    print(f"  - Arousal: {arousal.shape} (1-5 scale)")
    print(f"  - Dominance: {dominance.shape} (1-5 scale)")

print("\n" + "="*80)
print("✓ Dataset exploration complete!")
print("="*80)
