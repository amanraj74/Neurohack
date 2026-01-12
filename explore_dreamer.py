import scipy.io
import numpy as np

dataset_path = r'D:\hackathon\neurohack\data\DREAMER.mat'

print("Loading DREAMER dataset...")
data = scipy.io.loadmat(dataset_path)

# Extract DREAMER structure
dreamer = data['DREAMER'][0, 0]

print("\n" + "="*80)
print("MAIN DREAMER STRUCTURE")
print("="*80)
for field_name in dreamer.dtype.names:
    print(f"  - {field_name}")

# Get the data
subjects_data = dreamer['Data'][0]
print(f"\nTotal participants: {len(subjects_data)}")

# Explore first participant
first_subject = subjects_data[0]
print("\n" + "="*80)
print("FIRST PARTICIPANT STRUCTURE")
print("="*80)
for field_name in first_subject.dtype.names:
    field_data = first_subject[field_name]
    print(f"  - {field_name}: type={type(field_data)}, shape={field_data.shape}")

# Dive into EEG structure
eeg_data = first_subject['EEG'][0, 0]
print("\n" + "="*80)
print("EEG DATA STRUCTURE")
print("="*80)
for field_name in eeg_data.dtype.names:
    field_data = eeg_data[field_name]
    print(f"  - {field_name}: shape={field_data.shape}")
    
    # Try to print shape of nested data
    if hasattr(field_data, 'dtype') and field_data.dtype.names:
        print(f"    Nested fields:")
        for nested_field in field_data.dtype.names:
            print(f"      - {nested_field}")

# Check where valence/arousal/dominance are stored
print("\n" + "="*80)
print("SEARCHING FOR EMOTION RATINGS...")
print("="*80)

# Check if ratings are at participant level
if 'ScoreValence' in first_subject.dtype.names:
    print("✓ Found 'ScoreValence' at participant level")
    valence = first_subject['ScoreValence']
    print(f"  ScoreValence shape: {valence.shape}")
    
if 'ScoreArousal' in first_subject.dtype.names:
    print("✓ Found 'ScoreArousal' at participant level")
    arousal = first_subject['ScoreArousal']
    print(f"  ScoreArousal shape: {arousal.shape}")
    
if 'ScoreDominance' in first_subject.dtype.names:
    print("✓ Found 'ScoreDominance' at participant level")
    dominance = first_subject['ScoreDominance']
    print(f"  ScoreDominance shape: {dominance.shape}")

# Print sample values
print("\n" + "="*80)
print("SAMPLE EMOTION RATINGS (First 5 trials)")
print("="*80)
if 'ScoreValence' in first_subject.dtype.names:
    valence_vals = first_subject['ScoreValence'][0]
    arousal_vals = first_subject['ScoreArousal'][0]
    dominance_vals = first_subject['ScoreDominance'][0]
    
    for i in range(min(5, len(valence_vals))):
        print(f"Trial {i+1}: Valence={valence_vals[i]:.2f}, Arousal={arousal_vals[i]:.2f}, Dominance={dominance_vals[i]:.2f}")
