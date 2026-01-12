"""
MAIN EXECUTION SCRIPT - EEG EMOTION RECOGNITION
DREAMER Dataset Hackathon Solution

This script runs the complete pipeline:
- Phase 1: Exploratory Analysis (10 pts)
- Phase 2: EEG Preprocessing (40 pts)
- Phase 3: Feature Extraction (15 pts)
- Phase 4: Classification (20 pts)
- Phase 5: Bonus Analysis (15 pts)

Total: 100 points
"""

import sys
import matplotlib
matplotlib.use('Agg')  # Must be before importing other modules

# NOW add the rest
import os
import time
import pandas as pd
from src.data_loader import DREAMERDataLoader
from src.phase1_exploratory import ExploratoryAnalysis
from src.phase2_preprocessing import EEGPreprocessor
from src.phase3_feature_extraction import FeatureExtractor
from src.phase4_classification import EmotionClassifier
from src.phase5_bonus import BonusAnalysis

import numpy as np
import pandas as pd
from tqdm import tqdm


def print_header(text):
    """Print fancy header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def main():
    """
    Execute complete pipeline
    """
    start_time = time.time()
    
    print_header("🧠 EEG EMOTION RECOGNITION - COMPLETE PIPELINE 🧠")
    print("\nThis script will execute all 5 phases of the hackathon solution.")
    print("Estimated time: 45-60 minutes (depending on your system)")
    print("\nPhases:")
    print("  1. Exploratory Analysis        (10 points)")
    print("  2. EEG Preprocessing          (40 points - MOST IMPORTANT)")
    print("  3. Feature Extraction         (15 points)")
    print("  4. Emotion Classification     (20 points)")
    print("  5. Bonus Analysis             (15 points)")
    print("="*80)
    
    input("\nPress ENTER to start the pipeline...")
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    DATASET_PATH = r'D:\hackathon\neurohack\data\DREAMER.mat'
    
    # Preprocessing settings
    PREPROCESS_ALL_TRIALS = True  # Set to True to preprocess all 414 trials (takes ~3-4 hours)
    NUM_TRIALS_TO_PREPROCESS = 5   # Number of trials for demo (if PREPROCESS_ALL_TRIALS=False)
    
    # ========================================================================
    # PHASE 0: DATA LOADING
    # ========================================================================
    print_header("PHASE 0: LOADING DREAMER DATASET")
    
    loader = DREAMERDataLoader(DATASET_PATH)
    data = loader.extract_all_data()
    
    print(f"\n✓ Loaded: {len(data['eeg_signals'])} participants, {len(data['emotion_ratings'])} trials")
    
    # ========================================================================
    # PHASE 1: EXPLORATORY ANALYSIS (10 POINTS)
    # ========================================================================
    print_header("PHASE 1: EXPLORATORY ANALYSIS (10 POINTS)")
    
    phase1_start = time.time()
    
    analyzer = ExploratoryAnalysis(data['emotion_ratings'])
    analyzer.run_all()
    
    phase1_time = time.time() - phase1_start
    print(f"\n✓ Phase 1 completed in {phase1_time:.1f} seconds")
    print(f"✓ Check outputs/plots/ for visualizations")
    
    # ========================================================================
    # PHASE 2: EEG PREPROCESSING (40 POINTS)
    # ========================================================================
    print_header("PHASE 2: EEG PREPROCESSING (40 POINTS - MOST CRITICAL)")
    
    phase2_start = time.time()
    
    preprocessor = EEGPreprocessor(
        sampling_rate=data['sampling_rate'],
        channel_names=data['channel_names']
    )
    
    if PREPROCESS_ALL_TRIALS:
        print("\n⚠️  You selected to preprocess ALL 414 trials!")
        print("   This will take approximately 3-4 hours.")
        response = input("   Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("   Switching to demo mode (5 trials)...")
            PREPROCESS_ALL_TRIALS = False
    
    print(f"\n{'Preprocessing ALL trials' if PREPROCESS_ALL_TRIALS else f'Demo mode: Preprocessing {NUM_TRIALS_TO_PREPROCESS} trials'}")
    print("This demonstrates the complete preprocessing pipeline with visualizations.\n")
    
    preprocessed_signals = []
    
    if PREPROCESS_ALL_TRIALS:
        # Preprocess all trials
        total_trials = sum(len(participant_trials) for participant_trials in data['eeg_signals'])
        
        with tqdm(total=total_trials, desc="Preprocessing", unit="trial") as pbar:
            for p_idx, participant_trials in enumerate(data['eeg_signals']):
                participant_preprocessed = []
                for t_idx, eeg_trial in enumerate(participant_trials):
                    # Only visualize first trial
                    visualize = (p_idx == 0 and t_idx == 0)
                    
                    try:
                        preprocessed, reports = preprocessor.full_preprocessing_pipeline(
                            eeg_trial,
                            participant_id=p_idx + 1,
                            trial_id=t_idx + 1
                        )
                        participant_preprocessed.append(preprocessed)
                    except Exception as e:
                        print(f"\nWarning: Error preprocessing P{p_idx+1}_T{t_idx+1}: {e}")
                        print("Using original signal...")
                        participant_preprocessed.append(eeg_trial)
                    
                    pbar.update(1)
                
                preprocessed_signals.append(participant_preprocessed)
    else:
        # Demo mode: preprocess only first few trials
        print(f"Processing first {NUM_TRIALS_TO_PREPROCESS} trials with full visualizations...")
        
        for trial_idx in range(min(NUM_TRIALS_TO_PREPROCESS, len(data['eeg_signals'][0]))):
            eeg_trial = data['eeg_signals'][0][trial_idx]
            
            preprocessed, reports = preprocessor.full_preprocessing_pipeline(
                eeg_trial,
                participant_id=1,
                trial_id=trial_idx + 1
            )
            
            preprocessed_signals.append(preprocessed)
        
        print(f"\n✓ Demo preprocessing complete!")
        print(f"✓ For full competition submission, set PREPROCESS_ALL_TRIALS = True")
    
    phase2_time = time.time() - phase2_start
    print(f"\n✓ Phase 2 completed in {phase2_time/60:.1f} minutes")
    print(f"✓ Check outputs/preprocessed/plots/ for before/after visualizations")
    
    # ========================================================================
    # PHASE 3: FEATURE EXTRACTION (15 POINTS)
    # ========================================================================
    print_header("PHASE 3: FEATURE EXTRACTION (15 POINTS)")
    
    phase3_start = time.time()
    
    extractor = FeatureExtractor(
        sampling_rate=data['sampling_rate'],
        channel_names=data['channel_names']
    )
    
    # Use preprocessed data if available, otherwise use original
    if len(preprocessed_signals) > 0:
        if PREPROCESS_ALL_TRIALS:
            eeg_data_for_features = preprocessed_signals
        else:
            # For demo mode, use original data for all trials
            # but replace first few with preprocessed
            print("\nNote: Using original (unpreprocessed) data for feature extraction in demo mode.")
            print("For full pipeline, preprocess all trials first.")
            eeg_data_for_features = data['eeg_signals']
    else:
        eeg_data_for_features = data['eeg_signals']
    
    features_df = extractor.extract_features_all_trials(
        eeg_data_for_features,
        data['emotion_ratings']
    )
    
    phase3_time = time.time() - phase3_start
    print(f"\n✓ Phase 3 completed in {phase3_time/60:.1f} minutes")
    print(f"✓ Extracted {len(features_df)} feature vectors")
    print(f"✓ Features saved to: outputs/features/extracted_features.csv")
    
    # ========================================================================
    # PHASE 4: EMOTION CLASSIFICATION (20 POINTS)
    # ========================================================================
    print_header("PHASE 4: EMOTION CLASSIFICATION (20 POINTS)")
    
    phase4_start = time.time()
    
    classifier = EmotionClassifier()
    
    classification_results = classifier.run_full_classification(
        features_df,
        dimensions=['valence', 'arousal', 'dominance']
    )
    
    phase4_time = time.time() - phase4_start
    print(f"\n✓ Phase 4 completed in {phase4_time/60:.1f} minutes")
    print(f"✓ Classification results saved to: outputs/results/")
    
    # ========================================================================
    # PHASE 5: BONUS ANALYSIS (15 POINTS)
    # ========================================================================
    print_header("PHASE 5: BONUS ANALYSIS (15 POINTS)")
    
    phase5_start = time.time()
    
    bonus_analyzer = BonusAnalysis(
        sampling_rate=data['sampling_rate'],
        channel_names=data['channel_names']
    )
    
    bonus_analyzer.run_all(features_df)
    
    phase5_time = time.time() - phase5_start
    print(f"\n✓ Phase 5 completed in {phase5_time/60:.1f} minutes")
    print(f"✓ Bonus analysis saved to: outputs/bonus/")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    total_time = time.time() - start_time
    
    print_header("🎉 COMPLETE PIPELINE FINISHED! 🎉")
    
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Phase 1 (Exploratory):     {phase1_time:.1f}s   ✓")
    print(f"Phase 2 (Preprocessing):   {phase2_time/60:.1f}min ✓")
    print(f"Phase 3 (Features):        {phase3_time/60:.1f}min ✓")
    print(f"Phase 4 (Classification):  {phase4_time/60:.1f}min ✓")
    print(f"Phase 5 (Bonus):           {phase5_time/60:.1f}min ✓")
    print("-"*80)
    print(f"Total Time:                {total_time/60:.1f}min")
    print("="*80)
    
    print("\n📁 OUTPUT STRUCTURE:")
    print("="*80)
    print("outputs/")
    print("├── plots/                    # Phase 1 visualizations")
    print("│   ├── arousal_valence_plot.png")
    print("│   ├── vad_bar_diagrams_*.png")
    print("│   └── vad_correlation_heatmaps.png")
    print("├── preprocessed/plots/       # Phase 2 preprocessing visualizations")
    print("│   ├── filtering_psd_*.png")
    print("│   ├── ica_components_*.png")
    print("│   ├── asr_before_after_*.png")
    print("│   └── car_before_after_*.png")
    print("├── features/                 # Phase 3 features")
    print("│   ├── extracted_features.csv")
    print("│   └── plots/")
    print("├── results/                  # Phase 4 classification results")
    print("│   ├── classification_report.txt")
    print("│   └── plots/")
    print("└── bonus/                    # Phase 5 bonus analysis")
    print("    ├── bonus_analysis_report.txt")
    print("    └── plots/")
    print("="*80)
    
    print("\n📊 KEY RESULTS:")
    print("="*80)
    
    # Print best classification results
    for dimension, results in classification_results.items():
        best_model = max(results['model_results'].keys(), 
                        key=lambda k: results['model_results'][k]['test_f1'])
        best_f1 = results['model_results'][best_model]['test_f1']
        best_acc = results['model_results'][best_model]['test_accuracy']
        
        print(f"\n{dimension.capitalize()}:")
        print(f"  Best Model: {best_model}")
        print(f"  Accuracy:   {best_acc:.4f}")
        print(f"  F1-Score:   {best_f1:.4f}")
    
    print("\n" + "="*80)
    print("🏆 READY FOR PRESENTATION!")
    print("="*80)
    print("\nAll visualizations, reports, and results are ready.")
    print("Review the outputs/ directory and prepare your presentation.")
    print("\nGood luck with your hackathon! 🚀")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
