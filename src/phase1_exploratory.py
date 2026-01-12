import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os

class ExploratoryAnalysis:
    """
    Phase 1: Exploratory Analysis & Emotional Space Visualization (10 points)
    
    Required Outputs:
    1. 2D Arousal-Valence Plot (4 pts)
    2. Multiple Bar Diagrams for VAD (3 pts)
    3. Correlation Analysis (3 pts)
    """
    
    def __init__(self, emotion_data, output_dir='outputs/plots'):
        """
        Parameters:
        -----------
        emotion_data : pandas DataFrame
            Must contain columns: 'participant_id', 'trial_id', 'valence', 'arousal', 'dominance'
        """
        self.emotion_data = emotion_data
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set publication-quality style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.5)
        sns.set_palette("husl")
        
        print("="*80)
        print("PHASE 1: EXPLORATORY ANALYSIS & VISUALIZATION")
        print("="*80)
    
    def create_arousal_valence_plot(self):
        """
        Task 1: Create 2D Arousal-Valence Plot (4 points)
        
        Judges look for:
        - Correct identification of Circumplex Model shape
        - Clear quadrant labels
        - Interpretation of clustering and spread
        """
        print("\n[1/3] Creating Arousal-Valence Plot...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        valence = self.emotion_data['valence'].values
        arousal = self.emotion_data['arousal'].values
        dominance = self.emotion_data['dominance'].values
        
        # Create scatter plot with dominance as color
        scatter = ax.scatter(valence, arousal, c=dominance, 
                           cmap='viridis', s=100, alpha=0.6, 
                           edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Dominance Level', fontsize=14, fontweight='bold')
        
        # Add median lines to divide quadrants
        valence_median = np.median(valence)
        arousal_median = np.median(arousal)
        
        ax.axhline(y=arousal_median, color='red', linestyle='--', 
                  alpha=0.5, linewidth=2, label=f'Median Arousal = {arousal_median:.2f}')
        ax.axvline(x=valence_median, color='blue', linestyle='--', 
                  alpha=0.5, linewidth=2, label=f'Median Valence = {valence_median:.2f}')
        
        # Label emotional quadrants with emoji-like labels
        quadrant_props = dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.8)
        
        ax.text(0.98, 0.98, '😊 EXCITED\n(High Arousal\nPositive Valence)', 
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=quadrant_props)
        
        ax.text(0.02, 0.98, '😰 STRESSED\n(High Arousal\nNegative Valence)', 
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top', horizontalalignment='left',
                bbox=quadrant_props)
        
        ax.text(0.98, 0.02, '😌 RELAXED\n(Low Arousal\nPositive Valence)', 
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='bottom', horizontalalignment='right',
                bbox=quadrant_props)
        
        ax.text(0.02, 0.02, '😔 BORED\n(Low Arousal\nNegative Valence)', 
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='bottom', horizontalalignment='left',
                bbox=quadrant_props)
        
        ax.set_xlabel('Valence (Negative ← → Positive)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Arousal (Calm ← → Excited)', fontsize=16, fontweight='bold')
        ax.set_title('2D Arousal-Valence Emotional Space\n(Circumplex Model)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.5, 5.5])
        ax.set_ylim([0.5, 5.5])
        
        plt.tight_layout()
        filename = 'arousal_valence_plot.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")
        
        # Calculate quadrant distribution for interpretation
        high_arousal_pos = ((arousal > arousal_median) & (valence > valence_median)).sum()
        high_arousal_neg = ((arousal > arousal_median) & (valence <= valence_median)).sum()
        low_arousal_pos = ((arousal <= arousal_median) & (valence > valence_median)).sum()
        low_arousal_neg = ((arousal <= arousal_median) & (valence <= valence_median)).sum()
        total = len(valence)
        
        interpretation = {
            'Excited (High+/Pos+)': f'{high_arousal_pos} ({100*high_arousal_pos/total:.1f}%)',
            'Stressed (High+/Neg-)': f'{high_arousal_neg} ({100*high_arousal_neg/total:.1f}%)',
            'Relaxed (Low-/Pos+)': f'{low_arousal_pos} ({100*low_arousal_pos/total:.1f}%)',
            'Bored (Low-/Neg-)': f'{low_arousal_neg} ({100*low_arousal_neg/total:.1f}%)'
        }
        
        return interpretation
    
    def create_vad_bar_diagrams(self):
        """
        Task 2: Multiple Bar Diagrams for VAD (3 points)
        
        Judges look for:
        - Clear labels and trends
        - Proper error bars (std dev)
        - Meaningful interpretation
        """
        print("\n[2/3] Creating VAD Bar Diagrams...")
        
        # Create bar diagrams grouped by participant
        fig, axes = plt.subplots(3, 1, figsize=(16, 14))
        
        vad_dimensions = ['valence', 'arousal', 'dominance']
        vad_labels = ['Valence (Negative-Positive)', 
                      'Arousal (Calm-Excited)', 
                      'Dominance (Submissive-Dominant)']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for idx, (dimension, label, color) in enumerate(zip(vad_dimensions, vad_labels, colors)):
            ax = axes[idx]
            
            # Calculate mean and std for each participant
            grouped = self.emotion_data.groupby('participant_id')[dimension].agg(['mean', 'std', 'count'])
            
            x_pos = np.arange(len(grouped))
            bars = ax.bar(x_pos, grouped['mean'], 
                         yerr=grouped['std'], 
                         capsize=4, alpha=0.75, color=color, 
                         edgecolor='black', linewidth=1.2)
            
            # Add value labels on bars
            for i, (bar, mean_val, std_val) in enumerate(zip(bars, grouped['mean'], grouped['std'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.05,
                       f'{mean_val:.2f}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
            
            # Add horizontal line for dataset mean
            overall_mean = self.emotion_data[dimension].mean()
            ax.axhline(y=overall_mean, color='red', linestyle=':', 
                      linewidth=2, alpha=0.7, 
                      label=f'Overall Mean = {overall_mean:.2f}')
            
            ax.set_xlabel('Participant ID', fontsize=13, fontweight='bold')
            ax.set_ylabel(f'{label}\nRating', fontsize=13, fontweight='bold')
            ax.set_title(f'{label} Distribution Across Participants', 
                        fontsize=15, fontweight='bold', pad=15)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(grouped.index, rotation=0)
            ax.legend(loc='upper right', fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim([0, 5.5])
        
        plt.tight_layout()
        filename = 'vad_bar_diagrams_participants.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")
        
        # Also create stimulus-wise bar diagram
        fig, axes = plt.subplots(3, 1, figsize=(18, 14))
        
        for idx, (dimension, label, color) in enumerate(zip(vad_dimensions, vad_labels, colors)):
            ax = axes[idx]
            
            # Calculate mean across all participants for each stimulus
            grouped = self.emotion_data.groupby('trial_id')[dimension].agg(['mean', 'std'])
            
            x_pos = np.arange(len(grouped))
            bars = ax.bar(x_pos, grouped['mean'], 
                         yerr=grouped['std'], 
                         capsize=3, alpha=0.75, color=color, 
                         edgecolor='black', linewidth=1.0)
            
            # Highlight extreme stimuli
            max_idx = grouped['mean'].idxmax()
            min_idx = grouped['mean'].idxmin()
            
            bars[max_idx - 1].set_color('gold')
            bars[min_idx - 1].set_color('gray')
            
            ax.set_xlabel('Stimulus/Trial ID', fontsize=13, fontweight='bold')
            ax.set_ylabel(f'{label}\nRating', fontsize=13, fontweight='bold')
            ax.set_title(f'{label} Distribution Across Stimuli/Trials\n(Gold=Highest, Gray=Lowest)', 
                        fontsize=15, fontweight='bold', pad=15)
            ax.set_xticks(x_pos[::2])  # Show every 2nd label to avoid crowding
            ax.set_xticklabels(grouped.index[::2], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim([0, 5.5])
        
        plt.tight_layout()
        filename = 'vad_bar_diagrams_stimuli.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")
    
    def correlation_analysis(self):
        """
        Task 3: Correlation Analysis (3 points)
        
        Judges look for:
        - Correct calculation of Pearson/Spearman correlations
        - Statistical significance testing
        - Meaningful interpretation of emotional structure
        """
        print("\n[3/3] Performing Correlation Analysis...")
        
        vad_data = self.emotion_data[['valence', 'arousal', 'dominance']]
        
        # Calculate both Pearson and Spearman correlations
        pearson_corr = vad_data.corr(method='pearson')
        spearman_corr = vad_data.corr(method='spearman')
        
        # Create figure with two heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Pearson correlation
        mask_upper = np.triu(np.ones_like(pearson_corr, dtype=bool), k=1)
        sns.heatmap(pearson_corr, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1, square=True, ax=axes[0],
                   cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
                   annot_kws={'fontsize': 16, 'fontweight': 'bold'},
                   linewidths=2, linecolor='white')
        axes[0].set_title('Pearson Correlation Matrix\n(Linear Relationships)', 
                         fontsize=16, fontweight='bold', pad=15)
        
        # Spearman correlation
        sns.heatmap(spearman_corr, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1, square=True, ax=axes[1],
                   cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
                   annot_kws={'fontsize': 16, 'fontweight': 'bold'},
                   linewidths=2, linecolor='white')
        axes[1].set_title('Spearman Correlation Matrix\n(Monotonic Relationships)', 
                         fontsize=16, fontweight='bold', pad=15)
        
        plt.tight_layout()
        filename = 'vad_correlation_heatmaps.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")
        
        # Calculate p-values and detailed statistics
        correlations = {}
        dimensions = ['valence', 'arousal', 'dominance']
        
        for i in range(len(dimensions)):
            for j in range(i+1, len(dimensions)):
                dim1, dim2 = dimensions[i], dimensions[j]
                
                pearson_r, pearson_p = pearsonr(vad_data[dim1], vad_data[dim2])
                spearman_r, spearman_p = spearmanr(vad_data[dim1], vad_data[dim2])
                
                correlations[f'{dim1}_vs_{dim2}'] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'significant': pearson_p < 0.05
                }
        
        return correlations
    
    def generate_insights_report(self, quadrant_dist, correlations):
        """
        Generate comprehensive insights report for presentation
        """
        print("\n[BONUS] Generating Insights Report...")
        
        report = []
        report.append("="*80)
        report.append("PHASE 1: EXPLORATORY ANALYSIS - KEY INSIGHTS")
        report.append("="*80)
        report.append("")
        
        # Dataset overview
        report.append("1. DATASET OVERVIEW")
        report.append("-"*80)
        report.append(f"   Total samples: {len(self.emotion_data)}")
        report.append(f"   Participants: {self.emotion_data['participant_id'].nunique()}")
        report.append(f"   Trials per participant: {self.emotion_data.groupby('participant_id').size().iloc[0]}")
        report.append("")
        report.append("   VAD Rating Statistics:")
        for dim in ['valence', 'arousal', 'dominance']:
            mean_val = self.emotion_data[dim].mean()
            std_val = self.emotion_data[dim].std()
            min_val = self.emotion_data[dim].min()
            max_val = self.emotion_data[dim].max()
            report.append(f"     {dim.capitalize():12s}: Mean={mean_val:.3f}, Std={std_val:.3f}, Range=[{min_val:.1f}, {max_val:.1f}]")
        report.append("")
        
        # Emotional distribution
        report.append("2. EMOTIONAL DISTRIBUTION (Circumplex Model)")
        report.append("-"*80)
        for emotion, count in quadrant_dist.items():
            report.append(f"   {emotion:30s}: {count}")
        report.append("")
        
        # Correlation insights
        report.append("3. CORRELATION INSIGHTS")
        report.append("-"*80)
        
        for pair, stats in correlations.items():
            dim1, dim2 = pair.split('_vs_')
            pearson_r = stats['pearson_r']
            pearson_p = stats['pearson_p']
            
            if abs(pearson_r) > 0.5:
                strength = "STRONG"
            elif abs(pearson_r) > 0.3:
                strength = "MODERATE"
            else:
                strength = "WEAK"
            
            direction = "positive" if pearson_r > 0 else "negative"
            significance = "***" if pearson_p < 0.001 else "**" if pearson_p < 0.01 else "*" if pearson_p < 0.05 else "ns"
            
            report.append(f"   {dim1.capitalize()} vs {dim2.capitalize()}:")
            report.append(f"     Pearson  r = {pearson_r:+.4f} ({strength} {direction}) {significance}")
            report.append(f"     Spearman ρ = {stats['spearman_r']:+.4f}")
            report.append(f"     p-value = {pearson_p:.6f}")
            
            # Interpretation
            if pair == 'valence_vs_dominance' and abs(pearson_r) > 0.3:
                report.append(f"     → Interpretation: Positive emotions associated with feeling in control")
            elif pair == 'arousal_vs_valence' and abs(pearson_r) < 0.2:
                report.append(f"     → Interpretation: Arousal independent of valence (supports Circumplex Model)")
            
            report.append("")
        
        report.append("="*80)
        report.append("PHASE 1 COMPLETE - All visualizations saved to outputs/plots/")
        report.append("="*80)
        
        # Print and save
        report_text = '\n'.join(report)
        print(report_text)
        
        with open(os.path.join(self.output_dir, 'phase1_insights_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n  ✓ Saved: phase1_insights_report.txt")
        
        return report_text
    
    def run_all(self):
        """Execute all Phase 1 analyses"""
        quadrant_dist = self.create_arousal_valence_plot()
        self.create_vad_bar_diagrams()
        correlations = self.correlation_analysis()
        self.generate_insights_report(quadrant_dist, correlations)
        
        print("\n" + "="*80)
        print("✓ PHASE 1 COMPLETE!")
        print("="*80)


# Test function
if __name__ == "__main__":
    from data_loader import DREAMERDataLoader
    
    dataset_path = r'D:\hackathon\neurohack\data\DREAMER.mat'
    loader = DREAMERDataLoader(dataset_path)
    data = loader.extract_all_data()
    
    analyzer = ExploratoryAnalysis(data['emotion_ratings'])
    analyzer.run_all()
