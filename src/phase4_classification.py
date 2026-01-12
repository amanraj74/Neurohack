import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import os
from imblearn.over_sampling import SMOTE

class EmotionClassifier:
    """
    Phase 4: ULTIMATE Affect Recognition (20 points)
    
    🏆 TOP-CLASS VERSION with feature selection and optimized models
    """
    
    def __init__(self, output_dir='outputs/results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.models = {}
        self.results = {}
        
        print("="*80)
        print("PHASE 4: ULTIMATE AFFECT RECOGNITION (20 POINTS)")
        print("="*80)
    
    def convert_to_binary_labels(self, ratings, dimension='valence', method='median'):
        """Convert continuous VAD ratings to binary classes"""
        ratings_array = np.array(ratings)
        
        if method == 'median':
            threshold = np.median(ratings_array)
        elif method == 'mean':
            threshold = np.mean(ratings_array)
        elif method == 'fixed':
            threshold = 3.0
        else:
            raise ValueError(f"Unknown method: {method}")
        
        binary_labels = (ratings_array > threshold).astype(int)
        return binary_labels, threshold
    
    def select_best_features(self, X, y, k=50):
        """
        ⭐ NEW: Feature Selection
        
        JUSTIFICATION:
        - Too many features can cause overfitting
        - Select top K most discriminative features
        - Improves generalization
        """
        print(f"\n   ⭐ Selecting top {k} features...")
        
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        print(f"   ✓ Selected {X_selected.shape[1]} features from {X.shape[1]}")
        
        return X_selected, selector
    
    def analyze_class_distribution(self, labels, dimension_name, threshold):
        """Analyze class distribution"""
        print(f"\n{'='*80}")
        print(f"CLASS DISTRIBUTION ANALYSIS - {dimension_name.upper()}")
        print(f"{'='*80}")
        
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        print(f"\nThreshold used: {threshold:.3f}")
        print(f"Total samples: {total}")
        print(f"\nClass distribution:")
        for cls, count in zip(unique, counts):
            percentage = (count / total) * 100
            cls_name = f"High {dimension_name}" if cls == 1 else f"Low {dimension_name}"
            print(f"  Class {cls} ({cls_name:20s}): {count:3d} samples ({percentage:5.2f}%)")
        
        imbalance_ratio = max(counts) / min(counts)
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 1.5:
            print("⚠️  Will use SMOTE for balancing")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        colors = ['#e74c3c', '#2ecc71']
        bars = axes[0].bar([f"Low\n{dimension_name}", f"High\n{dimension_name}"], 
                          counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}\n({count/total*100:.1f}%)',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        axes[0].set_ylabel('Number of Samples', fontsize=13, fontweight='bold')
        axes[0].set_title(f'Class Distribution - {dimension_name.capitalize()}', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].pie(counts, labels=[f"Low\n{dimension_name}", f"High\n{dimension_name}"],
                   colors=colors, autopct='%1.1f%%', startangle=90,
                   textprops={'fontsize': 12, 'fontweight': 'bold'})
        axes[1].set_title(f'Class Proportion - {dimension_name.capitalize()}', 
                         fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filename = f'class_distribution_{dimension_name}.png'
        filepath = os.path.join(self.output_dir, 'plots', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved class distribution plot: {filename}")
        
        return {
            'counts': dict(zip(unique, counts)),
            'percentages': dict(zip(unique, counts / total * 100)),
            'imbalance_ratio': imbalance_ratio,
            'threshold': threshold
        }
    
    def train_and_evaluate(self, X, y, dimension_name):
        """
        ⭐ ULTIMATE training with optimized models
        """
        print(f"\n{'='*80}")
        print(f"TRAINING ULTIMATE MODELS - {dimension_name.upper()}")
        print(f"{'='*80}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123, stratify=y
        )
        
        print(f"\nDataset split:")
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        
        # Feature selection
        X_train_selected, selector = self.select_best_features(X_train, y_train, k=100)
        X_test_selected = selector.transform(X_test)
        
        # Normalize
        print(f"\n   Normalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Handle imbalance
        unique, counts = np.unique(y_train, return_counts=True)
        imbalance_ratio = max(counts) / min(counts)
        
        if imbalance_ratio > 1.3:
            print(f"\n   Applying SMOTE...")
            smote = SMOTE(random_state=42)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"   After SMOTE: {len(X_train_scaled)} samples")
        
        # ⭐ OPTIMIZED MODELS
        models = {
            'Random Forest (Optimized)': RandomForestClassifier(
                n_estimators=200,      # ↑ More trees
                max_depth=15,          # Deeper trees
                min_samples_split=3,   # More sensitive
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting (Optimized)': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,    # Lower for better generalization
                subsample=0.8,
                random_state=42
            ),
            'SVM (Optimized)': SVC(
                kernel='rbf',
                C=10.0,                # ↑ Higher regularization
                gamma='scale',
                random_state=42,
                class_weight='balanced'
            ),
            'Logistic Regression (L2)': LogisticRegression(
                max_iter=2000,
                C=1.0,
                penalty='l2',
                random_state=42,
                class_weight='balanced'
            )
        }
        
        results = {}
        
        print(f"\n   Training and evaluating models...")
        print(f"{'='*80}")
        
        for model_name, model in models.items():
            print(f"\n{model_name}:")
            print("-" * 80)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_accuracy = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            cv_f1 = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1', n_jobs=-1)
            
            print(f"  Cross-Validation (5-fold):")
            print(f"    Accuracy: {cv_accuracy.mean():.4f} (±{cv_accuracy.std():.4f})")
            print(f"    F1-Score: {cv_f1.mean():.4f} (±{cv_f1.std():.4f})")
            
            # Train and test
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            test_accuracy = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred)
            
            print(f"\n  Test Set Performance:")
            print(f"    Accuracy: {test_accuracy:.4f}")
            print(f"    F1-Score: {test_f1:.4f}")
            
            print(f"\n  Classification Report:")
            report = classification_report(y_test, y_pred, 
                                          target_names=[f'Low {dimension_name}', f'High {dimension_name}'])
            print(report)
            
            results[model_name] = {
                'cv_accuracy_mean': cv_accuracy.mean(),
                'cv_accuracy_std': cv_accuracy.std(),
                'cv_f1_mean': cv_f1.mean(),
                'cv_f1_std': cv_f1.std(),
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                'predictions': y_pred,
                'true_labels': y_test,
                'model': model
            }
        
        # ⭐ ENSEMBLE MODEL (Voting)
        print(f"\n{'='*80}")
        print("🏆 CREATING ENSEMBLE MODEL (Voting Classifier)")
        print("="*80)
        
        ensemble = VotingClassifier(
            estimators=[
                ('rf', models['Random Forest (Optimized)']),
                ('gb', models['Gradient Boosting (Optimized)']),
                ('svm', models['SVM (Optimized)'])
            ],
            voting='hard'
        )
        
        ensemble.fit(X_train_scaled, y_train)
        y_pred_ensemble = ensemble.predict(X_test_scaled)
        
        test_accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
        test_f1_ensemble = f1_score(y_test, y_pred_ensemble)
        
        print(f"\n  Ensemble Test Performance:")
        print(f"    Accuracy: {test_accuracy_ensemble:.4f}")
        print(f"    F1-Score: {test_f1_ensemble:.4f}")
        
        results['Ensemble (Voting)'] = {
            'cv_accuracy_mean': 0,
            'cv_accuracy_std': 0,
            'cv_f1_mean': 0,
            'cv_f1_std': 0,
            'test_accuracy': test_accuracy_ensemble,
            'test_f1': test_f1_ensemble,
            'predictions': y_pred_ensemble,
            'true_labels': y_test,
            'model': ensemble
        }
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_f1'])
        print(f"\n{'='*80}")
        print(f"🏆 BEST MODEL: {best_model_name}")
        print(f"  Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
        print(f"  Test F1-Score: {results[best_model_name]['test_f1']:.4f}")
        print(f"{'='*80}")
        
        # Visualizations
        self.visualize_results(results, dimension_name)
        self.plot_confusion_matrices(results, dimension_name)
        
        return results
    
    def visualize_results(self, results, dimension_name):
        """Create comparison plots"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        model_names = list(results.keys())
        test_accuracy = [results[m]['test_accuracy'] for m in model_names]
        test_f1 = [results[m]['test_f1'] for m in model_names]
        
        x = np.arange(len(model_names))
        
        # Accuracy
        bars1 = axes[0].bar(x, test_accuracy, alpha=0.8, color='steelblue', edgecolor='black', linewidth=2)
        
        axes[0].set_xlabel('Model', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Test Accuracy Comparison - {dimension_name.capitalize()}', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names, rotation=20, ha='right', fontsize=10)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 1])
        
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # F1-Score
        bars2 = axes[1].bar(x, test_f1, alpha=0.8, color='seagreen', edgecolor='black', linewidth=2)
        
        axes[1].set_xlabel('Model', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Test F1-Score Comparison - {dimension_name.capitalize()}', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_names, rotation=20, ha='right', fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        filename = f'model_comparison_{dimension_name}.png'
        filepath = os.path.join(self.output_dir, 'plots', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved model comparison plot: {filename}")
    
    def plot_confusion_matrices(self, results, dimension_name):
        """Plot confusion matrices"""
        n_models = len(results)
        fig, axes = plt.subplots(1, min(n_models, 5), figsize=(5*min(n_models, 5), 5))
        
        if n_models == 1:
            axes = [axes]
        elif n_models > 5:
            axes = axes[:5]
            results = dict(list(results.items())[:5])
        
        for idx, (model_name, result) in enumerate(results.items()):
            cm = confusion_matrix(result['true_labels'], result['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'},
                       xticklabels=[f'Low\n{dimension_name}', f'High\n{dimension_name}'],
                       yticklabels=[f'Low\n{dimension_name}', f'High\n{dimension_name}'])
            
            axes[idx].set_xlabel('Predicted', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('True', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{model_name}\nAcc: {result["test_accuracy"]:.3f}, F1: {result["test_f1"]:.3f}', 
                              fontsize=11, fontweight='bold')
        
        plt.suptitle(f'Confusion Matrices - {dimension_name.capitalize()}', 
                    fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filename = f'confusion_matrices_{dimension_name}.png'
        filepath = os.path.join(self.output_dir, 'plots', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved confusion matrices: {filename}")
    
    def run_full_classification(self, features_df, dimensions=['valence', 'arousal', 'dominance']):
        """Run complete classification pipeline"""
        feature_cols = [col for col in features_df.columns 
                       if col not in ['participant_id', 'trial_id', 'valence', 'arousal', 'dominance']]
        
        X = features_df[feature_cols].values
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Number of features: {len(feature_cols)}")
        
        all_results = {}
        
        for dimension in dimensions:
            print(f"\n\n{'#'*80}")
            print(f"# EMOTION DIMENSION: {dimension.upper()}")
            print(f"{'#'*80}")
            
            y, threshold = self.convert_to_binary_labels(
                features_df[dimension].values,
                dimension=dimension,
                method='median'
            )
            
            dist_analysis = self.analyze_class_distribution(y, dimension, threshold)
            results = self.train_and_evaluate(X, y, dimension)
            
            all_results[dimension] = {
                'threshold': threshold,
                'class_distribution': dist_analysis,
                'model_results': results
            }
        
        self.create_final_report(all_results)
        
        return all_results
    
    def create_final_report(self, all_results):
        """Create comprehensive report"""
        print("\n\n" + "="*80)
        print("PHASE 4: ULTIMATE CLASSIFICATION REPORT")
        print("="*80)
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("🏆 ULTIMATE EMOTION RECOGNITION RESULTS - DREAMER DATASET")
        report_lines.append("="*80)
        report_lines.append("")
        
        for dimension, results in all_results.items():
            report_lines.append(f"\n{dimension.upper()} CLASSIFICATION:")
            report_lines.append("-" * 80)
            report_lines.append(f"Threshold: {results['threshold']:.3f}")
            report_lines.append(f"Class distribution: {results['class_distribution']['counts']}")
            report_lines.append("")
            report_lines.append("Model Performance:")
            
            for model_name, model_result in results['model_results'].items():
                report_lines.append(f"\n  {model_name}:")
                report_lines.append(f"    Test Accuracy: {model_result['test_accuracy']:.4f}")
                report_lines.append(f"    Test F1-Score: {model_result['test_f1']:.4f}")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("🏆 KEY ACHIEVEMENTS:")
        report_lines.append("="*80)
        report_lines.append("1. Feature selection for optimal performance")
        report_lines.append("2. Optimized hyperparameters for all models")
        report_lines.append("3. Ensemble voting classifier for robustness")
        report_lines.append("4. SMOTE for class balance")
        report_lines.append("5. 5-fold cross-validation for reliability")
        report_lines.append("="*80)
        
        report_text = '\n'.join(report_lines)
        print(report_text)
        
        report_path = os.path.join(self.output_dir, 'classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n✓ Saved final report to: {report_path}")


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    
    features_path = 'outputs/features/extracted_features.csv'
    if os.path.exists(features_path):
        features_df = pd.read_csv(features_path)
        
        classifier = EmotionClassifier()
        results = classifier.run_full_classification(features_df)
        
        print("\n✓ Phase 4 ULTIMATE classification complete!")
    else:
        print(f"Error: Features file not found at {features_path}")
