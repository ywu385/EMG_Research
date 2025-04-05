from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pickle
from collections import Counter
import pandas as pd
import os

# Ensure working_models directory exists
os.makedirs('./working_models', exist_ok=True)

def bagged_predict(models, X):
    """
    Make predictions using an ensemble of models, taking the most common prediction
    for each sample.
    
    Args:
        models: List of trained classifier models
        X: Features to predict on
        
    Returns:
        Array of predictions
    """
    all_preds = [model.predict(X) for model in models]
    return np.array([
        Counter(col).most_common(1)[0][0] for col in zip(*all_preds)
    ])

def main():
    # Load the dataset
    df = pd.read_csv('./emg_recordings/training_data.csv')
    X = df.drop(columns=['label', 'name', 'activity', 'mixed_label'])
    y = df['mixed_label']
    
    print(f"Training on dataset with {X.shape[0]} samples and {X.shape[1]} features")
    
    # Setup cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    models = []
    
    # Train models
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Training fold {i+1}/10...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        clf = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=10,
            num_leaves=31,
            min_child_samples=10,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        
        clf.fit(X_train, y_train)
        val_score = clf.score(X_val, y_val)
        print(f"Fold {i+1} validation accuracy: {val_score:.4f}")
        models.append(clf)
    
    # Save models
    with open('./working_models/LGBM.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print(f"Model saved to ./working_models/LGBM.pkl")
    
    # Create a small test to verify the saved model works
    test_sample = X.iloc[:5]
    ensemble_pred = bagged_predict(models, test_sample)
    print(f"Test prediction on first 5 samples: {ensemble_pred}")
    print("Done!")

if __name__ == "__main__":
    main()