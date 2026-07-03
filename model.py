"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # TODO: score how mixed the labels are; 0 for a pure set, larger for more mixed sets.
    if len(labels) == 0:
        return 0.0
        
    
    _, counts = np.unique(labels, return_counts=True)
    
    
    probabilities = counts / len(labels)
    
    
    gini = 1.0 - np.sum(probabilities ** 2)
    
    return float(gini)

    pass

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    left_mask = features[:, feature_index] <= threshold
    right_mask = ~left_mask  
    
   
    left_features = features[left_mask]
    left_labels = labels[left_mask]
    
    right_features = features[right_mask]
    right_labels = labels[right_mask]
    
    return left_features, left_labels, right_features, right_labels

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    n = len(parent_labels)
    w_l = len(left_labels) / n
    w_r = len(right_labels) / n
    return impurity(parent_labels) - (w_l * impurity(left_labels) + w_r * impurity(right_labels))

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    best_res = {
        'feature_index': None,
        'threshold': None,
        'score': 0.0
    }
    
    if len(labels) <= 1:
        return best_res

    for feature_idx in feature_indices:
        X_column = features[:, feature_idx]
        sort_idx = np.argsort(X_column)
        X_sort, y_sort = X_column[sort_idx], labels[sort_idx]
        
        # Evaluate midpoints between consecutive distinct values
        for i in range(1, len(X_sort)):
            if X_sort[i] == X_sort[i - 1]:
                continue
                
            threshold = (X_sort[i] + X_sort[i - 1]) / 2.0
            y_left = y_sort[:i]
            y_right = y_sort[i:]
            
            # Compute purity improvement
            score = split_score(y_sort, y_left, y_right)
            
            if score > best_res['score']:
                best_res['score'] = score
                best_res['feature_index'] = feature_idx
                best_res['threshold'] = threshold
                
    return best_res

# Step 5 - should_stop (not yet solved)
# TODO: implement

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

