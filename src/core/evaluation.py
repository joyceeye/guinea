import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from .confidence import calculate_confidence

def evaluate_model(true_labels, predictions, threshold=0.5):
    # convert predictions to binary
    binary_preds = (predictions > threshold).astype(int)
    
    # calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, 
        binary_preds, 
        average='samples'
    )
    
    # calculate subset accuracy (exact match)
    subset_accuracy = accuracy_score(true_labels, binary_preds)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'subset_accuracy': subset_accuracy
    }

def calibrate_confidence_threshold(val_predictions, val_true_labels, 
                                  confidence_method, label_threshold,
                                  num_thresholds=10):
    # calculate confidence for each prediction
    confidences = np.array([
        calculate_confidence(pred, method=confidence_method, threshold=label_threshold)
        for pred in val_predictions
    ])
    
    # convert predictions to binary
    binary_preds = (val_predictions > label_threshold).astype(int)
    
    # calculate correctness of each prediction
    correctness = np.all(binary_preds == val_true_labels, axis=1)
    
    # try different thresholds
    thresholds = np.linspace(0, 1, num_thresholds)
    results = []
    
    for threshold in thresholds:
        # predictions with confidence above threshold
        high_conf_mask = confidences >= threshold
        
        # accuracy of high confidence predictions
        if np.sum(high_conf_mask) > 0:
            high_conf_accuracy = np.mean(correctness[high_conf_mask])
        else:
            high_conf_accuracy = 0
        
        # percentage of predictions routed to LLM
        llm_routing_percentage = 1 - np.mean(high_conf_mask)
        
        results.append({
            'threshold': threshold,
            'high_conf_accuracy': high_conf_accuracy,
            'llm_routing_percentage': llm_routing_percentage
        })
    
    return results