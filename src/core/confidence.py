import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

def avg_max_prob_confidence(predictions, threshold=0.5):
    # get predicted labels
    predicted_labels = (predictions > threshold).astype(int)
    
    # if no label is predicted, return 0 confidence
    if np.sum(predicted_labels) == 0:
        return 0.0
    
    # get probabilities for predicted labels
    probs = predictions * predicted_labels
    
    # calculate average probability for predicted labels
    avg_prob = np.sum(probs) / np.sum(predicted_labels)
    
    return avg_prob

def entropy_confidence(predictions):
    # normalize to ensure it sums to 1
    normalized_preds = predictions / np.sum(predictions)
    
    # calculate entropy
    ent = entropy(normalized_preds)
    
    # higher entropy means lower confidence
    max_entropy = np.log(len(predictions))
    confidence = 1 - (ent / max_entropy)
    
    return confidence

def jensen_shannon_confidence(predictions):
    # uniform distribution
    uniform = np.ones(len(predictions)) / len(predictions)
    
    # normalize predictions to ensure it sums to 1
    normalized_preds = predictions / np.sum(predictions)
    
    # calculate JS divergence
    js_div = jensenshannon(normalized_preds, uniform)
    
    # JS div is between 0 and 1, where 0 means identical to uniform (low confidence)
    return js_div

def calculate_confidence(predictions, method="avg_max_prob", threshold=0.5):
    if method == "avg_max_prob":
        return avg_max_prob_confidence(predictions, threshold)
    elif method == "entropy":
        return entropy_confidence(predictions)
    elif method == "jensen_shannon":
        return jensen_shannon_confidence(predictions)
    else:
        raise ValueError(f"Unknown confidence method: {method}")
