import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import pickle

from src.core.utils import select_tokenizer
from src.llm_wrapper import LLM
from src.models.baseline import BertMultiLabelClassifier
from src.models.llm_classifier import classify
from src.core.confidence import calculate_confidence
import src.config as config

def load_model_and_mlb():
    # load multilabel binarizer
    with open('mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
    
    # initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertMultiLabelClassifier(len(mlb.classes_))
    model.load_state_dict(torch.load('best_model.pt', map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    return model, mlb, device

def predict_with_routing(text, model, mlb, tokenizer, device,
                        confidence_method, confidence_threshold, label_threshold):
    # tokenize text
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=config.MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # get model prediction
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask)
    
    # convert to numpy
    predictions = outputs.cpu().numpy()[0]
    
    # calculate confidence
    confidence = calculate_confidence(
        predictions, 
        method=confidence_method,
        threshold=label_threshold
    )
    
    # decide whether to use model prediction or route to LLM
    if confidence >= confidence_threshold:
        # use model prediction
        binary_preds = (predictions > label_threshold).astype(int)
        predicted_labels = mlb.inverse_transform(binary_preds.reshape(1, -1))[0]
        source = "model"
    else:
        # route to LLM
        llm = LLM()
        predicted_labels = classify(llm, text, mlb.classes_)
        source = "llm"

    return {
        'predicted_labels': predicted_labels,
        'confidence': confidence,
        'source': source
    }

# prediction with bert only
def predict_with_bert_only(text, model, mlb, tokenizer, device, label_threshold):
    # tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=config.MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask)

    predictions = outputs.cpu().numpy()[0]
    binary_preds = (predictions > label_threshold).astype(int)
    predicted_labels = mlb.inverse_transform(binary_preds.reshape(1, -1))[0]

    return {
        'predicted_labels': predicted_labels,
        'source': 'model'
    }

def main():
    # load model and mlb
    model, mlb, device = load_model_and_mlb()
    
    # load tokenizer
    tokenizer = select_tokenizer()
    
    # sample text for prediction
    sample_text = '''
        Monocular depth prediction plays a crucial role in understanding 3D scene
        geometry. Although recent methods have achieved impressive progress in
        evaluation metrics such as the pixel-wise relative error, most methods neglect
        the geometric constraints in the 3D space. In this work, we show the importance
        of the high-order 3D geometric constraints for depth prediction. By designing a
        loss term that enforces one simple type of geometric constraints, namely,
        virtual normal directions determined by randomly sampled three points in the
        reconstructed 3D space, we can considerably improve the depth prediction
        accuracy. Significantly, the byproduct of this predicted depth being
        sufficiently accurate is that we are now able to recover good 3D structures of
        the scene such as the point cloud and surface normal directly from the depth,
        eliminating the necessity of training new sub-models as was previously done.
        Experiments on two benchmarks: NYU Depth-V2 and KITTI demonstrate the
        effectiveness of our method and state-of-the-art performance.
    '''

    # predict using BERT only
    result = predict_with_bert_only(
        sample_text,
        model,
        mlb,
        tokenizer,
        device,
        config.THRESHOLD
    )

    print(" Bert Only prediction")
    print(f"Predicted labels: {result['predicted_labels']}")
    # print(f"Confidence: {result['confidence']:.4f}")
    print(f"Source: {result['source']}")

    # predict with routing
    result1 = predict_with_routing(
        sample_text,
        model,
        mlb,
        tokenizer,
        device,
        config.CONFIDENCE_METHOD,
        config.CONFIDENCE_THRESHOLD,
        config.THRESHOLD
    )
    print(" Routed Prediction (BERT + LLM if needed)")
    print(f"Predicted labels: {result1['predicted_labels']}")
    print(f"Confidence: {result1['confidence']:.4f}")
    print(f"Source: {result1['source']}")

if __name__ == "__main__":
    main()