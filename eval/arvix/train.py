import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.data.data_loader import load_data, prepare_data
from src.data.data_splitting import split_data
from src.models.baseline import BertMultiLabelClassifier
from src.core.evaluation import evaluate_model, calibrate_confidence_threshold
from src.core.utils import print_color, select_tokenizer
import src.config as config

from src.core.confidence import calculate_confidence
from src.models.llm_classifier import classify
from src.llm_wrapper import LLM
import numpy as np

# check if GPU is available (m1 chip)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def train_model(model, train_dataloader, val_dataloader, 
                criterion, optimizer, device, epochs):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # calculate average training loss
        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
    
    # plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    
    return train_losses, val_losses

def evaluate_with_llm_routing(val_predictions, val_true_labels, val_dataloader, tokenizer, config):
    """
    Evaluate combined BERT + LLM predictions.
    - val_predictions: np.ndarray, shape (n_samples, n_labels), raw BERT probs
    - val_true_labels: np.ndarray, shape (n_samples, n_labels)
    - val_dataloader: torch DataLoader with validation texts (needed to extract raw input text)
    - tokenizer: tokenizer used to decode input_ids
    - config: contains thresholds and confidence method
    """
    # initialize LLM
    llm = LLM()
    
    # extract all texts from val_dataloader
    all_input_ids = []
    
    # we need to extract all texts first
    for batch in val_dataloader:
        all_input_ids.extend(batch['input_ids'].cpu().numpy())
    
    final_preds = []
    routed_to_llm = 0
    
    # process each prediction
    for i, probs in enumerate(val_predictions):
        # calculate confidence
        confidence = calculate_confidence(
            probs, 
            method=config.CONFIDENCE_METHOD,
            threshold=config.THRESHOLD
        )
        
        # decide whether to use model prediction or route to LLM
        if confidence >= config.CONFIDENCE_THRESHOLD:
            # use model prediction
            # convert raw probabilities to binary predictions
            binary_pred = (probs > config.THRESHOLD).astype(float)
            final_preds.append(binary_pred)
        else:
            routed_to_llm += 1
            # decode text from input_ids
            input_id = all_input_ids[i]
            text = tokenizer.decode(input_id, skip_special_tokens=True)
            
            # route to LLM
            llm_labels = classify(llm, text, mlb.classes_)
            
            # convert LLM labels to binary vector
            llm_binary = mlb.transform([llm_labels])[0]
            
            # add to final predictions
            final_preds.append(llm_binary)
    
    final_preds = np.array(final_preds)
    
    # evaluate combined model
    from src.core.evaluation import evaluate_model
    metrics = evaluate_model(val_true_labels, final_preds, config.THRESHOLD)
    
    # calculate routing percentage
    routing_percentage = routed_to_llm / len(val_predictions)
    metrics['llm_routing_percentage'] = routing_percentage
    
    return metrics

def main():
    # load data
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), config.DATA_PATH))
    df = load_data(DATA_PATH)
    
    df_sampled = df.sample(frac=config.SAMPLE_RATIO, random_state=config.RANDOM_SEED)
    print(f"Using {len(df_sampled)} samples (originally {len(df)})")
    
    # get all possible labels before splitting
    all_possible_labels = set()
    for labels in df['labels']:
        all_possible_labels.update(labels)
    
    # split the sampled data
    train_df, val_df, test_df = split_data(
        df_sampled, 
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.RANDOM_SEED
    )
    
    # initialize tokenizer
    tokenizer = select_tokenizer()
    
    # create mlb with all possible labels first
    mlb = MultiLabelBinarizer(classes=list(all_possible_labels))
    mlb.fit([all_possible_labels])
    
    # prepare dataloaders using pre-fit mlb
    train_dataloader, _ = prepare_data(
        train_df,
        tokenizer,
        config.MAX_LENGTH,
        config.BATCH_SIZE,
        mlb=mlb
    )
    
    val_dataloader, _ = prepare_data(
        val_df,
        tokenizer,
        config.MAX_LENGTH,
        config.BATCH_SIZE,
        mlb=mlb  # use the same mlb for consistent dimensions
    )
    
    test_dataloader, _ = prepare_data(
        test_df,
        tokenizer,
        config.MAX_LENGTH,
        config.BATCH_SIZE,
        mlb=mlb  # use the same mlb for consistent dimensions
    )
    
    # save label binarizer
    with open('mlb.pkl', 'wb') as f:
        pickle.dump(mlb, f)
    
    # initialize model
    num_labels = len(mlb.classes_)
    model = BertMultiLabelClassifier(num_labels)
    model.to(device)
    
    print_color("initializing model...\n", "GREEN")
    print_color(f"model: {model}", "GREEN")
    print_color(f"device: {device}\n", "GREEN")

    # initialize loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    print_color("starting training...\n", "GREEN")

    # train model
    train_model(
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        device,
        config.EPOCHS
    )
    
    print_color("training complete\n", "BLUE")

    # load best model
    model.load_state_dict(torch.load('best_model.pt', map_location=device, weights_only=True))
    
    # evaluate on validation set
    model.eval()
    val_predictions = []
    val_true_labels = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask)
            
            val_predictions.extend(outputs.cpu().numpy())
            val_true_labels.extend(labels.numpy())
    
    val_predictions = np.array(val_predictions)
    val_true_labels = np.array(val_true_labels)
    
    # calibrate confidence threshold
    calibration_results = calibrate_confidence_threshold(
        val_predictions,
        val_true_labels,
        config.CONFIDENCE_METHOD,
        config.THRESHOLD
    )
    
    # print calibration results
    for result in calibration_results:
        print(f"Threshold: {result['threshold']:.2f}, "
              f"High conf accuracy: {result['high_conf_accuracy']:.4f}, "
              f"LLM routing %: {result['llm_routing_percentage']:.2%}")
    
    #### bert+llm evaluation:
    val_dataloader_single, _ = prepare_data(
        val_df,
        tokenizer,
        config.MAX_LENGTH,
        batch_size=1,
        mlb=mlb
    )
    
    print_color("evaluating with BERT+LLM routing...\n", "GREEN")
    
    # Use adjusted confidence threshold if needed
    adjusted_conf_threshold = 0.6
    
    # Create a temporary config with adjusted threshold
    import types
    temp_config = types.SimpleNamespace()
    for key, value in vars(config).items():
        setattr(temp_config, key, value)
    temp_config.CONFIDENCE_THRESHOLD = adjusted_conf_threshold
    
    # Updated function call with adjusted config
    metrics_combined = evaluate_with_llm_routing(
        val_predictions, 
        val_true_labels, 
        val_dataloader_single,
        tokenizer,
        temp_config
    )
    
    print(f"BERT+LLM Combined Validation Metrics: {metrics_combined}")


    # plot calibration results
    plt.figure(figsize=(10, 5))
    thresholds = [r['threshold'] for r in calibration_results]
    accuracies = [r['high_conf_accuracy'] for r in calibration_results]
    routing_percentages = [r['llm_routing_percentage'] for r in calibration_results]
    
    plt.plot(thresholds, accuracies, 'b-', label='High Conf Accuracy')
    plt.plot(thresholds, routing_percentages, 'r-', label='LLM Routing %')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('calibration_curve.png')
    
    # save configs and results
    results = {
        'metrics': metrics,
        'metrics_combined': metrics_combined,
        'calibration': calibration_results
    }
    
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()