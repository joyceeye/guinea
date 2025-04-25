import torch.nn as nn
from src.core.utils import select_model
import src.config as config

class BertMultiLabelClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertMultiLabelClassifier, self).__init__()
        self.bert = select_model()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        if config.MODEL_NAME == "distilbert-base-uncased":
            pooled_output = outputs.last_hidden_state[:, 0]
        else:
            pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return self.sigmoid(logits)