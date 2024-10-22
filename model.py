from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
# from model_cnn import CNN_Text

class BertForSimilary(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSimilary, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.cnn = CNN_Text(1, 768, [2,3], self.num_labels, config.hidden_dropout_prob)
#         self.classifier1 = nn.Linear(config.hidden_size*2, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

#         sequence_output = outputs[0] # batch*seq*hidden_size
        pooled_output = outputs[1]
    
        # classification
#         logits = self.cnn(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
#         logits = self.classifier1(logits)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
#             loss_fct = CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 0.296, 0.684])).float(), size_average=True).to('cuda')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = loss
        else:
            outputs = nn.functional.softmax(logits,-1)

        return outputs