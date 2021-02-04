import torch
import torch.nn as nn
from transformers import BertModel, DistilBertModel

from reddit_tokenizer import bert_tokenizer, distilbert_tokenizer


class BERTClassifierModel(nn.Module):
    def __init__(self, bert_encoder_type, dim_feedforward=768, dropout=0.1):
        super(BERTClassifierModel, self).__init__()

        assert bert_encoder_type in ['BERT-Base', 'DistilBERT'], \
            "Only BERT-Base and DistilBERT encoders have been implemented"

        # Load pre-trained model weights and resize model vocabulary to add
        # new special tokens
        if bert_encoder_type == 'BERT-Base':
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = bert_tokenizer
        elif bert_encoder_type == 'DistilBERT':
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.tokenizer = distilbert_tokenizer
        self.bert.resize_token_embeddings(len(self.tokenizer))

        # 768 is dimension of BERT embeddings. 9-way classification task.
        self.classifier = nn.Linear(768, 9)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # [CLS] embedding used for finetuning is at 0th index of output.
        if token_type_ids:
            x = self.bert(input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)['last_hidden_state'][:,0,:]
        else:
            x = self.bert(input_ids,
                          attention_mask=attention_mask)['last_hidden_state'][:,0,:]
        x = self.dropout(x)
        x = self.classifier(x)
        return x