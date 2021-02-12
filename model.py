import torch
import torch.nn as nn
from transformers import BertModel, DistilBertModel
from transformers import BertTokenizer, DistilBertTokenizer

from reddit_tokens import SPECIAL_TOKENS


class BERTClassifierModel(nn.Module):
    def __init__(self, bert_encoder_type, num_classes=9, dropout=0.1):
        super(BERTClassifierModel, self).__init__()

        assert bert_encoder_type in ['BERT-Base', 'DistilBERT'], \
            "Only BERT-Base and DistilBERT encoders have been implemented"

        # Load pre-trained model weights and initialize corresponding tokenizer.
        if bert_encoder_type == 'BERT-Base':
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif bert_encoder_type == 'DistilBERT':
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Add special tokens and resize model vocabulary.
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        self.bert.resize_token_embeddings(len(self.tokenizer))

        # 768 is dimension of BERT embeddings.
        # 9 is for the 9-way discourse act classification task.
        self.classifier = nn.Linear(self.bert.hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # [CLS] embedding used for finetuning is at 0th index of output.
        if token_type_ids is not None:
            x = self.bert(input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)['last_hidden_state'][:,0,:]
        else:
            x = self.bert(input_ids,
                          attention_mask=attention_mask)['last_hidden_state'][:,0,:]
        x = self.dropout(x)
        x = self.classifier(x)
        return x