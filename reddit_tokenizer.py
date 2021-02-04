from transformers import BertTokenizer, DistilBertTokenizer


# Dictionary of all new special tokens to add to BERT model.
SPECIAL_TOKENS = {'additional_special_tokens': ['[CTXTSEP]',
                                                '[QUES]',
                                                '[ANS]',
                                                '[ANNOUNC]',
                                                '[AGREE]',
                                                '[DISAGREE]',
                                                '[NEGREACT]',
                                                '[APPR]',
                                                '[ELAB]',
                                                '[HUMOR]']}

# Dictionary mapping text labels to model input tokens.
LABELS2TOKENS = {'context_sep': '[CTXTSEP]',
                 'question': '[QUES]',
                 'answer': '[ANS]',
                 'announcement': '[ANNOUNC]',
                 'agreement': '[AGREE]',
                 'disagreement': '[DISAGREE]',
                 'negativereaction': '[NEGREACT]',
                 'appreciation': '[APPR]',
                 'elaboration': '[ELAB]',
                 'humor': '[HUMOR]'}

# Dictionary mapping text labels to indices for classification.
LABELS2IDX = {'question': 0,
              'answer': 1,
              'announcement': 2,
              'agreement': 3,
              'disagreement': 4,
              'negativereaction': 5,
              'appreciation': 6,
              'elaboration': 7,
              'humor': 8}

# Dictionary mapping label indices to text labels.
IDX2LABELS = {0: 'question',
              1: 'answer',
              2: 'announcement',
              3: 'agreement',
              4: 'disagreement',
              5: 'negativereaction',
              6: 'appreciation',
              7: 'elaboration',
              8: 'humor'}


# Initialize BERT tokenizer and add new special tokens.
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_tokenizer.add_special_tokens(SPECIAL_TOKENS)
distilbert_tokenizer.add_special_tokens(SPECIAL_TOKENS)