import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from model import BERTClassifierModel
from data_load import RedditDataset
from RedditTrainer import RedditDiscourseActTrainer
from utils import get_linear_schedule_with_warmup

# os.environ['KMP_DUPLICATE_LIB_OK']='True'


def parse_args():
    parser = argparse.ArgumentParser(description="BERTClassifierModel")

    parser.add_argument('--data_directory', type=str, default="./data/reddit_coarse_discourse", help='Directory where sentence embeddings are saved')

    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay parameter')

    parser.add_argument('--bert_encoder_type', type=str, default='DistilBERT', help='Type of BERT encoder (either BERT-Base or DistilBERT)')
    parser.add_argument('--dim_feedforward', type=int, default=768, help='Size of feedforward layer for finetuning')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability in feedforward layers')
    
    parser.add_argument('--max_subtree_depth', type=int, default=3, help='Maximum number of ancestor nodes to consider for context information')
    parser.add_argument('--use_ancestor_labels', dest='use_ancestor_labels', action='store_true', help='Use the labels of ancestor nodes in context information')
    parser.add_argument('--randomize_prob', type=float, default=0.1, help='Probability of using random ancestor label during training')
    parser.set_defaults(use_ancestor_labels=True)

    parser.add_argument('--ckpt_directory', type=str, default="./checkpoints", help='Path to save model checkpoints')
    # parser.add_argument('--log_file', type=str, default="./log.csv", help='Path to log file')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Directory where dataset is located (JSON files).
    data_directory = args.data_directory

    # General training hyperparameters.
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    weight_decay = args.weight_decay

    # Model hyperparameters.
    bert_encoder_type = args.bert_encoder_type
    dim_feedforward = args.dim_feedforward
    dropout = args.dropout

    # Hyperparameters for processing data before feeding into model.
    max_subtree_depth = args.max_subtree_depth
    use_ancestor_labels = args.use_ancestor_labels
    randomize_prob = args.randomize_prob

    # Checkpoint directory for saving model weights.
    ckpt_directory = args.ckpt_directory
    # os.makedirs(ckpt_directory, exist_ok=True)

    # Get device; detect if there is a GPU available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda:0"):
        print("Using GPU.")
    else:
        print("Using CPU.")

    # Initialize model and send to device.
    model = BERTClassifierModel(bert_encoder_type,
                                dim_feedforward=dim_feedforward,
                                dropout=dropout)
    model = model.to(device)

    # Initialize dataset.
    reddit_dataset = RedditDataset(data_directory,
                                   model.tokenizer,
                                   max_subtree_depth,
                                   use_ancestor_labels,
                                   randomize_prob)

    # Initialize dataloader.
    reddit_loader = torch.utils.data.DataLoader(reddit_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=4,
                                                drop_last=False)

    # Initialize optimizer and learning rate scheduler.
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)
    num_training_steps = len(reddit_loader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Initialize object that wraps together all parameters and objects
    # for training.
    RedditTrainer = RedditDiscourseActTrainer(model,
                                              device,
                                              optimizer,
                                              scheduler,
                                              ckpt_directory)

    # Train the model.
    RedditTrainer.train(reddit_loader, reddit_dataset, num_epochs)

