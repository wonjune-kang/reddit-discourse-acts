import os
import argparse
import torch
import torch.optim as optim
from transformers.optimization import get_linear_schedule_with_warmup

from format_reddit_data import process_all_trees
from model import BERTClassifierModel
from data_load import RedditDataset
from RedditTrainer import RedditDiscourseActTrainer
from utils import get_xval_splits, get_train_val_test_splits

# os.environ['KMP_DUPLICATE_LIB_OK']='True'


def parse_args():
    parser = argparse.ArgumentParser(description="RedditDiscourseActTrainer")

    parser.add_argument('--data_directory', type=str, default="./data/reddit_coarse_discourse", help='Directory where sentence embeddings are saved')

    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay parameter')

    parser.add_argument('--bert_encoder_type', type=str, default='DistilBERT', help='Type of BERT encoder (either BERT-Base or DistilBERT)')
    parser.add_argument('--dim_feedforward', type=int, default=768, help='Size of feedforward layer for finetuning')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability in feedforward layers')
    
    parser.add_argument('--max_subtree_depth', type=int, default=2, help='Maximum number of ancestor nodes to consider for context information')
    parser.add_argument('--use_ancestor_labels', dest='use_ancestor_labels', action='store_true', help='Use labels of ancestor nodes in context information')
    parser.add_argument('--no_ancestor_labels', dest='use_ancestor_labels', action='store_false', help='Do not use labels of ancestor nodes in context information')
    parser.add_argument('--randomize_prob', type=float, default=0.1, help='Probability of using random ancestor label during training')
    parser.set_defaults(use_ancestor_labels=True)

    parser.add_argument('--run_name', type=str, help='String identifying the name of the run (e.g. using hyperparameters)')
    parser.add_argument('--save_directory', type=str, default="./model_weights", help='Path to save model weights')
    parser.add_argument('--log_file', type=str, default="./logs/log.csv", help='Path to log file')
    parser.add_argument('--run_xval', dest='run_xval', action='store_true', help='Run 10-fold cross validation')
    parser.add_argument('--no_xval', dest='run_xval', action='store_false', help='Do not run 10-fold cross validation')
    parser.set_defaults(run_xval=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Directory where dataset is located.
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
    run_name = args.run_name
    save_directory = args.save_directory
    run_xval = args.run_xval
    # os.makedirs(save_directory, exist_ok=True)

    # Get device; detect if there is a GPU available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda:0"):
        print("Using GPU.")
    else:
        print("Using CPU.")

    # Generate trees for all Reddit thread JSON files in data directory.
    all_thread_trees = process_all_trees(data_directory)
    xval_splits = get_xval_splits(all_thread_trees)

    for test_idx in range(10):
        # Initialize model and send to device.
        model = BERTClassifierModel(bert_encoder_type,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout)
        model = model.to(device)

        train_data, val_data, test_data = get_train_val_test_splits(xval_splits, test_idx)

        # Initialize dataset.
        reddit_dataset = RedditDataset(train_data,
                                        val_data,
                                        test_data,
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
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps,
                                                    num_training_steps)

        # Initialize Trainer that wraps all training parameters and objects.
        RedditTrainer = RedditDiscourseActTrainer(run_name,
                                                  model,
                                                  device,
                                                  optimizer,
                                                  scheduler,
                                                  save_directory)

        # Train the model.
        RedditTrainer.train(reddit_loader, reddit_dataset, num_epochs)

        if run_xval == False:
            print("\nNot running 10-fold cross validation. Exiting after running 1 fold.")
            exit()