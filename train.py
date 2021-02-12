import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from transformers.optimization import get_linear_schedule_with_warmup

from format_reddit_data import process_all_trees
from model import BERTClassifierModel
from data_load import RedditDataset
from RedditTrainer import RedditDiscourseActTrainer
from utils import get_xval_splits, get_train_val_test_splits

# os.environ['KMP_DUPLICATE_LIB_OK']='True'


def init_score_file(score_file):
    """
    Initializes a CSV file for logging scores with five headers:
    Fold, Accuracy, Precision, Recall, and F1 score.
    """
    with open(score_file, 'a', newline='') as log_file:
        csv_writer = csv.writer(log_file, delimiter='\t')
        csv_writer.writerow(['Fold', 'Acc', 'Prec', 'Recall', 'F1'])
    return

def write_scores(score_file, fold_idx, scores):
    """
    Writes the scores for a run into a CSV log file. scores is a 4-tuple
    containing accuracy, precision, recall, and F1 score in that order.
    """
    with open(score_file, 'a', newline='') as log_file:
        csv_writer = csv.writer(log_file, delimiter='\t')
        csv_writer.writerow([fold_idx]+[round(x, 4) for x in scores])
    return

def parse_args():
    parser = argparse.ArgumentParser(description="RedditDiscourseActTrainer")

    parser.add_argument('--run_name', type=str, required=True, help='String identifying the name of the run (e.g. using hyperparameters)')
    parser.add_argument('--data_path', type=str, default="./data/reddit_coarse_discourse_clean", help='path where sentence embeddings are saved')

    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay parameter')

    parser.add_argument('--bert_encoder_type', type=str, default='BERT-Base', help='Type of BERT encoder (either BERT-Base or DistilBERT)')
    parser.add_argument('--dim_feedforward', type=int, default=768, help='Size of feedforward layer for finetuning')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability in feedforward layers')
    
    parser.add_argument('--max_subtree_depth', type=int, default=2, help='Maximum number of ancestor nodes to consider for context information')
    parser.add_argument('--use_ancestor_labels', dest='use_ancestor_labels', action='store_true', help='Use labels of ancestor nodes in context information')
    parser.add_argument('--no_ancestor_labels', dest='use_ancestor_labels', action='store_false', help='Do not use labels of ancestor nodes in context information')
    parser.add_argument('--randomize_prob', type=float, default=0.1, help='Probability of using random ancestor label during training')
    parser.set_defaults(use_ancestor_labels=True)

    parser.add_argument('--xval_test_idx', type=int, default=0, help='Cross validation split index to use for testing (must be integer in range [0,9]).')
    parser.add_argument('--save_path', type=str, default="./results", help='Path to save model weights and logs.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Set the identifier string for the run name.
    run_name = args.run_name

    # Path where the dataset is located.
    data_path = args.data_path

    # General training hyperparameters.
    num_gpus = args.num_gpus
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

    # Index of test set for 10-fold cross validation.
    xval_test_idx = args.xval_test_idx
    assert xval_test_idx in range(0, 10), "Cross validation split index must be an integer in range [0, 9]."

    # Set save path for logs and model weights.
    save_path = os.path.join(args.save_path, run_name)
    model_save_path = os.path.join(save_path, 'models')
    os.makedirs(model_save_path, exist_ok=True)

    # Initialize model weight filename and validation and test score log files.
    model_weight_file = '_'.join([run_name, 'fold'+str(xval_test_idx)])+'.model'
    val_score_file = os.path.join(save_path, 'val_scores.txt')
    if not os.path.exists(val_score_file):
        init_score_file(val_score_file)
    test_score_file = os.path.join(save_path, 'test_scores.txt')
    if not os.path.exists(test_score_file):
        init_score_file(test_score_file)

    if os.path.exists(os.path.join(model_save_path, model_weight_file)):
        raise Exception("The given run name and cross validation index "
                        "combination appears to already exist. Please provide "
                        "a new combination or delete the existing weight file.")

    # Initialize model and send to device.
    model = BERTClassifierModel(bert_encoder_type,
                                dim_feedforward=dim_feedforward,
                                dropout=dropout)

    # Get device; detect if there is a GPU available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda:0"):
        print("Using GPU(s).\n")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    else:
        print("Using CPU.\n")
    
    model = model.to(device)

    # Generate trees for all Reddit thread JSON files in data path.
    print("Processing dataset into tree structure:")
    all_thread_trees = process_all_trees(data_path)

    # Split the data into train-val-test sets at the thread (tree) level.
    xval_splits = get_xval_splits(all_thread_trees)
    train_data, val_data, test_data = get_train_val_test_splits(xval_splits, xval_test_idx)

    # Initialize dataset.
    reddit_dataset = RedditDataset(train_data,
                                   val_data,
                                   test_data,
                                   model.module.tokenizer,
                                   max_subtree_depth,
                                   use_ancestor_labels,
                                   randomize_prob)

    # Initialize dataloader.
    reddit_loader = torch.utils.data.DataLoader(reddit_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=0,
                                                drop_last=True)

    # Initialize optimizer.
    optimizer = optim.Adam(model.parameters(),
                            lr=lr,
                            weight_decay=weight_decay)

    # Set up learning rate scheduler.
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
                                              model_save_path,
                                              model_weight_file)

    # Train the model.
    val_scores, test_scores = RedditTrainer.train(reddit_loader, reddit_dataset, num_epochs)
    
    # Write validation and test scores to logs.
    write_scores(val_score_file, xval_test_idx, val_scores)
    write_scores(test_score_file, xval_test_idx, test_scores)
