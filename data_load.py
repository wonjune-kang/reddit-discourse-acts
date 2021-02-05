import torch
from torch.utils.data import Dataset

from format_reddit_data import process_all_trees
from reddit_tokenizer import LABELS2IDX


class RedditDataset(Dataset):
    def __init__(self, train_data, val_data, test_data, tokenizer,
                 max_subtree_depth, use_ancestor_labels, randomize_prob):
        # Split data into train, val, test sets at the tree (thread) level.
        self.train_trees = train_data
        self.val_trees = val_data
        self.test_trees = test_data

        # Tokenizer for BERT encoder.
        self.tokenizer = tokenizer

        # Denotes the maximum number of ancestors to consider for context
        # information.
        self.max_subtree_depth = max_subtree_depth

        # Flag for whether to include ancestor labels tokens in inputs.
        self.use_ancestor_labels = use_ancestor_labels

        # Flag for whether to randomize ancestor labels during training or not.
        self.randomize_prob = randomize_prob

        # Get list of all post nodes in the dataset for length and indexing.
        self.train_nodes = [node for tree in self.train_trees for node in tree.nodes.values()]
        self.val_nodes = [node for tree in self.val_trees for node in tree.nodes.values()]
        self.test_nodes = [node for tree in self.test_trees for node in tree.nodes.values()]

        # Get the label index data for all nodes.
        self.train_labels = [LABELS2IDX[node.label] for node in self.train_nodes]
        self.val_labels = [LABELS2IDX[node.label] for node in self.val_nodes]
        self.test_labels = [LABELS2IDX[node.label] for node in self.test_nodes]

    def __len__(self):
        # Only the length of the train dataset is needed for the dataloader.
        return len(self.train_nodes)

    def __getitem__(self, idx):
        # Choose a node from the training set.
        node = self.train_nodes[idx]

        # Compute the input strings for the target node and context string
        # for its ancestors, and feed into the tokenizer.
        target_string, context_string = get_subtree_string(node,
                                                           self.max_subtree_depth,
                                                           self.use_ancestor_labels,
                                                           self.randomize_prob,
                                                           eval=False)
        tokenized_subtree = self.tokenizer(target_string,
                                           context_string,
                                           truncation=True,
                                           padding='max_length')

        # Convert the encoded tokenization into dictionary format and
        # add label information to the dictionary.
        encoding = {k: torch.tensor(v) for k, v in tokenized_subtree.items()}
        encoding['labels'] = torch.tensor(self.train_labels[idx])

        return encoding