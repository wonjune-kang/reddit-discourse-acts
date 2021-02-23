import numpy as np
import torch.optim as optim

from reddit_tokens import LABELS2TOKENS, IDX2LABELS


def get_xval_splits(all_data, k=10, random_seed=0):
    """
    Shuffle the data and split into k(=10) folds for cross validation.
    """
    np.random.seed(random_seed)
    np.random.shuffle(all_data)
    xval_splits = np.array_split(all_data, k)
    return xval_splits

def get_train_val_test_splits(xval_splits, test_idx):
    """
    Given the 10 folds from the dataset, split it into train-val-test sets
    in an 80-10-10 ratio. The validation set index is set as test_idx - 1.
    """
    assert test_idx >= 0, "This function was written to work for positive test_idx values."

    val_idx = test_idx - 1
    test_data = xval_splits[test_idx]
    val_data = xval_splits[val_idx]
    if test_idx == 0:
        train_data = np.concatenate(xval_splits[test_idx+1:val_idx])
    else:
        train_data = np.concatenate(xval_splits[:val_idx] + xval_splits[test_idx+1:])

    return train_data, val_data, test_data

def get_subtree_string(target_node, max_subtree_depth, use_ancestor_label=True,
                       randomize_prob=0.1, eval=False):
    """
    Get the strings corresponding to the target node and the context information
    from its ancestors that will be tokenized and fed into the model.
    target_string is the text of the target node's body.
    context_string has format:
    "[CTXT1_LABEL] context1_text [CTXTSEP] [CTXT2_LABEL] context2_text ..."
    """
    tree = target_node.tree

    # Get a list of nodes denoting the subtree ending at the target node.
    # List is in bottom-up order, i.e. index 0 is the target node and
    # index -1 is the root of the tree.
    subtree = tree.traverse_to_root(target_node.post_id)

    # Consider only the closest max_subtree_depth ancestors.
    ancestors = subtree[1:max_subtree_depth+1]

    # Process the ancestor text by adding special tokens for context start,
    # ancestor node label, and context end. Results in a string as follows:
    # "[LABEL_TOKEN] ancestor_text [CTXTSEP]"
    ancestor_contents = []
    for i, ancestor_node in enumerate(ancestors):
        # If including ancestor node labels as context information:
        if use_ancestor_label:
            # For inference, use the predicted label.
            if eval:
                label = ancestor_node.pred
                assert label is not None, "There must be a valid node label"
            
            # For training:
            else:
                # Use the ground truth label with probability 1 - randomize_prob
                # and a random label with probability randomize_prob.
                use_true_label = np.random.uniform(0.0, 1.0)
                if use_true_label < 1.0 - randomize_prob:
                    label = ancestor_node.label
                else:
                    rand_label_idx = np.random.randint(0, len(IDX2LABELS))
                    label = IDX2LABELS[rand_label_idx]
            
            # Add the label token to the start of the post body string.
            label_token = LABELS2TOKENS[label]
            processed = ' '.join([label_token, ancestor_node.body])
        
        # Otherwise, just use the node text.
        else:
            processed = ancestor_node.body
        
        # Add [CTXTSEP] token.
        if i < len(ancestors) - 1:
            processed = ' '.join([processed, LABELS2TOKENS['context_sep']])
        
        ancestor_contents.append(processed)

    target_string = target_node.body
    context_string = ' '.join(ancestor_contents)

    return target_string, context_string