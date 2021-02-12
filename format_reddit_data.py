import os
import json
from tqdm import tqdm


class RedditNode:
    """
    Represents the information in a single post in a Reddit thread.
    """
    def __init__(self, tree, post_id, parent_id, body, label):
        # The RedditTree object that the node belongs to.
        self.tree = tree

        # The post ID, used as a unique identifier for the node.
        self.post_id = post_id

        # The post ID of the node's immediate parent.
        self.parent_id = parent_id

        # The parent RedditNode object; is None if the node has no parent
        # (i.e. is the root of the tree).
        self.parent = self.tree.nodes[self.parent_id] if self.parent_id is not None else None

        # List of children RedditNode objects.
        self.children = []

        # The body text of the post.
        self.body = body

        # The ground truth label string.
        self.label = label

        # Placeholder for storing prediction strings for the node.
        self.pred = None


class RedditTree:
    """
    Represents a single Reddit thread as a tree of RedditNode objects.
    """
    def __init__(self, title):
        # Title of Reddit thread (string).
        self.title = title

        # Dictionary mapping post IDs to RedditNode objects.
        self.nodes = {}

        # Root node of tree.
        self.root = None

    # Add RedditNode object for a comment to tree's node dictionary, 
    # whose key is the post ID.
    def add_node(self, node_id, node_obj):
        self.nodes[node_id] = node_obj

    # Find the path from a node to the root of the tree, in bottom up fashion.
    def traverse_to_root(self, node_id):
        path_nodes = []
        current_node = self.nodes[node_id]
        while current_node.parent_id is not None:
            path_nodes.append(current_node)
            current_node = self.nodes[current_node.parent_id]
        path_nodes.append(current_node)
        return path_nodes 


def generate_tree(thread_data):
    """
    Generate a RedditTree object from JSON of thread data.
    """
    title = thread_data["title"]    
    thread_tree = RedditTree(title)

    posts = thread_data["posts"]
    for post in posts:
        # Flag for checking whether to add the post as a node or not.
        remove = False

        # Use the post ID as a unique identifier for the node.
        post_id = post["id"]

        # Check if the post is the root of the tree.
        is_root = "is_first_post" in post

        # If the post isn't the root, use the post's majority link as the
        # parent comment if it exists. Otherwise, use the comment that it
        # responded to as the response label.
        if is_root:
            parent_id = None
        elif "majority_link" in post and post["majority_link"] != "none":
            parent_id = post["majority_link"]
        else:
            parent_id = post["in_reply_to"]

        # Check if the post has a body; flag for removal if it does not.
        if "body" in post:
            body = post["body"]
            if body == "[deleted]":
                remove = True
        else:
            remove = True
        
        # Check if the post has a majority type label; flag for removal if not.
        if "majority_type" in post and post["majority_type"] != "other":
            label = post["majority_type"]
        else:
            remove = True

        # If the post has all valid labels, add a node to the tree object.
        if (is_root or parent_id in thread_tree.nodes) and remove == False:
            post_node = RedditNode(thread_tree, post_id, parent_id, body, label)
            thread_tree.add_node(post_id, post_node)

            # If the node is the root, set it as the tree's root.
            # Otherwise, add it to its parent's children.
            if is_root:
                thread_tree.root = post_node
            else:
                thread_tree.nodes[parent_id].children.append(post_node)

    return thread_tree

def process_all_trees(data_directory):
    """
    Generate and return a list of RedditTree objects for all thread JSON files
    in the given data directory.
    """
    all_trees = []
    for filename in tqdm(sorted(os.listdir(data_directory))):
        json_file = os.path.join(data_directory, filename)
        with open(json_file, 'r') as f:
            thread_data = json.load(f)

        posts = thread_data["posts"]

        # Generate the RedditTree.
        tree = generate_tree(thread_data)
        
        # Use the tree only if it corresponds to a Reddit thread with
        # at least one valid response in addition to the initial post.
        if len(tree.nodes) > 1:
            all_trees.append(tree)

    return all_trees
