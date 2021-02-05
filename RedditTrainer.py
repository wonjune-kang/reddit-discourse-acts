import time
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score

from reddit_tokenizer import LABELS2IDX, IDX2LABELS
from utils import get_subtree_string


class RedditDiscourseActTrainer:
    def __init__(self, run_name, model, device, optimizer, scheduler, save_directory):
        self.run_name = run_name
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_directory = save_directory

    def predict_node(self, target_node, max_subtree_depth, use_ancestor_labels):
        """
        Makes a prediction for a single RedditNode in a RedditTree object and
        assigns it to the node's pred attribute.
        """
        # Compute strings for the target node and its context information
        # and feed into the tokenizer.
        target_string, context_string = get_subtree_string(target_node,
                                                           max_subtree_depth,
                                                           use_ancestor_labels,
                                                           eval=True)
        encoded_subtree = self.model.tokenizer(target_string, context_string,
                                               truncation=True,
                                               padding='max_length',
                                               return_tensors='pt')

        # Send the model inputs to the device.
        input_ids = encoded_subtree['input_ids'].to(self.device)
        attention_mask = encoded_subtree['attention_mask'].to(self.device)
        token_type_ids = encoded_subtree['token_type_ids'] if 'token_type_ids' in encoded_subtree else None

        # Feed the inputs through the model and get the prediction.
        logits = self.model(input_ids, attention_mask, token_type_ids)
        pred = torch.argmax(logits).item()

        # Assign the string label for the prediction to the target node's
        # pred attribute.
        pred_text = IDX2LABELS[pred]
        target_node.pred = pred_text
        return

    def predict_tree(self, tree, max_subtree_depth, use_ancestor_labels):
        """
        Uses breadth-first search (BFS) to compute the predictions for all nodes
        in a RedditTree in top-down fashion, starting from the root node.
        """
        root_node = tree.root
        queue = [root_node]
        while len(queue) > 0:
            self.predict_node(queue[0], max_subtree_depth, use_ancestor_labels)
            node = queue.pop(0)
            queue.extend(node.children)
        return

    def evaluate(self, dataset, eval_set='val'):
        """
        Evaluate the model's performance on the dataset's validation split.
        Make predictions for all RedditTree objects in the validation split
        and compute the resulting node-level accuracy.
        """
        if eval_set == 'val':
            eval_trees = dataset.val_trees
            eval_nodes = dataset.val_nodes
        elif eval_set == 'test':
            eval_trees = dataset.test_trees
            eval_nodes = dataset.test_nodes
        else:
            raise Exception("Must evaluate on either 'val' or 'test' set.")

        # Make predictions for all trees in the validation split.
        for tree in eval_trees:
            self.predict_tree(tree, dataset.max_subtree_depth, dataset.use_ancestor_labels)

        # Iterate through all nodes and compute accuracy. Maintain lists of
        # labels and predictions for computing precision, recall, and F1 score.
        eval_labels = []
        eval_pred = []
        correct = 0
        for node in eval_nodes:
            if node.label == node.pred:
                correct += 1
            eval_labels.append(LABELS2IDX[node.label])
            eval_pred.append(LABELS2IDX[node.pred])
        accuracy = correct/len(eval_nodes)

        return eval_labels, eval_pred, accuracy

    def save_model_weights(self):
        self.model.eval().cpu()
        save_filename = self.run_name+".model"
        save_path = os.path.join(self.save_directory, save_filename)
        torch.save(self.model.state_dict(), save_path)

    def train(self, dataloader, dataset, num_epochs):
        output_interval = len(dataloader) // 10
        criterion = nn.CrossEntropyLoss()

        best_val_accuracy, best_test_accuracy = 0.0, 0.0
        best_val_precision, best_test_precision = 0.0, 0.0
        best_val_recall, best_test_recall = 0.0, 0.0
        best_val_f1, best_test_f1 = 0.0, 0.0
        for epoch in range(num_epochs):
            print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            # Training phase
            self.model.train()

            train_loss = 0.0
            train_correct = 0
            train_samples = 0
            train_labels = []
            train_pred = []
            for batch_id, batch_encodings in enumerate(dataloader):
                input_ids = batch_encodings['input_ids'].to(self.device)
                attention_mask = batch_encodings['attention_mask'].to(self.device)
                token_type_ids = batch_encodings['token_type_ids'] if 'token_type_ids' in batch_encodings else None
                labels = batch_encodings['labels'].to(self.device)

                # Zero parameter gradients.
                self.optimizer.zero_grad()

                # Pass the inputs through the model.
                outputs = self.model(input_ids, attention_mask, token_type_ids)

                # Backpropagate loss.
                loss = criterion(outputs, labels)
                loss.backward()

                # # Clip gradients.
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Step optimizer and learning rate scheduler.
                self.optimizer.step()
                self.scheduler.step()

                train_loss += loss.item()

                # Compute number of correct predictions in batch.
                _, pred = torch.max(outputs.data, dim=1)
                correct = (pred == labels).float().sum()
                train_correct += correct
                train_samples += len(pred)

                # Maintain a running list of labels and predictions for
                # computing precision,recall, and F1 score.
                train_labels.extend(labels.cpu().numpy().tolist())
                train_pred.extend(pred.cpu().numpy().tolist())

                # Print running loss and accuracy for the current epoch.
                if (batch_id+1) % output_interval == 0:
                    msg = "{0}\tBatch: {1}/{2}\tBatch Loss: {3:.4f}\tAvg Loss/Batch: {4:.4f}\tRunning Acc: {5:.4f}".format(
                            time.ctime(), batch_id+1, len(dataset)//batch_size,
                            loss, train_loss/(batch_id+1), train_correct/train_samples)
                    print(msg)

            # Compute and print statistics for training epoch.
            avg_train_loss = train_loss / (train_samples // batch_size)
            train_accuracy = train_correct / train_samples
            train_precision = precision_score(train_labels, train_pred, average='weighted')
            train_recall = recall_score(train_labels, train_pred, average='weighted')
            train_f1 = f1_score(train_labels, train_pred, average='weighted')
            
            print("\nAverage train loss per batch: {:.4f}".format(avg_train_loss))
            print("Train accuracy: {:.4f}".format(train_accuracy))
            print("Train precision: {:.4f}".format(train_precision))
            print("Train recall: {:.4f}".format(train_recall))
            print("Train F1 score: {:.4f}".format(train_f1))

            # Validation and evaluation phase.
            with torch.no_grad():
                self.model.eval()

                # Evaluate on the dataset's validation split.
                val_labels, val_pred, val_accuracy = self.evaluate(dataset, eval_set='val')

                # Compute and print statistics for validation split.
                val_precision = precision_score(val_labels, val_pred, average='weighted')
                val_recall = recall_score(val_labels, val_pred, average='weighted')
                val_f1 = f1_score(val_labels, val_pred, average='weighted')

                print("\nValidation accuracy: {:.4f}".format(val_accuracy))
                print("Validation precision: {:.4f}".format(val_precision))
                print("Validation recall: {:.4f}".format(val_recall))
                print("Validation F1 score: {:.4f}".format(val_f1))

            # Save model weights and evaluate on test split if best
            # validation F1 score.
            if val_f1 > best_val_f1:
                print("\nAchieved best validation F1 score. Evaluating on test set.")            

                best_val_accuracy = val_accuracy
                best_val_precision = val_precision
                best_val_recall = val_recall
                best_val_f1 = val_f1

                test_labels, test_pred, test_accuracy = self.evaluate(dataset, eval_set='test')

                # Compute and print statistics for test split.
                test_precision = precision_score(test_labels, test_pred, average='weighted')
                test_recall = recall_score(test_labels, test_pred, average='weighted')
                test_f1 = f1_score(test_labels, test_pred, average='weighted')

                best_test_accuracy = test_accuracy
                best_test_precision = test_precision
                best_test_recall = test_recall
                best_test_f1 = test_f1

                print("\nTest accuracy: {:.4f}".format(test_accuracy))
                print("Test precision: {:.4f}".format(test_precision))
                print("Test recall: {:.4f}".format(test_recall))
                print("Test F1 score: {:.4f}".format(test_f1))

                if self.save_directory is not None:
                    print("Saving model weights to", self.save_directory)
                    self.save_model_weights()
                    self.model.to(device).train()

        # # Save final model
        # model.eval().cpu()
        # save_filename = "final_epoch_" + str(epoch) + ".model"
        # save_model_path = os.path.join(ckpt_directory, save_filename)
        # torch.save(model.state_dict(), save_model_path)
        # print("\nTraining done. Final model parameters saved at", save_model_path)