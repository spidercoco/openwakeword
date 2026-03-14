import torch
from torch import optim, nn
import torchinfo
import torchmetrics
import copy
import os
import sys
import numpy as np
import collections
import argparse
import logging
import yaml
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import openwakeword
from openwakeword.data import mmap_batch_generator

# 设置日志
logging.basicConfig(level=logging.INFO)


# Base model class for an openwakeword model
class Model(nn.Module):
    def __init__(self, n_classes=1, input_shape=(16, 96), model_type="dnn",
                 layer_dim=128, n_blocks=1, seconds_per_example=None):
        super().__init__()

        # Store inputs as attributes
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.seconds_per_example = seconds_per_example
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.best_models = []
        self.best_model_scores = []
        self.best_val_fp = 1000
        self.best_val_accuracy = 0
        self.best_val_recall = 0
        self.best_train_recall = 0

        # Define model (currently on fully-connected network supported)
        if model_type == "dnn":
            # self.model = nn.Sequential(
            #     nn.Flatten(),
            #     nn.Linear(input_shape[0]*input_shape[1], layer_dim),
            #     nn.LayerNorm(layer_dim),
            #     nn.ReLU(),
            #     nn.Linear(layer_dim, layer_dim),
            #     nn.LayerNorm(layer_dim),
            #     nn.ReLU(),
            #     nn.Linear(layer_dim, n_classes),
            #     nn.Sigmoid() if n_classes == 1 else nn.ReLU(),
            # )

            class FCNBlock(nn.Module):
                def __init__(self, layer_dim):
                    super().__init__()
                    self.fcn_layer = nn.Linear(layer_dim, layer_dim)
                    self.relu = nn.ReLU()
                    self.layer_norm = nn.LayerNorm(layer_dim)

                def forward(self, x):
                    return self.relu(self.layer_norm(self.fcn_layer(x)))

            class Net(nn.Module):
                def __init__(self, input_shape, layer_dim, n_blocks=1, n_classes=1):
                    super().__init__()
                    self.flatten = nn.Flatten()
                    self.layer1 = nn.Linear(input_shape[0]*input_shape[1], layer_dim)
                    self.relu1 = nn.ReLU()
                    self.layernorm1 = nn.LayerNorm(layer_dim)
                    self.blocks = nn.ModuleList([FCNBlock(layer_dim) for i in range(n_blocks)])
                    self.last_layer = nn.Linear(layer_dim, n_classes)
                    self.last_act = nn.Sigmoid() if n_classes == 1 else nn.ReLU()

                def forward(self, x):
                    x = self.relu1(self.layernorm1(self.layer1(self.flatten(x))))
                    for block in self.blocks:
                        x = block(x)
                    x = self.last_act(self.last_layer(x))
                    return x
            self.model = Net(input_shape, layer_dim, n_blocks=n_blocks, n_classes=n_classes)
        elif model_type == "rnn":
            class Net(nn.Module):
                def __init__(self, input_shape, n_classes=1):
                    super().__init__()
                    self.layer1 = nn.LSTM(input_shape[-1], 64, num_layers=2, bidirectional=True,
                                          batch_first=True, dropout=0.0)
                    self.layer2 = nn.Linear(64*2, n_classes)
                    self.layer3 = nn.Sigmoid() if n_classes == 1 else nn.ReLU()

                def forward(self, x):
                    out, h = self.layer1(x)
                    return self.layer3(self.layer2(out[:, -1]))
            self.model = Net(input_shape, n_classes)

        # Define metrics
        if n_classes == 1:
            self.fp = lambda pred, y: (y-pred <= -0.5).sum()
            self.recall = torchmetrics.Recall(task='binary')
            self.accuracy = torchmetrics.Accuracy(task='binary')
        else:
            def multiclass_fp(p, y, threshold=0.5):
                probs = torch.nn.functional.softmax(p, dim=1)
                neg_ndcs = y == 0
                fp = (probs[neg_ndcs].argmax(axis=1) != 0 & (probs[neg_ndcs].max(axis=1)[0] > threshold)).sum()
                return fp

            def positive_class_recall(p, y, negative_class_label=0, threshold=0.5):
                probs = torch.nn.functional.softmax(p, dim=1)
                pos_ndcs = y != 0
                rcll = (probs[pos_ndcs].argmax(axis=1) > 0
                        & (probs[pos_ndcs].max(axis=1)[0] >= threshold)).sum()/pos_ndcs.sum()
                return rcll

            def positive_class_accuracy(p, y, negative_class_label=0):
                probs = torch.nn.functional.softmax(p, dim=1)
                pos_preds = probs.argmax(axis=1) != negative_class_label
                acc = (probs[pos_preds].argmax(axis=1) == y[pos_preds]).sum()/pos_preds.sum()
                return acc

            self.fp = multiclass_fp
            self.acc = positive_class_accuracy
            self.recall = positive_class_recall

        self.n_fp = 0
        self.val_fp = 0

        # Define logging dict (in-memory)
        self.history = collections.defaultdict(list)

        # Define optimizer and loss
        self.loss = torch.nn.functional.binary_cross_entropy if n_classes == 1 else nn.functional.cross_entropy
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def save_model(self, output_path):
        """
        Saves the weights of a trained Pytorch model
        """
        if self.n_classes == 1:
            torch.save(self.model, output_path)

    def export_to_onnx(self, output_path, class_mapping=""):
        obj = self
        # Make simple model for export based on model structure
        if self.n_classes == 1:
            # Save ONNX model
            torch.onnx.export(self.model.to("cpu"), torch.rand(self.input_shape)[None, ], output_path,
                              output_names=[class_mapping])

        elif self.n_classes >= 1:
            class M(nn.Module):
                def __init__(self):
                    super().__init__()

                    # Define model
                    self.model = obj.model.to("cpu")

                def forward(self, x):
                    return torch.nn.functional.softmax(self.model(x), dim=1)

            # Save ONNX model
            torch.onnx.export(M(), torch.rand(self.input_shape)[None, ], output_path,
                              output_names=[class_mapping])

    def lr_warmup_cosine_decay(self,
                               global_step,
                               warmup_steps=0,
                               hold=0,
                               total_steps=0,
                               start_lr=0.0,
                               target_lr=1e-3
                               ):
        # Cosine decay
        learning_rate = 0.5 * target_lr * (1 + np.cos(np.pi * (global_step - warmup_steps - hold)
                                           / float(total_steps - warmup_steps - hold)))

        # Target LR * progress of warmup (=1 at the final warmup step)
        warmup_lr = target_lr * (global_step / warmup_steps)

        # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether
        # `global_step < warmup_steps` and we're still holding.
        # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
        if hold > 0:
            learning_rate = np.where(global_step > warmup_steps + hold,
                                     learning_rate, target_lr)

        learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def forward(self, x):
        return self.model(x)

    def summary(self):
        return torchinfo.summary(self.model, input_size=(1,) + self.input_shape, device='cpu')

    def average_models(self, models=None):
        """Averages the weights of the provided models together to make a new model"""

        if models is None:
            models = self.best_models

        # Clone a model from the list as the base for the averaged model
        averaged_model = copy.deepcopy(models[0])
        averaged_model_dict = averaged_model.state_dict()

        # Initialize a running total of the weights
        for key in averaged_model_dict:
            averaged_model_dict[key] *= 0  # set to 0

        for model in models:
            model_dict = model.state_dict()
            for key, value in model_dict.items():
                averaged_model_dict[key] += value

        for key in averaged_model_dict:
            averaged_model_dict[key] /= len(models)

        # Load the averaged weights into the model
        averaged_model.load_state_dict(averaged_model_dict)

        return averaged_model

    def _select_best_model(self, false_positive_validate_data, val_set_hrs=11.3, max_fp_per_hour=0.5, min_recall=0.20):
        """
        Select the top model based on the false positive rate on the validation data

        Args:
            false_positive_validate_data (torch.DataLoader): A dataloader with validation data
            n (int): The number of models to select

        Returns:
            list: A list of the top n models
        """
        # Get false positive rates for each model
        false_positive_rates = [0]*len(self.best_models)
        for batch in false_positive_validate_data:
            x_val, y_val = batch[0].to(self.device), batch[1].to(self.device)
            for mdl_ndx, model in tqdm(enumerate(self.best_models), total=len(self.best_models),
                                       desc="Find best checkpoints by false positive rate"):
                with torch.no_grad():
                    val_ps = model(x_val)
                    false_positive_rates[mdl_ndx] = false_positive_rates[mdl_ndx] + self.fp(val_ps, y_val[..., None]).detach().cpu().numpy()
        false_positive_rates = [fp/val_set_hrs for fp in false_positive_rates]

        candidate_model_ndx = [ndx for ndx, fp in enumerate(false_positive_rates) if fp <= max_fp_per_hour]
        candidate_model_recall = [self.best_model_scores[ndx]["val_recall"] for ndx in candidate_model_ndx]
        if max(candidate_model_recall) <= min_recall:
            logging.warning(f"No models with recall >= {min_recall} found!")
            return None
        else:
            best_model = self.best_models[candidate_model_ndx[np.argmax(candidate_model_recall)]]
            best_model_training_step = self.best_model_scores[candidate_model_ndx[np.argmax(candidate_model_recall)]]["training_step_ndx"]
            logging.info(f"Best model from training step {best_model_training_step} out of {len(candidate_model_ndx)}"
                         f"models has recall of {np.max(candidate_model_recall)} and false positive rate of"
                         f" {false_positive_rates[candidate_model_ndx[np.argmax(candidate_model_recall)]]}")

        return best_model

    def auto_train(self, X_train, X_val, false_positive_val_data, steps=50000, max_negative_weight=1000,
                   target_fp_per_hour=0.2):
        """A sequence of training steps that produce relatively strong models
        automatically, based on validation data and performance targets provided.
        After training merges the best checkpoints and returns a single model.
        """

        # Get false positive validation data duration
        val_set_hrs = 11.3

        # Sequence 1
        logging.info("#"*50 + "\nStarting training sequence 1...\n" + "#"*50)
        lr = 0.0001
        weights = np.linspace(1, max_negative_weight, int(steps)).tolist()
        val_steps = np.linspace(steps-int(steps*0.25), steps, 20).astype(np.int64)
        self.train_model(
                    X=X_train,
                    X_val=X_val,
                    false_positive_val_data=false_positive_val_data,
                    max_steps=steps,
                    negative_weight_schedule=weights,
                    val_steps=val_steps, warmup_steps=steps//5,
                    hold_steps=steps//3, lr=lr, val_set_hrs=val_set_hrs)

        # Sequence 2
        logging.info("#"*50 + "\nStarting training sequence 2...\n" + "#"*50)
        lr = lr/10
        steps = steps/10

        # Adjust weights as needed based on false positive per hour performance from first sequence
        if self.best_val_fp > target_fp_per_hour:
            max_negative_weight = max_negative_weight*2
            logging.info("Increasing weight on negative examples to reduce false positives...")

        weights = np.linspace(1, max_negative_weight, int(steps)).tolist()
        val_steps = np.linspace(1, steps, 20).astype(np.int16)
        self.train_model(
                    X=X_train,
                    X_val=X_val,
                    false_positive_val_data=false_positive_val_data,
                    max_steps=steps,
                    negative_weight_schedule=weights,
                    val_steps=val_steps, warmup_steps=steps//5,
                    hold_steps=steps//3, lr=lr, val_set_hrs=val_set_hrs)

        # Sequence 3
        logging.info("#"*50 + "\nStarting training sequence 3...\n" + "#"*50)
        lr = lr/10

        # Adjust weights as needed based on false positive per hour performance from second sequence
        if self.best_val_fp > target_fp_per_hour:
            max_negative_weight = max_negative_weight*2
            logging.info("Increasing weight on negative examples to reduce false positives...")

        weights = np.linspace(1, max_negative_weight, int(steps)).tolist()
        val_steps = np.linspace(1, steps, 20).astype(np.int16)
        self.train_model(
                    X=X_train,
                    X_val=X_val,
                    false_positive_val_data=false_positive_val_data,
                    max_steps=steps,
                    negative_weight_schedule=weights,
                    val_steps=val_steps, warmup_steps=steps//5,
                    hold_steps=steps//3, lr=lr, val_set_hrs=val_set_hrs)

        # Merge best models
        logging.info("Merging checkpoints above the 90th percentile into single model...")
        accuracy_percentile = np.percentile(self.history["val_accuracy"], 90)
        recall_percentile = np.percentile(self.history["val_recall"], 90)
        fp_percentile = np.percentile(self.history["val_fp_per_hr"], 10)

        # Get models above the 90th percentile
        models = []
        for model, score in zip(self.best_models, self.best_model_scores):
            if score["val_accuracy"] >= accuracy_percentile and \
                    score["val_recall"] >= recall_percentile and \
                    score["val_fp_per_hr"] <= fp_percentile:
                models.append(model)

        if len(models) > 0:
            combined_model = self.average_models(models=models)
        else:
            combined_model = self.model

        # Report validation metrics for combined model
        with torch.no_grad():
            for batch in X_val:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                val_ps = combined_model(x)

            combined_model_recall = self.recall(val_ps, y[..., None]).detach().cpu().numpy()
            combined_model_accuracy = self.accuracy(val_ps, y[..., None].to(torch.int64)).detach().cpu().numpy()

            combined_model_fp = 0
            for batch in false_positive_val_data:
                x_val, y_val = batch[0].to(self.device), batch[1].to(self.device)
                val_ps = combined_model(x_val)
                combined_model_fp += self.fp(val_ps, y_val[..., None])

            combined_model_fp_per_hr = (combined_model_fp/val_set_hrs).detach().cpu().numpy()

        logging.info(f"\n################\nFinal Model Accuracy: {combined_model_accuracy}"
                     f"\nFinal Model Recall: {combined_model_recall}\nFinal Model False Positives per Hour: {combined_model_fp_per_hr}"
                     "\n################\n")

        return combined_model

    def predict_on_features(self, features, model=None):
        """
        Predict on Tensors of openWakeWord features corresponding to single audio clips

        Args:
            features (torch.Tensor): A Tensor of openWakeWord features with shape (batch, features)
            model (torch.nn.Module): A Pytorch model to use for prediction (default None, which will use self.model)

        Returns:
            torch.Tensor: An array of predictions of shape (batch, prediction), where 0 is negative and 1 is positive
        """
        if len(features) < 3:
            features = features[None, ]

        features = features.to(self.device)
        predictions = []
        for x in tqdm(features, desc="Predicting on clips"):
            x = x[None, ]
            batch = []
            for i in range(0, x.shape[1]-16, 1):  # step size of 1 (80 ms)
                batch.append(x[:, i:i+16, :])
            batch = torch.vstack(batch)
            if model is None:
                preds = self.model(batch)
            else:
                preds = model(batch)
            predictions.append(preds.detach().cpu().numpy()[None, ])

        return np.vstack(predictions)

    def predict_on_clips(self, clips, model=None):
        """
        Predict on Tensors of 16-bit 16 khz audio data

        Args:
            clips (np.ndarray): A Numpy array of audio clips with shape (batch, samples)
            model (torch.nn.Module): A Pytorch model to use for prediction (default None, which will use self.model)

        Returns:
            np.ndarray: An array of predictions of shape (batch, prediction), where 0 is negative and 1 is positive
        """

        # Get features from clips
        F = AudioFeatures(device='cpu', ncpu=4)
        features = F.embed_clips(clips, batch_size=16)

        # Predict on features
        preds = self.predict_on_features(torch.from_numpy(features), model=model)

        return preds

    def export_model(self, model, model_name, output_dir):
        """Saves the trained openwakeword model to both onnx and tflite formats"""

        if self.n_classes != 1:
            raise ValueError("Exporting models to both onnx and tflite with more than one class is currently not supported! "
                             "Use the `export_to_onnx` function instead.")

        # Save ONNX model
        logging.info(f"####\nSaving ONNX mode as '{os.path.join(output_dir, model_name + '.onnx')}'")
        model_to_save = copy.deepcopy(model)
        torch.onnx.export(model_to_save.to("cpu"), torch.rand(self.input_shape)[None, ],
                          os.path.join(output_dir, model_name + ".onnx"), opset_version=13)

        return None

    def train_model(self, X, max_steps, warmup_steps, hold_steps, X_val=None,
                    false_positive_val_data=None, positive_test_clips=None,
                    negative_weight_schedule=[1],
                    val_steps=[250], lr=0.0001, val_set_hrs=1):
        # Move models and main class to target device
        self.to(self.device)
        self.model.to(self.device)

        # Train model
        accumulation_steps = 1
        accumulated_samples = 0
        accumulated_predictions = torch.Tensor([]).to(self.device)
        accumulated_labels = torch.Tensor([]).to(self.device)
        for step_ndx, data in tqdm(enumerate(X, 0), total=max_steps, desc="Training"):
            # get the inputs; data is a list of [inputs, labels]
            x, y = data[0].to(self.device), data[1].to(self.device)
            y_ = y[..., None].to(torch.float32)

            # Update learning rates
            for g in self.optimizer.param_groups:
                g['lr'] = self.lr_warmup_cosine_decay(step_ndx, warmup_steps=warmup_steps, hold=hold_steps,
                                                      total_steps=max_steps, target_lr=lr)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # Get predictions for batch
            predictions = self.model(x)

            # Construct batch with only samples that have high loss
            neg_high_loss = predictions[(y == 0) & (predictions.squeeze() >= 0.001)]  # thresholds were chosen arbitrarily but work well
            pos_high_loss = predictions[(y == 1) & (predictions.squeeze() < 0.999)]
            y = torch.cat((y[(y == 0) & (predictions.squeeze() >= 0.001)], y[(y == 1) & (predictions.squeeze() < 0.999)]))
            y_ = y[..., None].to(torch.float32)
            predictions = torch.cat((neg_high_loss, pos_high_loss))

            # Set weights for batch
            if len(negative_weight_schedule) == 1:
                w = torch.ones(y.shape[0])*negative_weight_schedule[0]
                pos_ndcs = y == 1
                w[pos_ndcs] = 1
                w = w[..., None]
            else:
                if self.n_classes == 1:
                    w = torch.ones(y.shape[0])*negative_weight_schedule[step_ndx]
                    pos_ndcs = y == 1
                    w[pos_ndcs] = 1
                    w = w[..., None]

            if predictions.shape[0] != 0:
                # Do backpropagation, with gradient accumulation if the batch-size after selecting high loss examples is too small
                loss = self.loss(predictions, y_ if self.n_classes == 1 else y, w.to(self.device))
                loss = loss/accumulation_steps
                accumulated_samples += predictions.shape[0]

                if predictions.shape[0] >= 128:
                    accumulated_predictions = predictions
                    accumulated_labels = y_
                if accumulated_samples < 128:
                    accumulation_steps += 1
                    accumulated_predictions = torch.cat((accumulated_predictions, predictions))
                    accumulated_labels = torch.cat((accumulated_labels, y_))
                else:
                    loss.backward()
                    self.optimizer.step()
                    accumulation_steps = 1
                    accumulated_samples = 0

                    self.history["loss"].append(loss.detach().cpu().numpy())

                    # Compute training metrics and log them
                    fp = self.fp(accumulated_predictions, accumulated_labels if self.n_classes == 1 else y)
                    self.n_fp += fp
                    self.history["recall"].append(self.recall(accumulated_predictions, accumulated_labels).detach().cpu().numpy())

                    accumulated_predictions = torch.Tensor([]).to(self.device)
                    accumulated_labels = torch.Tensor([]).to(self.device)

            # Run validation and log validation metrics
            if step_ndx in val_steps and step_ndx > 1 and false_positive_val_data is not None:
                # Get false positives per hour with false positive data
                val_fp = 0
                for val_step_ndx, data in enumerate(false_positive_val_data):
                    with torch.no_grad():
                        x_val, y_val = data[0].to(self.device), data[1].to(self.device)
                        val_predictions = self.model(x_val)
                        val_fp += self.fp(val_predictions, y_val[..., None])
                val_fp_per_hr = (val_fp/val_set_hrs).detach().cpu().numpy()
                self.history["val_fp_per_hr"].append(val_fp_per_hr)

            # Get recall on test clips
            if step_ndx in val_steps and step_ndx > 1 and positive_test_clips is not None:
                tp = 0
                fn = 0
                for val_step_ndx, data in enumerate(positive_test_clips):
                    with torch.no_grad():
                        x_val = data[0].to(self.device)
                        batch = []
                        for i in range(0, x_val.shape[1]-16, 1):
                            batch.append(x_val[:, i:i+16, :])
                        batch = torch.vstack(batch)
                        preds = self.model(batch)
                        if any(preds >= 0.5):
                            tp += 1
                        else:
                            fn += 1
                self.history["positive_test_clips_recall"].append(tp/(tp + fn))

            if step_ndx in val_steps and step_ndx > 1 and X_val is not None:
                # Get metrics for balanced test examples of positive and negative clips
                for val_step_ndx, data in enumerate(X_val):
                    with torch.no_grad():
                        x_val, y_val = data[0].to(self.device), data[1].to(self.device)
                        val_predictions = self.model(x_val)
                        val_recall = self.recall(val_predictions, y_val[..., None]).detach().cpu().numpy()
                        val_acc = self.accuracy(val_predictions, y_val[..., None].to(torch.int64))
                        val_fp = self.fp(val_predictions, y_val[..., None])
                self.history["val_accuracy"].append(val_acc.detach().cpu().numpy())
                self.history["val_recall"].append(val_recall)
                self.history["val_n_fp"].append(val_fp.detach().cpu().numpy())

            # Save models with a validation score above/below the 90th percentile
            # of the validation scores up to that point
            if step_ndx in val_steps and step_ndx > 1:
                if self.history["val_n_fp"][-1] <= np.percentile(self.history["val_n_fp"], 50) and \
                   self.history["val_recall"][-1] >= np.percentile(self.history["val_recall"], 5):
                    # logging.info("Saving checkpoint with metrics >= to targets!")
                    self.best_models.append(copy.deepcopy(self.model))
                    self.best_model_scores.append({"training_step_ndx": step_ndx, "val_n_fp": self.history["val_n_fp"][-1],
                                                   "val_recall": self.history["val_recall"][-1],
                                                   "val_accuracy": self.history["val_accuracy"][-1],
                                                   "val_fp_per_hr": self.history.get("val_fp_per_hr", [0])[-1]})
                    self.best_val_recall = self.history["val_recall"][-1]
                    self.best_val_accuracy = self.history["val_accuracy"][-1]

            if step_ndx == max_steps-1:
                break


# ================= 主程序 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r').read(), yaml.Loader)

    task_root = os.path.dirname(os.path.abspath(args.config))
    
    # 路径解析
    pos_train_feat = os.path.join(task_root, "positive_features_train.npy")
    neg_train_feat = os.path.join(task_root, "negative_features_train.npy")
    pos_test_feat = os.path.join(task_root, "positive_features_test.npy")
    neg_test_feat = os.path.join(task_root, "negative_features_test.npy")

    F = openwakeword.utils.AudioFeatures(device='cpu')
    input_shape = np.load(pos_test_feat).shape[1:]

    oww = Model(n_classes=1, input_shape=input_shape, model_type=config.get("model_type", "dnn"),
                layer_dim=config.get("layer_size", 128), seconds_per_example=1280*input_shape[0]/16000)

    # Create data transform function for batch generation to handle differ clip lengths (todo: write tests for this)
    def f(x, n=input_shape[0]):
        """Simple transformation function to ensure negative data is the appropriate shape for the model size"""
        if n > x.shape[1] or n < x.shape[1]:
            x = np.vstack(x)
            new_batch = np.array([x[i:i+n, :] for i in range(0, x.shape[0]-n, n)])
        else:
            return x
        return new_batch

    ## Create label transforms as needed for model (currently only supports binary classification models)
    data_transforms = {key: f for key in config["feature_data_files"].keys()}
    label_transforms = {}
    for key in ["positive"] + list(config["feature_data_files"].keys()) + ["adversarial_negative"]:
        if key == "positive":
            label_transforms[key] = lambda x: [1 for i in x]
        else:
            label_transforms[key] = lambda x: [0 for i in x]

    base_project_root  = os.path.dirname(os.path.abspath(__file__))
    def get_abs_path(path):
        # 移除 ./ 前缀
        clean_p = path.replace("./", "")
        return os.path.join(base_project_root, clean_p)

    print(get_abs_path(config["feature_data_files"]['ACAV100M_sample']))

    # Add generated positive and adversarial negative clips to the feature data files dictionary
    config["feature_data_files"]['positive'] = pos_train_feat
    config["feature_data_files"]['adversarial_negative'] = neg_train_feat
    config["feature_data_files"]['ACAV100M_sample'] = get_abs_path(config["feature_data_files"]['ACAV100M_sample'])
    config['false_positive_validation_data_path'] = get_abs_path(config['false_positive_validation_data_path'])

    print(config)
    
    # Make PyTorch data loaders for training and validation data
    batch_generator = mmap_batch_generator(
        config["feature_data_files"],
        n_per_class=config["batch_n_per_class"],
        data_transform_funcs=data_transforms,
        label_transform_funcs=label_transforms
    )

    class IterDataset(torch.utils.data.IterableDataset):
        def __init__(self, generator):
            self.generator = generator
    
        def __iter__(self):
            return self.generator
    
    n_cpus = os.cpu_count()
    if n_cpus is None:
        n_cpus = 1
    else:
        n_cpus = n_cpus//2
    X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                          batch_size=None, num_workers=n_cpus, prefetch_factor=16)

    class IterDataset(torch.utils.data.IterableDataset):
        def __init__(self, generator): self.generator = generator
        def __iter__(self): return self.generator

    X_train = torch.utils.data.DataLoader(IterDataset(batch_generator), batch_size=None)

    X_val_fp = np.load(config["false_positive_validation_data_path"])
    X_val_fp = np.array([X_val_fp[i:i+input_shape[0]] for i in range(0, X_val_fp.shape[0]-input_shape[0], 1)])  # reshape to match model
    X_val_fp_labels = np.zeros(X_val_fp.shape[0]).astype(np.float32)
    X_val_fp = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_val_fp), torch.from_numpy(X_val_fp_labels)),
        batch_size=len(X_val_fp_labels)
    )

    X_val_pos = np.load(pos_test_feat)
    X_val_neg = np.load(neg_test_feat)
    labels = np.hstack((np.ones(X_val_pos.shape[0]), np.zeros(X_val_neg.shape[0]))).astype(np.float32)

    X_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(np.vstack((X_val_pos, X_val_neg))),
            torch.from_numpy(labels)
            ),
        batch_size=len(labels)
    )
    # 运行训练
    best_model = oww.auto_train(
        X_train=X_train,
        X_val=X_val,
        false_positive_val_data=X_val_fp,
        steps=config["steps"],
        max_negative_weight=config["max_negative_weight"],
        target_fp_per_hour=config["target_false_positives_per_hour"],
    )

    # 导出
    oww.export_model(model=best_model, model_name=config["model_name"], output_dir=task_root)

    # 保存最终指标
    final_metrics = {
        "recall": float(oww.history.get("val_recall", [0])[-1]),
        "accuracy": float(oww.history.get("val_accuracy", [0])[-1]),
        "fp_per_hr": float(oww.history.get("val_fp_per_hr", [0])[-1]),
        "timestamp": json.dumps(json.dumps(str(torch.utils.data.datetime.now())))
    }
    # 尝试更准确地获取最后一步指标（如果 history 为空则使用 oww 属性）
    if not final_metrics["recall"]: final_metrics["recall"] = float(oww.best_val_recall)
    if not final_metrics["accuracy"]: final_metrics["accuracy"] = float(oww.best_val_accuracy)

    with open(os.path.join(task_root, "metrics.json"), "w") as f:
        json.dump(final_metrics, f)

    # 保存图表与指标
    plt.figure()
    plt.plot(oww.history['loss'], label="loss")
    plt.plot(oww.history['recall'], label="recall")
    plt.legend(); plt.ylim(0, 1.1)
    plt.savefig(os.path.join(task_root, "training_plot.png"))
    

if __name__ == "__main__":
    main()
