# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PyTorch BERT model. The following code is inspired by the HuggingFace version
and it is built upon in places (license below). In particular it can be
used and modified for commercial use:
https://github.com/huggingface/pytorch-pretrained-BERT/
This code will run the BERT multitask learning training and evaluation scripts
based on the run_config file
"""
import json
import logging
from pathlib import Path
import random

import torch

# from pytorch_pretrained_bert import BertTokenizer TODO: Remove when open sourcing code
from bert_sb import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from tensorboardX import SummaryWriter

import numpy as np
from itertools import cycle
from tqdm import trange

from data_processing import SST2Processor, SST5Processor, IMDBProcessor, SemEval_QA_BProcessor, SemEval_QA_MProcessor
from extract_features import convert_examples_to_features
from modelling import MultiTaskModel, MODEL_NAMES

CONFIGS_FOLDER = Path(__file__).resolve().parent / 'configs'
RUN_CONFIG_DIR = CONFIGS_FOLDER / 'run_config.json'

with open(RUN_CONFIG_DIR, 'r') as file_dir:
    RUN_CONFIG = json.load(file_dir)

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    level=logging.INFO)
LOGGER = logging.getLogger(__name__)

TOKENIZERS = {'BERT': BertTokenizer}
PROCESSORS = {'SST-2': SST2Processor, 'SST-5': SST5Processor,
              'IMDB': IMDBProcessor, 'SemEval_QA_B': SemEval_QA_BProcessor,
              'SemEval_QA_M': SemEval_QA_MProcessor}
TASK_WEIGHTINGS = ['num_examples_per_task', 'importance']
SAMPLING_MODES = ['sequential', 'random', 'prop', 'sqrt', 'anneal']
# TODO: Other sampling modes (potentially): squaring, reverse anneal

class MultiTaskLearning:
    """
    Wrapper for running Multi Task Learning built on top of HuggingFace's PyTorch implementation
    """
    def __init__(self, run_config):
        """
        Initialise the model with the run_config

        Parameters
        ----------
        run_config : dict
            Settings for the training run/loop (see below for parameters)
        """
        # Set up the run config
        run_config["data_dir"] = run_config.get("data_dir", None)
        if run_config["data_dir"] is None:
            raise ValueError("""Please enter {'data_dir': 'path/to/data'} in
                             the run config JSON file""")

        run_config["tasks"] = run_config.get("tasks", "SST-2")
        run_config["model_name"] = run_config.get("model_name", None)
        run_config["model_config"] = run_config.get("model_config", None)
        if run_config["model_name"] is None:
            if run_config['model_config'] is None:
                raise ValueError("You must enter one of (model_name, model_config) to the run config")
            if not isinstance(run_config["model_config"], dict):
                raise TypeError(f"The model config must be a dict! You entered type: {type(run_config['model_config'])}")
        else:
            run_config["model_name"] = run_config["model_name"].lower()
        run_config["base_params_to_unfreeze"] = run_config.get("base_params_to_unfreeze", "all")
        # TODO: Can we infer max_seq_length from the data?
        run_config["max_seq_length"] = run_config.get("max_seq_length", 128)
        run_config["train_batch_size"] = run_config.get("train_batch_size", 24)
        run_config["dev_batch_size"] = run_config.get("dev_batch_size", 24)
        run_config["learning_rate"] = run_config.get("learning_rate", 2e-5)
        run_config["warmup_prop"] = run_config.get("warmup_prop", 0.1)
        run_config["num_epochs"] = run_config.get("num_epochs", 5)
        run_config["sampling_mode"] = run_config.get("sampling_mode", "sequential")
        if run_config["sampling_mode"] == "anneal":
            run_config["anneal_constant"] = run_config.get("anneal_constant", 0.9)
        run_config["task_weightings"] = run_config.get("task_weightings", "importance")
        run_config["steps_to_log"] = run_config.get("steps_to_log", 500)
        run_config["seed"] = run_config.get("seed", 42)

        self.run_config = run_config

        # Check the base model is a valid name
        if self.run_config["model_name"] not in MODEL_NAMES:
            raise ValueError(f"Please enter a valid model name - you entered {self.run_config['model_name']} "
                             f"- try one of {MODEL_NAMES}")
        else:
            self.baseLM_model_name = self.run_config["model_name"].split("-")[0].upper()
        # Get the device we are working on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()

        # TODO: Implement distributed/multiple GPU support if running in the cloud/open sourcing?

        # Set the random seed for somewhat reproducible results
        random.seed(run_config["seed"])
        np.random.seed(run_config["seed"])
        torch.manual_seed(run_config["seed"])
        if n_gpu > 0:
            torch.cuda.manual_seed_all(run_config["seed"])

        # Get some useful values and log our task setup according to run_config
        self.task_names = list(self.run_config["tasks"].split(", "))
        self.num_tasks = len(self.task_names)

        if any(task_name not in PROCESSORS.keys() for task_name in self.task_names):
            raise ValueError(f"Non implemented task string in run_config please try one of {PROCESSORS.keys()}")

        LOGGER.info(f"Number of tasks: {self.num_tasks}, "
                    f"Name(s): {', '.join(self.task_names)}")

        # Set up variables for loading in the data via load_data
        self.data_dirs = [Path(self.run_config["data_dir"]) / task_name if task_name.split("_")[0] != 'SemEval'
                          else Path(self.run_config["data_dir"]) / 'SemEval' for task_name in self.task_names]
        self.train_loaders = []
        self.dev_loaders = []
        self.train_examples_per_task = [None] * self.num_tasks  # Updated in load_data()
        self.data_loaded = False
        self.processor_list = [PROCESSORS[task_name](data_dir)
                               for task_name, data_dir in zip(self.task_names, self.data_dirs)]

        # Get our specific task configs based on the tasks we want to run
        self.task_configs = {task_name: {"num_labels": processor.num_labels,
                                         "task_type": processor.task_type,
                                         "output_type": processor.output_type}
                             for task_name, processor in zip(self.task_names, self.processor_list)}

        # Create the writer, logging key hyperparameters in the log name and all parameters as text
        hparams_in_logname = {'LR': self.run_config["learning_rate"], 'SM': self.run_config["sampling_mode"],
                              'BS': self.run_config["train_batch_size"], 'Tasks': "|".join(self.task_names)}
        logname = '_'.join('{}_{}'.format(*param) for param in sorted(hparams_in_logname.items()))
        self.writer = SummaryWriter(comment=logname)
        for config_param, config_value in self.run_config.items():
            self.writer.add_text(str(config_param), str(config_value))

        # Initialise the tokenizer based on the model fed in
        do_lower_case = False if self.run_config["model_name"].split("-")[-1] == "cased" else True
        self.tokenizer = TOKENIZERS[self.baseLM_model_name].from_pretrained(run_config["model_name"], do_lower_case=do_lower_case)

        # Instantiate the model and save it to the device (CPU or GPU)
        model_name_or_config = (self.run_config["model_config"] if self.run_config["model_config"] is not None
                                 else self.run_config["model_name"])
        self.model = MultiTaskModel(task_configs=self.task_configs, model_name_or_config=model_name_or_config)
        self.model.unfreeze_base_layers(base_params_to_unfreeze=self.run_config["base_params_to_unfreeze"])
        self.model.to(self.device)

        model_params = sum(param.numel() for param in self.model.parameters())
        trained_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        LOGGER.info(f"Model initialised with {model_params:3e} parameters , of which {trained_params:3e} "
                    f"are trainable i.e. {100 * trained_params/model_params:.3f}%")

        # Task weighting mode verification
        if self.run_config["task_weightings"] not in TASK_WEIGHTINGS:
            raise ValueError(f"Non implemented task weighting mode in run_config, please try one of {TASK_WEIGHTINGS}")
        self.task_weightings_mode = self.run_config["task_weightings"]

        # Sampling mode verification
        if self.run_config["sampling_mode"] not in SAMPLING_MODES:
            raise ValueError(f"Non implemented sampling mode in run_config, please try one of {SAMPLING_MODES}")
        self.sampling_mode = self.run_config["sampling_mode"]

    def load_data(self):
        """
        Loads training and test data for the tasks given in run_config
        into PyTorch DataLoader iterators
        """
        # Load the training and dev examples for each task
        label_list = [processor.get_labels() for processor in self.processor_list]
        train_examples = [processor.get_examples(set_type="train") for processor in self.processor_list]
        self.train_examples_per_task = [len(train_examples[task]) for task in range(self.num_tasks)]
        dev_examples = [processor.get_examples(set_type="dev") for processor in self.processor_list]

        train_bs = self.run_config["train_batch_size"]
        dev_bs = self.run_config["dev_batch_size"]
        max_seq_length = self.run_config["max_seq_length"]

        # TODO: Add in DistributedSampler if using more than 1 gpu and torch.nn.parallel.DistributedDataParallel
        for i in range(self.num_tasks):
            train_features = convert_examples_to_features(train_examples[i], label_list[i],
                                                          max_seq_length, self.tokenizer)
            all_input_ids = torch.tensor([feature.input_ids for feature in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([feature.segment_ids for feature in train_features], dtype=torch.long)
            all_input_masks = torch.tensor([feature.input_mask for feature in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([feature.label_id for feature in train_features], dtype=torch.long)
            task_train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_masks, all_label_ids)

            train_sampler = RandomSampler(task_train_data)
            self.train_loaders.append(iter(DataLoader(task_train_data, sampler=train_sampler, batch_size=train_bs)))
        # TODO: At the moment I'm just performing evaluation on the dev set
        # Should this be changed to also incorporate the test set?
        for i in range(self.num_tasks):
            dev_features = convert_examples_to_features(dev_examples[i], label_list[i],
                                                        max_seq_length, self.tokenizer)
            all_input_ids = torch.tensor([feature.input_ids for feature in dev_features], dtype=torch.long)
            all_segment_ids = torch.tensor([feature.segment_ids for feature in dev_features], dtype=torch.long)
            all_input_masks = torch.tensor([feature.input_mask for feature in dev_features], dtype=torch.long)
            all_label_ids = torch.tensor([feature.label_id for feature in dev_features], dtype=torch.long)
            task_dev_data = TensorDataset(all_input_ids, all_segment_ids, all_input_masks, all_label_ids)

            dev_sampler = RandomSampler(task_dev_data)
            self.dev_loaders.append(DataLoader(task_dev_data, sampler=dev_sampler, batch_size=dev_bs))

        self.data_loaded = True

    def run(self):
        """
        Runs the Multitask learning model with the following settings
        as defined in the run config:
        sampling_mode - The way in which we choose the next task in each step over
                        the epoch, sampling from the task_weightings distribution
        num_epochs - Number of epochs to train for
        steps_per_epoch - Number of optimizer steps to take per epoch
        """
        def get_task_ids(task_weightings, num_epochs, steps_per_epoch, sampling_mode='sequential'):
            """
            A helper function for getting task_ids sampled from the distribution defined via sampling_mode

            Parameters
            ----------
            task_weightings : list of floats
                The weightings of the tasks that we transform into a distribution via sampling mode from which
                we sample the task_ids
            num_epochs : int
                The number of training epochs
            steps_per_epoch : int
                The number of steps per training epoch
            sampling_mode : str, optional
                How to sample the task_ids, by default 'sequential'

            Returns
            ----------
            task_ids : list of lists/arrays
                A list indexed as task_ids[e][s] that tells you the task id for step s in epoch e
            """
            alphas = {'random': 0, 'prop': 1, 'sqrt': 0.5, 'square': 2}

            if sampling_mode == 'sequential':
                task_ids = [[step % self.num_tasks for step in range(steps_per_epoch)] for epoch in range(num_epochs)]

            if sampling_mode not in ['sequential', 'anneal']:
                alpha = alphas[sampling_mode]
                probs = [weight**alpha for weight in task_weightings]
                probs = [prob/sum(probs) for prob in probs]
                task_ids = np.random.choice(self.num_tasks, size=[num_epochs, steps_per_epoch], p=probs)
            elif sampling_mode == 'anneal':
                anneal_constant = self.run_config["anneal_constant"]
                # Generate the list by looping over the epochs, since the alpha depends on the epoch we are on
                task_ids = []
                for epoch in range(num_epochs):
                    alpha = (1 - anneal_constant*(epoch/num_epochs))
                    probs = [weight**alpha for weight in task_weightings]
                    probs = [prob/sum(probs) for prob in probs]
                    task_ids.append(np.random.choice(self.num_tasks,size=steps_per_epoch, p=probs))
            return task_ids

        # Load the data from .load_data() if not already loaded
        if not self.data_loaded:
            self.load_data()

        # Cycle train_loaders iterations ready for training
        self.train_loaders = [cycle(it) for it in self.train_loaders]

        # Prepare the settings for the run
        sampling_mode = self.sampling_mode

        if self.task_weightings_mode == "num_examples_per_task":
            task_weightings = self.train_examples_per_task
        else:
            task_types = [self.task_configs[task_name]["task_type"] for task_name in self.task_names]
            # TODO: Add in flexibility/functionality for the below map?
            task_type_map = {'Primary': 4, 'Secondary': 2, 'Tertiary': 1}
            task_weightings = [task_type_map[task_type] for task_type in task_types]

        num_epochs = self.run_config["num_epochs"]
        # TODO: Check most principled way for steps_per_epoch. Max, sum? Might need readjusting factor...
        # Readjusting factor might be proportional to the number of tasks i.e self.num_tasks
        # steps_per_epoch = int((max(self.train_examples_per_task) / self.run_config["train_batch_size"]))
        steps_per_epoch = int((np.mean(self.train_examples_per_task) * self.num_tasks / self.run_config["train_batch_size"]))
        num_training_steps = steps_per_epoch * num_epochs

        optimizer = self.model.prepare_optimizer(num_training_steps, learning_rate=self.run_config["learning_rate"],
                                                 warmup_proportion=self.run_config["warmup_prop"])

        # Initialise the global step and generate task_ids according to our sampling mode, then start the training loop
        global_step = 0
        task_ids = get_task_ids(task_weightings, num_epochs, steps_per_epoch, self.sampling_mode)
        # task_names = np.vectorize(lambda task_id: self.task_names[task_id])(task_ids)  # To map onto names instead of ids
        # task_order = {'Epoch ' + str(epoch): task_names[epoch] for epoch in range(num_epochs)}

        for epoch in trange(num_epochs, desc="Epoch"):
            # Make the model trainable again after evaluating every epoch
            self.model.train()
            # Initialise/reset train_losses, train_steps and num_examples per task
            train_losses = {task_name: 0 for task_name in self.task_names}
            n_train_steps, n_train_examples = 0, 0
            for step in trange(steps_per_epoch, desc="Step"):
                # Get the task_id from our generated list, as well as the task name, load the appropriate
                # batch from the right task, run it through the model and backprop the loss
                task_id = task_ids[epoch][step]
                task_name = self.task_names[task_id]
                batch = next(self.train_loaders[task_id])
                batch = tuple(model_input.to(self.device) for model_input in batch)
                input_ids, segment_ids, input_masks, label_ids = batch
                loss, _ = self.model(input_ids, segment_ids, input_masks, task_name, label_ids)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Increment all relevant values (mean will aggregate if computation over multiple GPUs)
                train_losses[task_name] += loss.mean().item()
                n_train_examples += input_ids.size(0)
                n_train_steps += 1
                global_step += 1
                if step % self.run_config["steps_to_log"] == 0:
                    LOGGER.info(f"Task: {task_name} - Step: {step} - Loss: {train_losses[task_name]/n_train_steps}")

            # After the epoch, normalise the accumulated training losses and set eval accs/losses to 0 for updating
            train_losses = {task_name: train_loss / n_train_steps for task_name, train_loss in train_losses.items()}
            eval_accs = {task_name: 0 for task_name in self.task_names}
            eval_losses = {task_name: 0 for task_name in self.task_names}

            # Run the evaluation method for each of the tasks
            for task_id, task_name in enumerate(self.task_names):
                eval_accs[task_name], eval_losses[task_name] = self.evaluate_model(task_id, global_step)

            # Log the results
            result_dict = {'global_step': global_step, 'train_loss': train_losses,
                           'eval_losses': eval_losses, 'eval_accuracies': eval_accs}
            LOGGER.info(f"End of epoch {epoch+1} - Results: {result_dict}")

            # TODO: save evaluation with settings to some file? e.g writer.add_text, or leave in comment string?
            # Add losses and accuracies to tensorboard
            for task_name in self.task_names:
                self.writer.add_scalars('loss', {task_name + '_train_loss': train_losses[task_name]}, global_step)
                self.writer.add_scalars('loss', {task_name + '_eval_loss': eval_losses[task_name]}, global_step)
                # Load each of the (2,3,4)-class accuracies separately in tensorboard
                if isinstance(eval_accs[task_name], float):
                    self.writer.add_scalars('acc', {task_name + '_eval_acc': eval_accs[task_name]}, global_step)
                elif isinstance(eval_accs[task_name], dict):
                    self.writer.add_scalars('acc', {'_'.join([task_name, acc_name, '_eval_acc']): eval_acc
                                            for acc_name, eval_acc in eval_accs[task_name].items()}, global_step)

    def evaluate_model(self, task_id, global_step):
        """
        Evaluation logic for the Multitask learning model

        Parameters
        ----------
        task_id : int
            The id (starting from 0) of the corresponding task
        task_name : str
            Name of the task
        global_step : int
            Global step i.e how many steps the optimiser has done
            so far overall (across epochs)
        train_loss : float
            The training loss so far for the current task (to write
            to the self.writer for tensorboard)

        Returns
        -------
        float
            Evaluation accuracy over the dev set
        """
        def accuracy(logits, labels):
            """
            Calculates the 'unnormalised' accuracy in numpy
            between the argmax of the logits and the labels
            Note: we do not divide by number of examples!

            Parameters
            ----------
            logits : tensor
                Pytorch tensor of logits
            labels : tensor
                Pytorch tensor of label ids

            Returns
            -------
            float
                number of correct predictions (not exactly accuracy
                as it isnt divided by number of examples). It is normalised
                outside this loop as we are doing batch evaluation
            """
            preds = np.argmax(logits, axis=1)
            return np.sum(preds == labels)

        def SemEval_PRF(y_true, y_preds):
            """
            Calculate the precision, recall and f1 score for the SemEval dataset

            Parameters
            ----------
            y_true : (numpy) array
                Array of true y values
            y_preds : (numpy) array
                Array of predicted y values

            Returns
            -------
            tuple
                precision, recall, f1_score
            """
            pred_sent_count = 0
            actual_sent_count = 0
            intersection_sent_count = 0

            # For each unique text, loop over each possible sentiment and
            # append predicted, actual and intersected sents
            for text_id in range(len(y_preds)//5):
                pred_sents = set()
                actual_sents = set()
                for sent_id in range(5):
                    # ID 4 corresponds to "None" in this setup i.e no label
                    if y_preds[5*text_id + sent_id] != 4:  # i.e if not "None"
                        pred_sents.add(sent_id)
                    if y_true[5*text_id + sent_id] != 4:  # i.e if not "None"
                        actual_sents.add(sent_id)
                if len(actual_sents) == 0:
                    continue
                intersected_sents = actual_sents.intersection(pred_sents)
                pred_sent_count += len(pred_sents)
                actual_sent_count += len(actual_sents)
                intersection_sent_count += len(intersected_sents)

            # Calculate the precision, recall and f1_score
            precision = intersection_sent_count / pred_sent_count
            recall = intersection_sent_count / actual_sent_count
            f1_score = (2 * precision * recall) / (precision + recall)
            return precision, recall, f1_score

        def SemEval_acc(y_true, y_preds, scores, n_classes=4):
            """
            Calculate the n_class accuracy for the SemEval Dataset

            Parameters
            ----------
            y_true : (numpy) array
                Array of true y values
            y_preds : (numpy) array
                Array of predicted y values as outputted by our model
            scores : (numpy) array
                a 2D array where each row is a vector of scores/logits for each possible label
            n_classes : int, optional
                Number of classes to calculate the accuracy to, by default 4

            Returns
            -------
            float
                The n_class accuracy

            Raises
            ------
            ValueError
                if n_classes is not in [2,3,4]
            """
            if n_classes not in [2, 3, 4]:
                raise ValueError("The number of classes for SemEval accuracy must be in [2, 3, 4]!")

            # For each true label < n_classes, calculate the best prediction out of those classes
            # and return the accuracy
            total_correct, total = 0, 0
            for i in range(len(y_true)):
                if y_true[i] >= n_classes:
                    continue
                total += 1
                pred = y_preds[i]
                if pred >= n_classes:
                    # Get the prediction for the number of classes we are calculating the accuracy to
                    pred = np.argmax(scores[i][:n_classes])
                if pred == y_true[i]:
                    total_correct += 1
            acc = total_correct / total
            return acc

        # Set the model to eval mode and initialise counters and task_name
        self.model.eval()
        task_name = self.task_names[task_id]

        # Evaluation for SemEval (TABSA) is more complicated, so will require additional evaluation logic
        if task_name.split("_")[0] == 'SemEval':
            eval_loss, n_eval_steps = 0, 0
            # Sorted label names returns labels as ["Conflict", "Negative", "Neutral", "None", "Positive"]
            label_order = {0: "Conflict", 1: "Negative", 2: "Neutral", 3: "None", 4: "Positive"}
            # Remap them so not alphabetical. This will be easier to remove the final 3 classes sequentially
            # when calculating 2 class, 3 class and 4 class accuracies in SemEval Acc function
            label_remap = {"Negative": 0, "Positive": 1, "Neutral": 2, "Conflict": 3, "None": 4}

            label_map = {key: label_remap[value] for key, value in label_order.items()}
            # Get the evaluation labels
            y_preds = None
            y_true = None
            scores = None
            for batch in self.dev_loaders[task_id]:
                input_ids, segment_ids, input_masks, label_ids = tuple(model_input.to(self.device)
                                                                       for model_input in batch)
                with torch.no_grad():
                    loss, logits = self.model(input_ids, segment_ids, input_masks, task_name, label_ids)

                eval_loss += loss.mean().item()
                logits = logits.detach().cpu().numpy()
                if y_preds is None and y_true is None and scores is None:
                    scores = logits
                    y_preds = np.argmax(logits, axis=1)
                    y_true = label_ids.to('cpu').numpy()
                else:
                    scores = np.concatenate([scores, logits])
                    y_preds = np.concatenate([y_preds, np.argmax(logits, axis=1)])
                    y_true = np.concatenate([y_true, label_ids.to('cpu').numpy()])
                n_eval_steps += 1

            # Map labels from alphabetical to new labels defined by label_map above
            y_preds = np.vectorize(label_map.get)(y_preds)
            y_true = np.vectorize(label_map.get)(y_true)

            precision, recall, f1_score = SemEval_PRF(y_true, y_preds)
            acc_4_class = SemEval_acc(y_true, y_preds, scores, n_classes=4)
            acc_3_class = SemEval_acc(y_true, y_preds, scores, n_classes=3)
            acc_2_class = SemEval_acc(y_true, y_preds, scores, n_classes=2)
            eval_accuracy = {'2_class': acc_2_class, '3_class': acc_3_class, '4_class': acc_4_class}

            LOGGER.info(f"Precision - {precision} | Recall - {recall} | "
                        f"F1 Score - {f1_score} | Accuracies - {eval_accuracy}")

            # Normalise the values we have incremented via batching
            eval_loss = eval_loss / n_eval_steps

        else:
            eval_loss, eval_accuracy = 0, 0
            n_eval_steps, n_eval_examples = 0, 0

            # Evaluate the model with batchwise accuracy like in training
            for batch in self.dev_loaders[task_id]:
                input_ids, segment_ids, input_masks, label_ids = tuple(model_input.to(self.device) for model_input in batch)
                with torch.no_grad():
                    loss, logits = self.model(input_ids, segment_ids, input_masks, task_name, label_ids)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                batch_eval_accuracy = accuracy(logits, label_ids)

                # The mean here will calculate the loss averaged over the GPUs.
                eval_loss += loss.mean().item()
                eval_accuracy += batch_eval_accuracy
                n_eval_examples += input_ids.size(0)
                n_eval_steps += 1

            # Normalise the values we have incremented via batching
            eval_loss = eval_loss / n_eval_steps
            eval_accuracy = eval_accuracy / n_eval_examples
        return eval_accuracy, eval_loss


if __name__ == '__main__':
    LOGGER.info(f"You entered run config: {RUN_CONFIG}")
    MTLModel = MultiTaskLearning(run_config=RUN_CONFIG)
    MTLModel.run()
