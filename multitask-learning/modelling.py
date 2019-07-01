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
This code will store all the modelling required to build up various BERT models
"""
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

# from pytorch_pretrained_bert import BertAdam, BertModel TODO: Remove comment when open sourcing
from bert_sb import BertAdam, BertModel


# Setup pretrained models to download - typically use bert-base
MODEL_NAMES = ['bert-base-uncased', 'bert-large-uncased',
               'bert-base-cased', 'bert-large-cased',
               'bert-base-multilingual-uncased',
               'bert-base-multilingual-cased', 'bert-base-chinese']


class MultiTaskModel(nn.Module):
    """
    Custom class for Multitask objective using BERT as the base model.
    It automatically generates a set of heads, with a simple linear layer onto
    each task to give a classification according to the task config i.e
    task_configs[task_name]['num_labels']
    task_configs must be a dict of the following form:
    {task_name: {'num_labels': <int>, 'task_type': <str>, 'output_type', <str>} for task_name in tasks} where:
    num_labels : number of labels (int)
    task_type : indicates if a task is a primary, secondary or tertiary task. Not used in this class.
    output_type : indicates if task if regression or classification ('REG'/'CLS')
    """
    def __init__(self, task_configs, model_name_or_config='bert-base-uncased'):
        super(MultiTaskModel, self).__init__()

        self.task_configs = task_configs

        # Load BERT model from either a specified config file or pretrained
        # Apply weights initialisation (automatic if from_pretrained)
        if isinstance(model_name_or_config, dict):
            # TODO: For now, assume if a config is passed in then it is applied to
            # the BERT model, may change in future
            self.baseLM = BertModel(config=model_name_or_config)
            self.baseLM.apply(self.__init_weights)
        elif isinstance(model_name_or_config, str):
            # Extract model name from the string
            model_name_or_config = model_name_or_config.lower()
            self.base_model_name = model_name_or_config.split("-")[0]
            if model_name_or_config not in MODEL_NAMES:
                raise ValueError(f"Please enter a valid model name - you entered {model_name_or_config}")
            if self.base_model_name == 'bert':
                self.baseLM = BertModel.from_pretrained(model_name_or_config)

        # Store our bert config file
        self.baseLM_config = self.baseLM.config

        # Add dropout layer
        self.dropout = nn.Dropout(self.baseLM_config.hidden_dropout_prob)

        # Add the heads (just simple linear layers mapping to num_labels)
        self.heads = nn.ModuleDict({task_name:
                                    nn.Linear(self.baseLM_config.hidden_size,
                                              task_config['num_labels'])
                                    for task_name, task_config
                                    in self.task_configs.items()})

        # Apply initialisation to the heads of the model
        self.heads.apply(self.__init_weights)

    def __init_weights(self, module):
        """
        Initialise the weights of the PyTorch nn module according to
        bert_config

        Parameters
        ----------
        module : PyTorch nn.Module
            Module inside the neural net
        """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            # Perform the same initialisation protocol as that in bert_config
            module.weight.data.normal_(mean=0.0,
                                       std=self.baseLM_config.initializer_range)
        if (isinstance(module, (nn.Linear, nn.LayerNorm)) and
                module.bias is not None):
            module.bias.data.zero_()

    def prepare_optimizer(self, num_train_optimization_steps,
                          learning_rate=1e-6, warmup_proportion=0.1,
                          weight_decay=0.0):
        """
        Prepares Adam optimizer for BERT. Stolen from KnowledgeBERT and adapted

        Parameters
        ----------
        epochs : int
            number of training epochs
        num_train_optimization_steps : int
            number of training optimization steps
        learning_rate : float
            the learning rate for the Adam optimiser
        warmup_proportion : float
            proportion of the learning rate that is a ramp,
            i.e. learning_rate_fn(0.1*total_steps) -> 10% of training steps
            = max(learning_rate) = learning_rate input param
        weight_decay : float
            weight decay (L2) for all params except the biases and
            normalisation layers

        Returns
        ------
        optimizer : BertAdam
            BertAdam optimizer
        """
        # Prepare optimizer
        param_list = list(self.baseLM.named_parameters()) + list(self.heads.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [param for name, param in param_list
                        if not any(no_decay_name in name for no_decay_name in no_decay)],
             'weight_decay': weight_decay},
            {'params': [param for name, param in param_list
                        if any(no_decay_name in name for no_decay_name in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)
        return optimizer

    def unfreeze_base_layers(self, base_params_to_unfreeze='all'):
        """
        Function to automatically freeze certain layers of the base (BERT) model

        Parameters
        ----------
        bert_params_to_unfreeze : list or str, optional
            List or str of parameters to unfreeze. It will match any part of the
            parameter name automatically outputted by pytorch. To see the list
            of module names run the following code:
            ```
            model = BertForMultiTask()
            param_names = [name for name, _ in model.named_parameters()]
            ```
            Note that the parameter names are separated/indexed by full stops
            by default 'all' (i.e every param is unfrozen)
            If you wanted a specific layer then 'layer.11' could be passed, for example
        """
        for name, param in self.baseLM.named_parameters():
            if ('all' in base_params_to_unfreeze or
                    any(unfreeze_param in name for unfreeze_param in base_params_to_unfreeze)):
                param.requires_grad = True
            else:
                param.detach()
                param.requires_grad = False

    def forward(self, input_ids, segment_ids, attention_mask, task_name, labels=None):
        """
        The forward pass for the BertForMultiTask module

        Parameters
        ----------
        input_ids : tensor
            PyTorch tensor of IDs as is compatable with BERT. Lookup ID via bert vocab
        segment_ids : tensor
            PyTorch tensor of segment IDs as is compatable with BERT. 0 = text_a, 1 = text_b
        attention_mask : tensor
            PyTorch tensor of the attention mask as is compatable with BERT. Mask of 1's over input text
        task_name : str
            Name of the task to forward pass through
        labels : tensor, optional
            PyTorch tensor of labels for the relevant task, by default None

        Returns
        -------
        tensor or tuple of tensors
            If labels is not None: output is loss, logits where the appropriate
            loss function (for classification or regression) has been chosen
            automatically based on the task config.
            If labels is None: return the logits
        """
        # Get pooled output from BERT Model and apply dropout
        # TODO: final _ is attn_data_list - to remove when open sourcing
        _, pooled_output, _ = self.baseLM(input_ids, segment_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        # Forward pass through the head of the corresponding task
        logits = self.heads[task_name](pooled_output)

        if labels is not None:
            if self.task_configs[task_name]["output_type"] == "CLS":
                loss_fn = CrossEntropyLoss()
                loss = loss_fn(logits, labels)
                return loss, logits
            elif self.task_configs[task_name]["output_type"] == "REG":
                loss_fn = MSELoss()
                loss = loss_fn(logits, labels.unsqueeze(1))
                return loss, logits
        else:
            return logits
