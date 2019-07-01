# Simple script to automatically run experiments given hyperparameters we want to test
import itertools
import logging
from pathlib import Path

import pandas as pd

from multitask_learning import MultiTaskLearning

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    level=logging.INFO)
LOGGER = logging.getLogger(__name__)

RUNS_FOLDER = Path(__file__).resolve().parent / 'runs'
EXPERIMENT_LOG = RUNS_FOLDER / 'experiment_log.csv'

def hp_list_cleaner(hp_list):
    """
    A helper function for cleaning the hyperparameter list of tuples

    Parameters
    ----------
    hp_list : list of tuples
        List of hyperparameter tuples
    """
    hp_list_cleaned = []
    for hp_tuple in hp_list:
        tasks, sampling_mode = hp_tuple
        # Input logic so that we dont test all sampling modes if we only have one task (only sequential and random)
        num_tasks = len(tasks.split(", "))
        if num_tasks == 1 and sampling_mode != 'sequential':
            continue
        else:
            hp_list_cleaned.append(hp_tuple)
    return hp_list_cleaned


if __name__ == '__main__':
    # TODO: Have a think about which hyperparameters to test (cf. MultiTaskLearning)
    hparams_to_test = {'tasks': ['SST-2', 'SemEval_QA_M', 'SST-2, SemEval_QA_M', 'IMDB', 'SST-2, IMDB'],
                       'sampling_mode': ['sequential', 'random', 'prop', 'sqrt', 'square', 'anneal']
                       }
    metrics_to_report = ['final_loss_train', 'final_loss_dev', 'final_acc_dev']
    # Create an experiment log if one is not already present
    if not RUNS_FOLDER.is_dir():
        RUNS_FOLDER.mkdir()
    if not EXPERIMENT_LOG.is_file():
        pd.DataFrame(columns=list(hparams_to_test.keys()) + metrics_to_report).to_csv(EXPERIMENT_LOG)

    total_experiments_so_far = len(pd.read_csv(EXPERIMENT_LOG))
    hp_list_cleaned = hp_list_cleaner(itertools.product(*hparams_to_test.values()))
    for experiment_num, (tasks, sampling_mode) in enumerate(hp_list_cleaned):
        # Allow us to pick up where we left off by skipping over experiments we have already run
        if experiment_num < total_experiments_so_far:
            continue

        # Perform Experiments with each setting of the hyperparameters
        # If hyperparameter not specified, the default will be used (see MultiTaskLearning class)
        run_config = {'data_dir': '../../../datascience-projects/internal/multitask_learning/processed_data',
                      'model_name': 'bert-base-cased',
                      'sampling_mode': sampling_mode,
                      'tasks': tasks}

        LOGGER.info(f"Running experiment with run config: {run_config}")
        MTLModel = MultiTaskLearning(run_config=run_config)
        final_loss_train, final_loss_dev, final_acc_dev = MTLModel.train()

        # Mark the experiment as done in the log (All metrics etc stored in the tensorboard log not in csv)
        with open(EXPERIMENT_LOG, 'a') as experiment_log:
            experiment = pd.DataFrame({'sampling_mode': sampling_mode, 'tasks': tasks,
                                       'final_loss_train': final_loss_train,
                                       'final_loss_dev': final_loss_dev,
                                       'final_acc_dev': final_acc_dev}, index=[experiment_num])
            experiment.to_csv(experiment_log, header=False)
