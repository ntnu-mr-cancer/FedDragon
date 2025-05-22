#%%
import sys
sys.path.append("..")
from client import FederatedDragonClient
from dragon_baseline import DragonBaseline
from client import Model
from dragon_eval import DragonEval
from pathlib import Path
import os
from dotenv import load_dotenv
import numpy as np
import json
import logging
from transformers import HfArgumentParser
from config import TaskArguments, ClientArguments
import argparse
load_dotenv()

# Command-line interface (disable abbreviation to catch typos like --task)
parser = argparse.ArgumentParser(description="Predict from federated model")
parser.add_argument('experiment', type=str, help='Name of the experiment')
parser.add_argument('--folds', type=int, nargs='+', default=[0,1,2,3,4], help='List of folds to evaluate. Example usage: --folds 0 1 2 will evaluate folds 0, 1, and 2')
parser.add_argument('--tasks', type=int, nargs='+', help='List of task numbers to evaluate (defaults to all tasks). Example usage: --tasks 2 5 10 will evaluate tasks Task002, Task005, and Task010)')
parser.add_argument('--do_not_predict', action='store_true', help='If set, will not perform predictions but only evaluate existing predictions')
args = parser.parse_args()

experiment = args.experiment
folds = args.folds

with open("taskLabels.json", "r") as f:
    task_labels = json.load(f)

input_base_path = Path(os.getenv("INPUT_COLLECTED_DIR"))
input_split_base_path = Path(os.getenv("INPUT_DIR"))

tasks = [os.path.basename(task) for task in os.listdir(input_split_base_path) if os.path.isdir(input_split_base_path / task)]
# Filter tasks if provided via CLI
if args.tasks:
    # args.tasks contains task numbers as strings
    tasks = [t for t in tasks if int(t.split("_")[0].replace("Task", "")) in args.tasks]
if args.folds:
    # Filter tasks based on folds
    tasks = [t for t in tasks if any(f"fold{fold}" in t for fold in args.folds)]

tasks_nums = set([task.split("_")[0].replace("Task", "") for task in tasks])

output_base_path = Path(os.getenv("OUTPUT_DIR")) / Path(experiment)
base_workdir = Path(os.getenv("WORK_DIR")) / Path(experiment)

exception_tasks = []

def get_weights_path(server_dir : Path, file_identifier : str = "best") -> Path:
    """
    Get the path to the weights file for a given task.
    """
    files = os.listdir(server_dir)
    f = [file for file in files if file.endswith(".npz") and file_identifier in file]
    if len(f) > 1:
        raise RuntimeError(f"Multiple weight files found in {server_dir} matching identifier '{file_identifier}': {f}")
    elif len(f) == 0:
        raise RuntimeError(f"No weight files found in {server_dir} matching identifier '{file_identifier}'")
    return server_dir / f[0]

#%%
if not args.do_not_predict:
    for task in tasks:
        centers = os.listdir(f"{input_split_base_path}/{task}")
        try:
            task_name = task.split('-')[0]
            config ={
                "task" : task_name,
                "input_path" : input_base_path / Path(task.replace(task.split('-')[1], 'fold0')),
                "output_path" : output_base_path / Path(task),
                "workdir" : base_workdir / Path(task) / Path(centers[0]),
                "experiment" : experiment,
            }

            config_parser = HfArgumentParser((TaskArguments, ClientArguments))
            task_arguments, client_arguments = config_parser.parse_dict(config)

            task_number = str(int(task.split("_")[0].replace("Task", "")))
            model_kwargs = {
                "model_name_or_path": "joeranbosma/dragon-bert-base-mixed-domain",
            }
            modelClass = Model(input_path = config["input_path"], output_path = config["output_path"], workdir = config["workdir"], model_kwargs=model_kwargs)
            modelClass.model_kwargs.update({'label2id' : task_labels.get(task_number, None)})
            modelClass.setup()

            client = FederatedDragonClient(dragon_baseline=modelClass, client_id="client", client_arguments=client_arguments, task_arguments=task_arguments)
    
            parameters = np.load(get_weights_path(config["workdir"].parent / "server", file_identifier="best"))
            parameters = [parameters[file] for file in parameters._files]
            client.set_parameters(parameters)

            predictions = client.dragon_baseline.predict(df=client.dragon_baseline.df_test)

            client.dragon_baseline.save(predictions)
            client.dragon_baseline.verify_predictions()


        except Exception as e:
            print(f"Error processing task {task}: {e}")
            logging.error(f"Error processing task {task}: {e}")
            exception_tasks.append(task)
            continue
    print(f"Tasks with exceptions: {exception_tasks}")
#%% 
dragonEval = DragonEval(
    ground_truth_path= Path(os.getenv("TEST_DIR")),
    predictions_path=output_base_path,
    output_file= output_base_path / Path("test_metrics.json"),
    folds=folds,
    tasks=tasks_nums
).evaluate()