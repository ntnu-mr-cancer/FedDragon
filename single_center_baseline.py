from Federation.Federation.client import Model
import os
from dotenv import load_dotenv
from pathlib import Path
from dragon_eval import DragonEval
import shutil
import json
from transformers import EarlyStoppingCallback

load_dotenv()
DATADIR = os.getenv("DATA_DIR")

with open("taskLabels.json", "r") as f:
    task_labels = json.load(f)

model_kwargs = {
    "model_name_or_path": "joeranbosma/dragon-bert-base-mixed-domain",
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "gradient_checkpointing": False,
    "max_seq_length": 512,
    "learning_rate": 5e-05,
    "lr_scheduler_type": "constant",
    "save_total_limit": 1,
    "warmup_ratio": 0,
    "num_train_epochs": 15,


    # precision/math
    "optim": "adamw_torch_fused",
    # "bf16" : True,

    # logging/checkpointing
    "logging_steps": 1000,
    "save_strategy": "best",
    "save_only_model": True,
    "load_best_model_at_end": True,
    "metric_for_best_model": "dragon",
    "greater_is_better": True,
    "use_mps_device" : True
}

if __name__ == "__main__":
    experiment = 'single_center_baseline'
    input_base_path = Path(os.getenv("INPUT_DIR"))
    task_list = sorted([p.name for p in input_base_path.glob("Task025*")])
    # task_list = ["Task002_nodule_clf-fold1"]
    print(task_list)
    for task in task_list:
        task = os.path.basename(task)
        for center in os.listdir(input_base_path / Path(task)):
            input_path =  input_base_path / Path(task) / Path(center)
            output_path = Path(os.getenv("OUTPUT_DIR")) / Path(f"{experiment}/{task}/{center}")
            workdir = Path(os.getenv("WORK_DIR")) / Path(f"{experiment}/{task}/{center}")

            output_path.mkdir(parents=True, exist_ok=True)
            workdir.mkdir(parents=True, exist_ok=True)

            mdl = Model(input_path = input_path, output_path = output_path, workdir = workdir, model_kwargs=model_kwargs)
            # manually setting the test dataset path to make sure we use the internal test set / validation set
            mdl.dataset_test_path = Path(os.getenv("INPUT_COLLECTED_DIR")) / Path(task) / 'nlp-test-dataset.json'
            mdl.model_kwargs.update({'label2id' : task_labels.get(str(int(task.split('_')[0].replace('Task', ''))), None)})
            mdl.setup()

            mdl.trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01))
            mdl.train()
            predictions = mdl.predict(df=mdl.df_test)
            mdl.save(predictions)
            mdl.verify_predictions()

experiment_output_base_path = Path(os.getenv("OUTPUT_DIR")) / Path(f"{experiment}")
reordered_output_folder =  experiment_output_base_path / "reordered"
reordered_output_folder.mkdir(parents=True, exist_ok=True)

for task in [e for e in os.listdir(experiment_output_base_path) if "Task" in e]:
    for center in os.listdir(experiment_output_base_path / Path(task)):
        new_out_folder = (reordered_output_folder / Path(center) / Path(task))
        new_out_folder.mkdir(parents=True, exist_ok=True)
        # copy predictions into the new folder structure
        shutil.copy(experiment_output_base_path / Path(task) / Path(center) / Path("nlp-predictions-dataset.json"), new_out_folder / Path("nlp-predictions-dataset.json")) 

# we do individual evaluation for each center and task
for center in os.listdir(reordered_output_folder):
    for task in os.listdir(reordered_output_folder / Path(center)):
        output_path = reordered_output_folder / Path(center) / Path(task)

        dragonEval = DragonEval(
            ground_truth_path= Path(os.getenv("TEST_DIR")),
            predictions_path= output_path.parent,
            output_file= output_path / Path("test_metrics.json"),
            folds=[task.split('-')[-1].replace('fold', '')],
            tasks=[task.split('_')[0].replace('Task', '')]
        ).evaluate()
