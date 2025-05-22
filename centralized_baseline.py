from Federation.Federation.client import Model
import os
from dotenv import load_dotenv
from pathlib import Path
from dragon_eval import DragonEval

load_dotenv()
DATADIR = os.getenv("DATA_DIR")

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

    # dataloader
    # "dataloader_num_workers": 4,
    # "dataloader_persistent_workers": True,
    # "dataloader_prefetch_factor": 2,
    "dataloader_drop_last": True,
    "group_by_length": True,

    # precision/math
    "optim": "adamw_torch_fused",
    # "torch_compile": True,
    # "torch_compile_mode": "reduce-overhead",
    # "bf16" : True,

    # logging/checkpointing
    "logging_steps": 1000,
    # "disable_tqdm": True,
    "save_strategy": "best",
    "save_only_model": True,
    "load_best_model_at_end": True,
    "metric_for_best_model": "dragon",
    "greater_is_better": True,
    "use_mps_device" : True
#     "max_train_samples" : 300,
#     # "max_eval_samples": 50,
}

# model_kwargs = {
#     # "model_name_or_path": "distilbert/distilbert-base-multilingual-cased",
#     "model_name_or_path": "joeranbosma/dragon-bert-base-mixed-domain",
#     # "model_name_or_path": "DTAI-KULeuven/robbert-2023-dutch-base",
#     "per_device_train_batch_size": 4,
#     "gradient_accumulation_steps": 2,
#     "gradient_checkpointing": False,
#     "max_seq_length": 512,
#     "learning_rate": 5e-05,
#     "lr_scheduler_type": "constant",
#     "save_total_limit": 1,
#     "warmup_ratio": 0,
#     "num_train_epochs": 15,
#     "metric_for_best_model": "dragon",
#     "greater_is_better": True,
#     "save_strategy": "best",
#     "load_best_model_at_end": True,
# #     "max_train_samples" : 300,
# #     # "max_eval_samples": 50,
# }

# model_kwargs = {
#     "model_name_or_path": "joeranbosma/dragon-roberta-large-domain-specific",
#     # "model_name_or_path": "joeranbosma/dragon-bert-base-mixed-domain",
#     "per_device_train_batch_size": 1,
#     "gradient_accumulation_steps": 2,
#     "gradient_checkpointing": True,
#     "max_seq_length": 512,
#     "learning_rate": 1e-05,
#     "lr_scheduler_type": "constant",
#     "save_total_limit": 2,
#     # "warmup_ratio": 0.1,
#     "num_train_epochs": 3,
#     "metric_for_best_model": "dragon",
#     "greater_is_better": True,
#     "save_strategy": "best",
#     "load_best_model_at_end": True,
#     "max_train_samples" : 300,
#     # "max_eval_samples": 20,
#     "fp16_backend": "auto",
# }

if __name__ == "__main__":
    experiment = 'speed_test_1'
    input_base_path = Path(os.getenv("INPUT_COLLECTED_DIR"))
    task_list = sorted([p.name for p in input_base_path.glob("Task*fold1")])
    # task_list = ["Task002_nodule_clf-fold1"]
    task_list = [
        # "Task010_prostate_radiology_clf-fold1",
        # "Task002_nodule_clf-fold1",
        # "Task016_recist_lesion_size_presence_clf-fold1",
        # "Task019_prostate_volume_reg-fold1",
        # "Task024_recist_lesion_size_reg-fold1",
        # "Task025_anonymisation_ner-fold0",
        "Task013_pathology_tissue_origin_clf-fold0"
        ]
    print(task_list)
    for task in task_list:

        task = os.path.basename(task)
        input_path =  input_base_path / Path(task)
        output_path = Path(os.getenv("OUTPUT_DIR")) / Path(f"{experiment}/{task}")
        workdir = Path(os.getenv("WORK_DIR")) / Path(f"{experiment}/{task}")

        output_path.mkdir(parents=True, exist_ok=True)
        workdir.mkdir(parents=True, exist_ok=True)

        mdl = Model(input_path = input_path, output_path = output_path, workdir = workdir, model_kwargs=model_kwargs)
        mdl.scaler_params = {"mean_": 1, "scale_": 2, "var_": 3}
        mdl.setup()
        mdl.train()
        predictions = mdl.predict(df=mdl.df_test)
        mdl.save(predictions)
        mdl.verify_predictions()

# dragonEval = DragonEval(
#     ground_truth_path= Path(os.getenv("TEST_DIR")),
#     predictions_path= output_path.parent,
#     output_file= output_path.parent / Path("test_metrics.json"),
#     folds=[0],
#     tasks=['025']
# ).evaluate()
