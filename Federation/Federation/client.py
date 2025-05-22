from dragon_baseline import DragonBaseline
import flwr as fl
from flwr.common import Context
from pathlib import Path
import torch
from typing import Dict, List, Tuple
from collections import OrderedDict
import numpy as np
import os
import json
from .config import TaskArguments, ClientArguments
from dragon_eval import DragonEval
from typing import Union
from transformers.modeling_outputs import SequenceClassifierOutput
from dragon_baseline.nlp_algorithm import ProblemType
from tqdm import tqdm
from scipy.special import expit, softmax
import pandas as pd

def append_to_json_file(file_path, new_data):
    # If file exists, load its content; otherwise, start with an empty list
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("JSON file content must be a list to append items.")
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append the new dictionary
    data.append(new_data)

    # Write the updated list back to the file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


class FederatedDragonClient(fl.client.NumPyClient):
    def __init__(self, dragon_baseline: DragonBaseline, client_id: str, client_arguments: ClientArguments, task_arguments: TaskArguments):
        self.dragon_baseline = dragon_baseline
        self.trainer = self.dragon_baseline.trainer
        self.model = self.dragon_baseline.trainer.model
        self.client_id = client_id
        self.client_arguments = client_arguments
        self.task_arguments = task_arguments

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        # Get model parameters as a list of numpy arrays
        state_dict = self.model.state_dict()
        return [val.cpu().numpy() for _, val in state_dict.items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of numpy arrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
    def fit(self, parameters, config):
        # phase = config.get("phase", "train")
        # if phase == "collect_scaler":
        #     # Collect the scaler from the client
        #     self._global_scaler = self.dragon_baseline.trainer.scaler
        #     return self.get_parameters(config={}), len(self.dragon_baseline.trainer.train_dataset), {}

        self.set_parameters(parameters)
        self.dragon_baseline.train()  # Train the model
        return self.get_parameters(config={}), len(self.dragon_baseline.trainer.train_dataset), {}
        
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = self.trainer.evaluate(eval_dataset=self.dragon_baseline.data_args.eval_dataset)

        # eval_predictions = self.dragon_baseline.predict(df=self.dragon_baseline.df_val)
        # if self.task_arguments.hot_eval:
        #     hot_eval = HotEval(model=self, eval_predictions=eval_predictions)
        #     dragon_eval_results = float(hot_eval.evaluate())
        #     import ipdb; ipdb.set_trace()
        #     metrics["eval_dragon"] = dragon_eval_results

            # import tempfile
            # with tempfile.TemporaryDirectory() as tmpdirname:
            #     eval_res_dir = Path(tmpdirname) / Path(self.client_arguments.input_path.name)
            #     eval_res_dir.mkdir(parents=True, exist_ok=True)
            #     hot_eval.save(predictions=eval_predictions, path=eval_res_dir / Path("nlp-predictions-dataset.json"))
            #     # writ gt to same dir
            #     val_gt_path = Path(self.client_arguments.input_path) / Path(self.client_arguments.center) / Path("nlp-validation-dataset.json")
            #     with open(val_gt_path, 'r') as f:
            #         val_gt_path = json.load(f)
            #         json.dump(val_gt_path, open(eval_res_dir / Path(self.task_arguments.task + '.json'), 'w'))
            #     dragon_eval = DragonEval(
            #         ground_truth_path=eval_res_dir,
            #         predictions_path=Path(tmpdirname),
            #         folds=[int(self.client_arguments.input_path.name.split('-')[-1].replace('fold', ''))],
            #         tasks=[self.task_arguments.task.split('_')[0].replace('Task', '')],
            #         output_file=eval_res_dir / Path("dragon_eval_val_res.json"),
            #     )
            #     dragon_eval.evaluate()
            #     metrics["eval_dragon_cold"] = dragon_eval._scores[self.task_arguments.task][self.client_arguments.input_path.name]

        max_eval_samples = self.dragon_baseline.data_args.max_eval_samples if self.dragon_baseline.data_args.max_eval_samples is not None else len(self.dragon_baseline.data_args.eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(self.dragon_baseline.data_args.eval_dataset))
        metrics["center"] = self.client_arguments.center
        return metrics["eval_loss"], metrics["eval_samples"], metrics

class HotEval(DragonEval):
    '''Evaluate the model using the DragonEval framework to get the most
    appropriate eval metrics to use for model selection.'''
    def __init__(self, model : FederatedDragonClient, eval_predictions, **kwargs):
        self._scores: Dict[str, Dict[str, float]] = {}
        self._join_key = "uid"
        self._predictions_cases = eval_predictions.copy()

        with open(model.client_arguments.input_path / Path(model.client_arguments.center) / Path("nlp-validation-dataset.json"), 'r') as f:
            self._ground_truth_cases = pd.DataFrame(json.load(f))
        label_column = [col for col in self._ground_truth_cases.columns if col.endswith("_target")][0]

        # Due to pandas automtic casting of dtypes strings/categories are sometimes converted to integers, which can cause issues with DragonEval.
        # In this case we ensure that the target and prediction columns are of the same dtype by explicitely casting.
        dtype_mismatch = self._ground_truth_cases[label_column].dtype != self._predictions_cases[label_column.replace('_target', '')].dtype
        gt_is_integer = np.issubdtype(self._ground_truth_cases[label_column].dtype, np.integer)
        if dtype_mismatch and gt_is_integer:
            self._predictions_cases[label_column.replace('_target', '')] = self._predictions_cases[label_column.replace('_target', '')].astype(self._ground_truth_cases[label_column].dtype)
        
        # self._ground_truth_cases = model.dragon_baseline.df_val
        self.model = model
        self.task = self.model.task_arguments.task

    def evaluate(self):
        self.merge_ground_truth_and_predictions()
        self.cross_validate()
        self.score(task_name=self.task, job_name='fed_eval')
        return self._scores[self.task]['fed_eval']
    
    def save(self, predictions: pd.DataFrame, path = None):
        """Save the predictions."""
        if path is None:
            path = self.test_predictions_path
        path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_json(path, orient="records")

    def verify_predictions(self, dataset_test_path = None):
        if dataset_test_path is not None:
            self.dataset_test_path = dataset_test_path
        super().verify_predictions()

class Model(DragonBaseline):
    def __init__(self, input_path: Union[Path, str] = Path("/input"), output_path: Union[Path, str] = Path("/output"), workdir: Union[Path, str] = Path("/opt/app"), model_kwargs : dict = None, **kwargs):
        """
        Adapt the DRAGON baseline to use the joeranbosma/dragon-roberta-large-domain-specific model.
        Note: when manually changing the model, update the Dockerfile to pre-download that model.
        """
        super().__init__(input_path=input_path, output_path=output_path, workdir=workdir, model_kwargs=model_kwargs, **kwargs)

        self.model_name_or_path = "joeranbosma/dragon-bert-base-mixed-domain"
        self.per_device_train_batch_size = 4
        self.gradient_accumulation_steps = 2
        self.gradient_checkpointing = False
        self.max_seq_length = 512

        self.learning_rate = 1e-05

        self.warmup_ratio = 0
        self.num_train_epochs = 1 # Train epochs per round
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.workdir = Path(workdir)

    def get_hyperparameters(self) -> Dict:
        """Helper function to get the hyperparameters for the model so that we
        can log them to wandb if enabled."""
        hyperparameters = {}
        # Get default hyperparameters from DragonBaseline
        dragon_baseline_keys = [
            "model_name","per_device_train_batch_size",
            "gradient_accumulation_steps","gradient_checkpointing",
            "max_seq_length","learning_rate","num_train_epochs",
            "warmup_ratio", "fp16", "create_strided_training_examples",
        ]
        for key in dragon_baseline_keys:
            if hasattr(self, key):
                hyperparameters[key] = getattr(self, key)
        # Add any additional hyperparameters or overwritten defaults from kwargs
        hyperparameters.update(self.model_kwargs)
        return hyperparameters
    

    def predict_huggingface(self, *, df: pd.DataFrame) -> pd.DataFrame:
        """Predict the labels for the test data. We have removed the error handling 
        from the parent class since it draws conclusions based on the df_train dataset which
        will vary per client."""
        # load the model and tokenizer
        tokenizer = self._get_tokenizer(self.model_save_dir, check_directory_for_vocab_files=True)
        model = self.trainer.model

        # predict
        results = []
        for _, row in tqdm(df.iterrows(), desc="Predicting", total=len(df)):
            # tokenize inputs
            inputs = row[self.task.input_name] if self.task.input_name == "text_parts" else [row[self.task.input_name]]
            tokenized_inputs = tokenizer(*inputs, return_tensors="pt", truncation=True).to(self.device)

            # predict
            result: SequenceClassifierOutput = model(**tokenized_inputs)

            if self.task.target.problem_type == ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION:
                logits: List[np.ndarray] = [logits.detach().cpu().numpy() for logits in result.logits]
            else:
                logits: np.ndarray = result.logits.detach().cpu().numpy()

            # convert to labels
            if self.task.target.problem_type == ProblemType.SINGLE_LABEL_REGRESSION:
                prediction = {self.task.target.prediction_name: logits[0][0]}

            elif self.task.target.problem_type == ProblemType.MULTI_LABEL_REGRESSION:
                prediction = {self.task.target.prediction_name: logits[0]}

            elif self.task.target.problem_type == ProblemType.SINGLE_LABEL_BINARY_CLASSIFICATION:
                # calculate sigmoid to map the logits to [0, 1]
                prediction = softmax(logits, axis=-1)[0, 1]
                prediction = {self.task.target.prediction_name: prediction}

            elif self.task.target.problem_type == ProblemType.SINGLE_LABEL_MULTI_CLASS_CLASSIFICATION:
                p = model.config.id2label[np.argmax(logits[0])]
                prediction = {self.task.target.prediction_name: p}

            elif self.task.target.problem_type == ProblemType.MULTI_LABEL_BINARY_CLASSIFICATION:
                expected_shape = (1, len(self.task.target.values))
                prediction = expit(logits)[0]  # calculate sigmoid to map the logits to [0, 1]
                prediction = {self.task.target.prediction_name: prediction}

            elif self.task.target.problem_type == ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION:
                preds = [np.argmax(p) for p in logits]
                prediction = {
                    self.task.target.prediction_name: [
                        id2label[str(p)]
                        for p, id2label in zip(preds, model.config.id2labels)
                    ]
                }
            else:
                raise ValueError(f"Unexpected problem type '{self.task.target.problem_type}'")

            results.append({"uid": row["uid"], **prediction})

        df_pred = pd.DataFrame(results)

        # scale the predictions (inverse of the normalization during preprocessing)
        df_pred = self.unscale_predictions(df_pred)

        return df_pred


# def set_up_client_function(task, experiment, input_base_path, output_base_path, base_workdir):
def set_up_client_function(task_arguments: TaskArguments, client_arguments: ClientArguments, model_kwargs: Dict = None, scaler_params: Dict = None) -> fl.client.ClientFn:
    def client_fn(context : Context) -> fl.client.Client:
        print(f"context.run_config: {context.run_config}")
        print(f"context.node_config: {context.node_config}")
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]

        centers = os.listdir(client_arguments.input_path)
        client_arguments.center = centers[partition_id] 

        print(f"center: {centers[partition_id]}") 
        
        task_number = str(int(task_arguments.task.split("_")[0].replace("Task", "")))
        modelClass = Model(
            input_path = Path(client_arguments.input_path) / Path(client_arguments.center),
            output_path = Path(client_arguments.output_path) / Path(client_arguments.center),
            workdir = Path(client_arguments.workdir) / Path(client_arguments.center),
            model_kwargs=model_kwargs if model_kwargs is not None else {},
            )
        modelClass.scaler_params = scaler_params
        modelClass.setup()

        return FederatedDragonClient(dragon_baseline=modelClass, client_id="client", client_arguments=client_arguments, task_arguments=task_arguments)
    return client_fn

