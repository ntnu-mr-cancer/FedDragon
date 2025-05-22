from dataclasses import dataclass, field
from dragon_baseline.util import ArgumentsClass
from dragon_eval.evaluation import TASK_TYPE, EvalType
from typing import List

model_selction_metric_mapper = {
    EvalType.SINGLE_LABEL_NER : "f1",
    EvalType.MULTI_LABEL_NER : "f1",
    EvalType.REGRESSION : "mse",
    EvalType.BINARY_CLASSIFICATION : "accuracy",
    EvalType.BINARY_CLASSIFICATION_NON_SHARED_TASK : "accuracy",
    EvalType.ORDINAL_MULTI_CLASS_CLASSIFICATION : "accuracy",
    EvalType.NONORDINAL_MULTI_CLASS_CLASSIFICATION : "accuracy",
}

all_metrics_mapper = {
	EvalType.SINGLE_LABEL_NER : ["f1", "precision", "recall", "accuracy"],
	EvalType.MULTI_LABEL_NER : ["f1", "precision", "recall", "accuracy"],
	EvalType.REGRESSION : ["mse"],
	EvalType.BINARY_CLASSIFICATION : ["accuracy"],
	EvalType.BINARY_CLASSIFICATION_NON_SHARED_TASK : ["accuracy"],
	EvalType.ORDINAL_MULTI_CLASS_CLASSIFICATION : ["accuracy"],
	EvalType.NONORDINAL_MULTI_CLASS_CLASSIFICATION : ["accuracy"],
}

is_multi_label_bin_clf = ["Task015_colon_pathology_clf", "Task016_recist_lesion_size_presence_clf", "Task104_Example_ml_bin_clf"]

@dataclass
class ServerArguments(ArgumentsClass):
	"""
	Configuration for the server in a federated learning setup.
	"""
	num_rounds: int = field(default=5, metadata={"help": "Number of rounds to run the simulation"})
	server_base_dir: str = field(default="server", metadata={"help": "Base directory for server files"})

	# Parameters for client sampling. Should be set to number of clients since we don't expect any failures.
	num_centers: int = field(default=1, metadata={"help": "Number of centers (clients) in the federation"})
	center_names : List[str] = field(default_factory=list, metadata={"help": "List of centers (clients) in the federation"})


@dataclass
class ClientArguments(ArgumentsClass):
	"""
	Configuration for the client in a federated learning setup.
	"""
	input_path: str = field(metadata={"help": "Input path for client data"})
	output_path: str = field(metadata={"help": "Output path for client results"})
	workdir: str = field(metadata={"help": "Working directory for the client"})
	center : str = field(init=False, metadata={"help": "Center identifier for the client"})

@dataclass
class TaskArguments(ArgumentsClass):
	"""
	Configuration for the experiment in a federated learning setup.
	"""
	experiment: str = field(default=None, metadata={"help": "Experiment identifier"})
	task : str = field(default=None, metadata={"help": "Task identifier for the experiment"})
	model_selection_metric: str = field(default= "dragon", metadata={"help": "Metric used for model selection"})
	greater_is_better: bool = field(default=None, metadata={"help": "Whether a greater value is better for the model selection metric"})
	all_evaluation_metrics : list = field(init=False, default_factory=list, metadata={"help": "List of all evaluation metrics to aggregate during evaluation."})
	log_to_wandb: bool = field(default=False, metadata={"help": "Whether to log results to Weights & Biases"})
	hot_eval: bool = field(default=True, metadata={"help": "Whether to perform hot evaluation (evaluate using DragonEval on the validation set after each round)"})

	def __post_init__(self):
		# Automatically set appropriate model selection metric if not excplicitly set
		if self.task in is_multi_label_bin_clf:
			self.all_evaluation_metrics = ["f1"]
		else:
			if self.model_selection_metric is None:
				self.model_selection_metric = model_selction_metric_mapper.get(TASK_TYPE[self.task], "loss")
			self.all_evaluation_metrics = all_metrics_mapper[TASK_TYPE[self.task]]

		if self.greater_is_better is None:
			self.greater_is_better = self.model_selection_metric not in ["loss", "mse"]

		if self.hot_eval:
			self.all_evaluation_metrics = self.all_evaluation_metrics + ["dragon"]

		if self.log_to_wandb:
			import wandb
			wandb.summary["model_selection_metric"] = self.model_selection_metric
			for metric in self.all_evaluation_metrics:
				# wandb.define_metric(
				# 	"eval_" + metric,
				# )
				wandb.define_metric(
					"eval_" + metric,
					summary="max" if metric not in ["loss", "mse"] else "min",
				)