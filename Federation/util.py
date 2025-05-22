from dragon_baseline import DragonBaseline
from dragon_baseline.main import CustomLogScaler
from pathlib import Path
import pandas as pd
import shutil
from dragon_baseline.nlp_algorithm import ProblemType
import os
import json

class DummyModel():
	"Class to create a dummy model for getting scaler parameters and task labels."
	def __init__(self, input_path: str = Path("/input")):
		self.input_path = input_path
		self.model = None
		self.set_up_dummy_model()

	def set_up_dummy_model(self):
		dummy_model = DragonBaseline(input_path=self.input_path)
		dummy_model.load()
		dummy_model.validate()
		dummy_model.analyze()
		dummy_model.scale_labels()
		self.model = dummy_model

	def get_scaler_params(self):
		"""
		Utility function to get the scaler parameters from a DragonBaseline model.
		"""
		if not self.model.label_scalers:
			return None
		elif type(self.model.label_scalers[self.model.task.target.label_name]) == CustomLogScaler:
			scaler = self.model.label_scalers[self.model.task.target.label_name].standard_scaler
		else:
			scaler = self.model.label_scalers[self.model.task.target.label_name]

		return {"mean_": scaler.mean_, "scale_": scaler.scale_, "var_": scaler.var_}

	def get_label2id(self):
		"""
		Utility function to get the label2id dict from a DragonBaseline model.
		Since not all clients have all labels, we need to configure this prior
		to splitting the data per center.
		"""
		if self.model.task.target.problem_type in [ProblemType.SINGLE_LABEL_MULTI_CLASS_CLASSIFICATION, ProblemType.MULTI_LABEL_NER, ProblemType.SINGLE_LABEL_NER]:
			return {str(label): idx for idx, label in enumerate(self.model.task.target.values)}
		else:
			return None

def get_center_names(input_path: str = Path("/input")):
	"""
	Utility function to get the center names from the input data directory.
	"""
	train_df = pd.read_json(Path(input_path) / Path("nlp-training-dataset.json"))
	return sorted(train_df['center'].unique().tolist())

def center_split_input_data(input_path: str = Path("/input"), split_input_path: str = Path("/opt/app/split_input")):
	for dataset in ["nlp-training-dataset.json", "nlp-validation-dataset.json", "nlp-test-dataset.json"]:
		df = pd.read_json(Path(input_path) / Path(dataset))
		for center in df['center'].unique():
			center_df = df[df['center'] == center]
			center_output_path = Path(split_input_path) / Path(str(center))
			center_output_path.mkdir(parents=True, exist_ok=True)
			center_df.to_json(center_output_path / Path(dataset), orient='records', lines=False)

			if dataset == "nlp-training-dataset.json":
				# also copy the label2id mapping for the training dataset
				shutil.copy(Path(input_path) / Path("nlp-task-configuration.json"), center_output_path / Path("nlp-task-configuration.json"))

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

def parse_config(config_file = '/opt/ml/model/config.json'):
	with open(config_file, 'r') as f:
		config = json.load(f)
	return config["federation_config"], config["model_kwargs"], config["backend_config"]