from Federation.Federation.server import set_up_server_function
from Federation.Federation.client import set_up_client_function, Model, FederatedDragonClient
from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation
from pathlib import Path
import numpy as np
from transformers import HfArgumentParser
from Federation.Federation.config import ServerArguments, ClientArguments, TaskArguments
from Federation.util import DummyModel, get_center_names, center_split_input_data, get_weights_path
import argparse


# Defaults (preserve original values)
DEFAULT_INPUT_BASE_DIR = Path("test-input")
DEFAULT_INPUT_SPLIT_BASE_DIR = Path("test-input-split")
DEFAULT_WORKDIR = Path("test-workdir")
DEFAULT_OUTPUT_PATH = Path("test-output")


def parse_args():
    parser = argparse.ArgumentParser(description="Run federated simulation over local test tasks")
    parser.add_argument(
        "--input-base-dir",
        type=Path,
        default=DEFAULT_INPUT_BASE_DIR,
        help=f"Directory containing Task* folders (default: {DEFAULT_INPUT_BASE_DIR})",
    )
    parser.add_argument(
        "--input-split-base-dir",
        type=Path,
        default=DEFAULT_INPUT_SPLIT_BASE_DIR,
        help=f"Directory where per-center split inputs are written (default: {DEFAULT_INPUT_SPLIT_BASE_DIR})",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=DEFAULT_WORKDIR,
        help=f"Working directory for artifacts (default: {DEFAULT_WORKDIR})",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Optional base output path (default: {DEFAULT_OUTPUT_PATH})",
    )

    parser.add_argument(
        "--num-rounds",
        type=int,
        default=5,
        help="Number of federated training rounds (default: 5)",
    )
    
    parser.add_argument(
        "--epochs_per_round",
        type=int,
        default=3,
        help="Number of local training epochs per round (default: 1)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, run in debug mode with a smaller dataset, rounds and epochs",
    )

    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="If set, run without GPU",
    )
    
    parser.add_argument(
        "--task_pattern",
        type=str,
        default="Task*",
        help="Pattern for task directories (default: 'Task*')",
    )
    return parser.parse_args()


args = parse_args()

input_base_dir = args.input_base_dir
input_split_base_dir = args.input_split_base_dir
input_split_base_dir.mkdir(parents=True, exist_ok=True)

workdir = args.workdir
workdir.mkdir(parents=True, exist_ok=True)

output_path = args.output_path
output_path.mkdir(parents=True, exist_ok=True)

task_list = list(input_base_dir.glob(args.task_pattern))
print('Task list:', task_list)

model_kwargs = dict(
    model_name_or_path = "joeranbosma/dragon-bert-base-mixed-domain",
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 2,
    gradient_checkpointing = False,
    max_seq_length = 512,
    learning_rate = 1e-05,
    lr_scheduler_type = "constant",
    save_total_limit = 1,
    warmup_ratio = 0,
    num_train_epochs = args.epochs_per_round,
    disable_tqdm = True,
    )

config = {"num_rounds": args.num_rounds}
backend_config = {"client_resources": {"num_cpus": 11, "num_gpus": 0.0 if args.no_gpu else 1.0}}

if args.debug:
    print("Running in debug mode with reduced rounds and epochs")
    config["num_rounds"] = 2
    model_kwargs["num_train_epochs"] = 1

for task_path in task_list:
    # setup task-specific variables
    task_name = task_path.name
    dummy_model = DummyModel(input_path=task_path)
    scaler_params = dummy_model.get_scaler_params()
    label2id = dummy_model.get_label2id()
    center_names = get_center_names(input_path=task_path)

    # center_split_input_data(input_path=task_path, split_input_path=input_split_base_dir / Path(f"{task_name}"))
    print(80 * "=")
    print(f"Running task: {task_name}\nScaler params: {scaler_params}\nLabel2id: {label2id}")
    print(80 * "=" + "\n")

    center_split_input_data(input_path=task_path, split_input_path=input_split_base_dir / Path(f"{task_name}"))

    model_kwargs.update({'label2id' : label2id})
    
    config.update({
        "task" : task_name.split('-')[0],
        "server_base_dir" : workdir / Path(task_name) / Path("server"),
        "center_names" : center_names,
        "num_centers" : len(center_names),
        "input_path" : input_split_base_dir / Path(task_name),
        "output_path" : output_path / Path(task_name),
        "workdir" : workdir / Path(task_name)
    })

    config_parser = HfArgumentParser((TaskArguments, ClientArguments, ServerArguments))
    task_arguments, client_arguments, server_arguments = config_parser.parse_dict(config)
    
    client_fn = set_up_client_function(client_arguments=client_arguments, task_arguments=task_arguments, model_kwargs=model_kwargs)
    server_fn = set_up_server_function(task_arguments = task_arguments, server_arguments=server_arguments)

    server = ServerApp(server_fn=server_fn)
    client = ClientApp(client_fn=client_fn)

    run_simulation(
        server_app = server,
        client_app = client,
        num_supernodes = config["num_centers"],
        backend_config = backend_config
    )

    # Predict when training is done
    config['workdir'] = config['workdir'] / Path(str(center_names[0]))
    config['input_path'] = input_base_dir / Path(task_name)

    predModel = Model(input_path = config["input_path"], output_path = config["output_path"], workdir = config["workdir"], model_kwargs=model_kwargs)
    predModel.setup()

    client = FederatedDragonClient(dragon_baseline=predModel, client_id="client", client_arguments=client_arguments, task_arguments=task_arguments)
    parameters = np.load(get_weights_path(config["server_base_dir"], file_identifier="best"))
    parameters = [parameters[file] for file in parameters._files]
    predictions = client.dragon_baseline.predict(df=client.dragon_baseline.df_test)
    client.dragon_baseline.save(predictions)
    client.dragon_baseline.verify_predictions()

