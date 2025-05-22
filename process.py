from Federation.Federation.server import set_up_server_function
from Federation.Federation.client import set_up_client_function, Model, FederatedDragonClient
from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation
from pathlib import Path
import numpy as np
from transformers import HfArgumentParser
from Federation.Federation.config import ServerArguments, ClientArguments, TaskArguments
from Federation.util import DummyModel, get_center_names, center_split_input_data, get_weights_path, parse_config
import argparse
import json

def get_args():
    parser = argparse.ArgumentParser(description="Federated learning simulation process script")
    parser.add_argument("--input_path", type=Path, default=Path("/input"), help="Path to the input directory")
    parser.add_argument("--output_path", type=Path, default=Path("/output"), help="Path to the output directory")
    parser.add_argument("--workdir", type=Path, default=Path("/opt/app"), help="Path to the working directory")
    parser.add_argument("--input_split_dir", type=Path, default=Path("/opt/app/split_input"), help="Path to the split input directory")
    parser.add_argument("--config_file", type=Path, default=Path("/opt/ml/model/config.json"), help="Path to the configuration file")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    with open(args.input_path / "nlp-task-configuration.json", 'r') as f:
        task_config = json.load(f)
    task_name = task_config["task_name"]
    dummy_model = DummyModel(input_path=args.input_path)
    scaler_params = dummy_model.get_scaler_params()
    label2id = dummy_model.get_label2id()
    center_names = get_center_names(input_path=args.input_path)

    
    print(80 * "=")
    print(f"Running task: {task_name}\nScaler params: {scaler_params}\nLabel2id: {label2id}")
    print(80 * "=" + "\n")

    center_split_input_data(input_path=args.input_path, split_input_path=args.input_split_dir)

    config = {
        "task" : task_name,
        "server_base_dir" : args.workdir / Path("server"),
        "center_names" : center_names,
        "num_centers" : len(center_names),
        "input_path" : args.input_split_dir,
        "output_path" : args.output_path,
        "workdir" : args.workdir
    }

    federation_config, model_kwargs, backend_config = parse_config(config_file=args.config_file)
    config.update(federation_config)

    model_kwargs.update({'label2id': label2id})
    
    config_parser = HfArgumentParser((TaskArguments, ClientArguments, ServerArguments))
    task_arguments, client_arguments, server_arguments = config_parser.parse_dict(config)
    
    client_fn = set_up_client_function(client_arguments=client_arguments, task_arguments=task_arguments, model_kwargs=model_kwargs, scaler_params=scaler_params)
    server_fn = set_up_server_function(task_arguments = task_arguments, server_arguments=server_arguments)

    server = ServerApp(server_fn=server_fn)
    client = ClientApp(client_fn=client_fn)

    run_simulation(
        server_app = server,
        client_app = client,
        num_supernodes = config["num_centers"],
        backend_config = backend_config
    )

    config['workdir'] = config['workdir'] / Path(str(center_names[0]))
    config['input_path'] = args.input_path

    predModel = Model(input_path = config["input_path"], output_path = config["output_path"], workdir = config["workdir"], model_kwargs=model_kwargs)
    predModel.scaler_params = scaler_params
    predModel.setup()

    client = FederatedDragonClient(dragon_baseline=predModel, client_id="client", client_arguments=client_arguments, task_arguments=task_arguments)

    parameters = np.load(get_weights_path(config["server_base_dir"], file_identifier="best"))
    parameters = [parameters[file] for file in parameters._files]
    client.set_parameters(parameters)

    predictions = client.dragon_baseline.predict(df=client.dragon_baseline.df_test)
    client.dragon_baseline.save(predictions)
    client.dragon_baseline.verify_predictions()

if __name__ == "__main__":
    main()