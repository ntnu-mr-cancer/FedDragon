from flwr.server import ServerApp, ServerConfig, ServerAppComponents, start_server
from flwr.server.strategy import FedAvg
from flwr.common import Context
import torch
from safetensors.torch import save_file
from flwr.common.typing import UserConfig
from .strategy import CustomFedAvg
from .config import ServerArguments, TaskArguments
from functools import partial

def set_up_server_function(server_arguments: ServerArguments, task_arguments : TaskArguments) -> ServerApp:
    strategy = CustomFedAvg(
        server_arguments=server_arguments,
        task_arguments=task_arguments,
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=server_arguments.num_centers, 
        min_evaluate_clients=server_arguments.num_centers,  # Never sample less than 1 clients for evaluation
        min_available_clients=server_arguments.num_centers,  # Wait until all 1 clients are available
    )
    def server_fn(context: Context) -> ServerAppComponents:
        """Construct components that set the ServerApp behaviour.

        You can use the settings in `context.run_config` to parameterize the
        construction of all elements (e.g the strategy or the number of rounds)
        wrapped in the returned ServerAppComponents object.
        """

        config = ServerConfig(num_rounds=server_arguments.num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)
        
    return server_fn