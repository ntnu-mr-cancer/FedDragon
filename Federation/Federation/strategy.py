from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays
from pathlib import Path
import json
import numpy as np
from .config import ServerArguments, TaskArguments
import shutil

def create_run_dir(client_base_dir):
    """Create a directory where to save results from this run."""
    # current_time = datetime.now()
    # run_dir = current_time.strftime("%Y-%m-%d-%H%M%S")
    # save_path = Path(f"{client_base_dir}/outputs/{run_dir}")
    save_path = Path(client_base_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path

class CustomFedAvg(FedAvg):
    def __init__(self, server_arguments : ServerArguments, task_arguments : TaskArguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_base_dir = server_arguments.server_base_dir 
        self.save_path = create_run_dir(self.server_base_dir)
        self.results = {}
        self.server_arguments = server_arguments
        self.task_arguments = task_arguments
        self.model_selection_metric = task_arguments.model_selection_metric
        self.current_best_metric = None
        self.parameters = None
        

    def save_fl_model(self, server_round, parameters):
        self.parameters = parameters
        ndarrays = parameters_to_ndarrays(parameters)
        print(f"Saving round {server_round} aggregated_ndarrays...")
        np.savez(self.save_path / f"weights_latest.npz", *ndarrays)

    def _store_results(self, tag: str, results_dict):
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        self._store_results(
            tag=tag,
            results_dict={"round": server_round, **results_dict},
        )

    def evaluate(self, server_round, parameters):
        self.save_fl_model(server_round, parameters)
        return None
    
    def aggregate_evaluate(self, server_round, results, failures):
        if failures:
            centers_without_failures = [r[1].metrics['center'] for r in results]
            centers_with_failures = [c for c in self.server_arguments.center_names if c not in centers_without_failures]
            print(f"Errors occurred during evaluation of clients: {centers_with_failures}")
            raise ValueError(f"Errors occurred during evaluation of clients: {failures}")
            
        # Aggregate client evaluation results
        metrics = [e[1].metrics for e in results]
        
        def weighted_average(metrics):
            metrics_to_average = self.task_arguments.all_evaluation_metrics + ['loss']
            metrics_to_average = ["eval_" + m for m in metrics_to_average]
            total_examples = sum([m["eval_samples"] for m in metrics])

            weighted_metrics = {
                metric_name: sum(m["eval_samples"] * m[metric_name] for m in metrics) / total_examples
                for metric_name in metrics_to_average
            }
            print(f"weighted_metrics: {weighted_metrics}")
            return weighted_metrics

        weighted_metrics = weighted_average(metrics)
        client_metrics = [dict(m) for m in metrics]
        self.store_results_and_log(
            server_round=server_round,
            tag="fed_evaluate",
            results_dict={"aggregated" : weighted_metrics, "client_metrics" : client_metrics},)
        
        if self.task_arguments.log_to_wandb:
            import wandb
            client_metric_dict = {}
            for i, cl in enumerate(client_metrics):
                prefix = cl["center"] + "/"
                client_metric_dict.update({
                        prefix + "eval_loss": cl["eval_loss"],
                        **{prefix + f"eval_{metric}": cl[f"eval_{metric}"] for metric in self.task_arguments.all_evaluation_metrics}
                    })
            metric_dict = {f"eval_{metric}": weighted_metrics[f"eval_{metric}"] for metric in self.task_arguments.all_evaluation_metrics}
            wandb.log(
                {
                    "eval_loss": weighted_metrics["eval_loss"],
                    **metric_dict,
                    **client_metric_dict,
                },
                step=server_round,
            )
            # wandb.log({ "eval_loss": weighted_metrics["eval_loss"], **metric_dict, **client_metric_dict}, step=server_round)
        def compare_metrics(metric_name):
            if self.task_arguments.greater_is_better:
                return weighted_metrics[metric_name] > self.current_best_metric
            else:
                return weighted_metrics[metric_name] < self.current_best_metric

        if self.current_best_metric is None or compare_metrics("eval_" + self.model_selection_metric):
            self.current_best_metric = weighted_metrics["eval_" + self.model_selection_metric]
            print(f"New best metric: {self.current_best_metric} at round {server_round}")
            # grab previous best weights for deletion
            delete_list = list(Path(self.save_path).glob("weights_best*.npz"))

            ndarrays = parameters_to_ndarrays(self.parameters)
            np.savez(self.save_path / f"weights_best_R_{server_round}_{self.model_selection_metric}_{self.current_best_metric:.2f}.npz", *ndarrays)

            if self.task_arguments.log_to_wandb:
                wandb.run.summary["best_round"] = server_round
            for f in delete_list:
                print(f"Deleting old best weights: {f}")
                f.unlink()
        return weighted_metrics["eval_loss"] , weighted_metrics
