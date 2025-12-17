from typing import Any, Callable, Dict, List, Literal, Optional
import wandb
from rich import print
from framework3.base import BaseMetric, XYData, BaseFilter


class WandbSweepManager:
    """
    A manager class for creating and handling Weights & Biases sweeps.

    This class provides methods to generate sweep configurations, create sweeps,
    and manage sweep runs for hyperparameter optimization using Weights & Biases.

    Key Features:
        - Generate sweep configurations from BaseFilter pipelines
        - Create sweeps with customizable parameters and metrics
        - Retrieve sweep information and best configurations
        - Restart sweeps by deleting specific runs

    Usage:
        ```python
        from framework3.utils.wandb import WandbSweepManager
        from framework3.base import BaseFilter, BaseMetric, XYData

        # Create a WandbSweepManager instance
        sweep_manager = WandbSweepManager()

        # Create a sweep
        pipeline = YourPipeline()
        scorer = YourScorer()
        x_data = XYData(...)
        y_data = XYData(...)
        sweep_id = sweep_manager.create_sweep(pipeline, "your_project", scorer, x_data, y_data)

        # Get the best configuration from a sweep
        best_config = sweep_manager.get_best_config("your_project", sweep_id, order="ascending")

        # Restart a sweep
        sweep = sweep_manager.get_sweep("your_project", sweep_id)
        sweep_manager.restart_sweep(sweep, states=["failed", "crashed"])
        ```

    Methods:
        get_grid(aux: Dict[str, Any], config: Dict[str, Any]) -> None:
            Recursively extract grid search parameters from a pipeline configuration.
        generate_config_for_pipeline(pipeline: BaseFilter) -> Dict[str, Any]:
            Generate a Weights & Biases sweep configuration from a BaseFilter pipeline.
        create_sweep(pipeline: BaseFilter, project_name: str, scorer: BaseMetric, x: XYData, y: XYData | None = None) -> str:
            Create a new sweep in Weights & Biases.
        get_sweep(project_name: str, sweep_id: str) -> Any:
            Retrieve a sweep object from Weights & Biases.
        get_best_config(project_name: str, sweep_id: str, order: str) -> Dict[str, Any]:
            Get the best configuration from a completed sweep.
        restart_sweep(sweep: Any, states: List[str] | Literal["all"] = "all") -> None:
            Restart a sweep by deleting runs with specified states.
        init(group: str, name: str, reinit: bool = True) -> Any:
            Initialize a new Weights & Biases run.
    """

    @staticmethod
    def get_grid(aux: Dict[str, Any], config: Dict[str, Any]):
        """
        Recursively extract grid search parameters from a pipeline configuration.

        Args:
            aux (Dict[str, Any]): The input configuration dictionary.
            config (Dict[str, Any]): The output configuration dictionary to be updated.
        """
        match aux["params"]:
            case {"filters": filters, **r}:
                for filter_config in filters:
                    WandbSweepManager.get_grid(filter_config, config)
            case {"filter": filter, **r}:
                WandbSweepManager.get_grid(filter, config)
            case {"pipeline": pipeline, **r}:  # noqa: F841
                WandbSweepManager.get_grid(pipeline, config)
            case p_params:
                if "_grid" in aux:
                    f_config = {}
                    for param, value in aux["_grid"].items():
                        print(f"categorical param: {param}: {value}")
                        p_params.update({param: value})
                        if isinstance(value, list):
                            f_config[param] = {"values": value}
                        elif isinstance(value, dict):
                            f_config[param] = value
                        else:
                            f_config[param] = {"value": value}
                    if len(f_config) > 0:
                        config["parameters"]["filters"]["parameters"][
                            str(aux["clazz"])
                        ] = {"parameters": f_config}

    @staticmethod
    def generate_config_for_pipeline(pipeline: BaseFilter) -> Dict[str, Any]:
        """
        Generate a Weights & Biases sweep configuration from a BaseFilter pipeline.

        Args:
            pipeline (BaseFilter): The pipeline to generate the configuration for.

        Returns:
            Dict[str, Any]: A Weights & Biases sweep configuration.
        """
        sweep_config: Dict[str, Dict[str, Dict[str, Any]]] = {
            "parameters": {"filters": {"parameters": {}}, "pipeline": {"value": {}}}
        }

        dumped_pipeline = pipeline.item_dump(include=["_grid"])

        WandbSweepManager.get_grid(dumped_pipeline, sweep_config)

        sweep_config["parameters"]["pipeline"]["value"] = dumped_pipeline

        return sweep_config

    def create_sweep(
        self,
        pipeline: BaseFilter,
        project_name: str,
        scorer: BaseMetric,
        x: XYData,
        y: XYData | None = None,
        method: Literal["grid", "random", "bayes"] = "grid",
        n_trials: Optional[int] = None,
        early_terminate: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new sweep in Weights & Biases.

        Args:
            pipeline (BaseFilter): The pipeline to be optimized.
            project_name (str): The name of the Weights & Biases project.
            scorer (BaseMetric): The metric used to evaluate the pipeline.
            x (XYData): The input data.
            y (XYData | None): The target data (optional).
            method (Literal["grid", "random", "bayes"]): Search method.
                - "grid": Exhaustive grid search (default antes)
                - "random": Random search
                - "bayes": Bayesian optimization (RECOMMENDED for DL)
            n_trials (Optional[int]): Number of trials to run (for random/bayes).
                If None and method="grid", runs all combinations.
            early_terminate (Optional[Dict[str, Any]]): Early termination config.
                Example: {"type": "hyperband", "min_iter": 5}

        Returns:
            str: The ID of the created sweep.

        Examples:
            >>> # Bayesian optimization (recommended)
            >>> sweep_id = manager.create_sweep(
            ...     pipeline, "my_project", scorer, x, y,
            ...     method="bayes",
            ...     n_trials=20,
            ...     early_terminate={"type": "hyperband", "min_iter": 5}
            ... )

            >>> # Grid search (exhaustive, backward compatible)
            >>> sweep_id = manager.create_sweep(
            ...     pipeline, "my_project", scorer, x, y,
            ...     method="grid"
            ... )
        """
        sweep_config = WandbSweepManager.generate_config_for_pipeline(pipeline)

        # ✅ NUEVO: Set method
        sweep_config["method"] = method

        # ✅ NUEVO: Set run_cap for random/bayes
        if method in ["random", "bayes"] and n_trials is not None:
            sweep_config["run_cap"] = n_trials

        # ✅ NUEVO: Early termination (saves compute!)
        if early_terminate is not None:
            sweep_config["early_terminate"] = early_terminate

        # Fixed parameters
        sweep_config["parameters"]["x_dataset"] = {"value": x._hash}
        sweep_config["parameters"]["y_dataset"] = (
            {"value": y._hash} if y is not None else {"value": "None"}
        )

        # Metric
        sweep_config["metric"] = {
            "name": scorer.__class__.__name__,
            "goal": "maximize" if scorer.higher_better else "minimize",
        }

        print("______________________SWEEP CONFIG_____________________")
        print(sweep_config)
        print("_______________________________________________________")

        return wandb.sweep(sweep_config, project=project_name)  # type: ignore

    def get_sweep(self, project_name, sweep_id) -> Any:
        """
        Retrieve a sweep object from Weights & Biases.

        Args:
            project_name (str): The name of the Weights & Biases project.
            sweep_id (str): The ID of the sweep to retrieve.

        Returns:
            (Any): The sweep object.
        """
        sweep = wandb.Api().sweep(f"citius-irlab/{project_name}/sweeps/{sweep_id}")  # type: ignore
        return sweep

    def get_best_config(self, project_name, sweep_id, order) -> Dict[str, Any]:
        """
        Get the best configuration from a completed sweep.

        Args:
            project_name (str): The name of the Weights & Biases project.
            sweep_id (str): The ID of the sweep.
            order (str): The order to use when selecting the best run ("ascending" or "descending").

        Returns:
            (Dict[str, Any]): The configuration of the best run.
        """
        sweep = self.get_sweep(project_name, sweep_id)
        winner_run = sweep.best_run(order=order)
        return dict(winner_run.config)

    def restart_sweep(self, sweep, states: List[str] | Literal["all"] = "all"):
        """
        Restart a sweep by deleting runs with specified states.

        Args:
            sweep (Any): The sweep object to restart.
            states (List[str] | Literal["all"]): The states of runs to delete, or "all" to delete all runs.
        """
        # Eliminar todas las ejecuciones fallidas
        for run in sweep.runs:
            if run.state in states or states == "all":
                run.delete()
                print("Deleting run:", run.id)

    def init(self, group: str, name: str, reinit=True) -> Any:
        """
        Initialize a new Weights & Biases run.

        Args:
            group (str): The group name for the run.
            name (str): The name of the run.
            reinit (bool): Whether to reinitialize if a run is already in progress.

        Returns:
            (Any): The initialized run object.
        """
        run = wandb.init(group=group, name=name, reinit=reinit)  # type: ignore
        return run


class WandbAgent:
    """
    A class to create and run Weights & Biases agents for sweeps.

    This class provides a callable interface to create and run Weights & Biases agents,
    which are used to execute sweep runs with specified configurations.

    Key Features:
        - Create and run Weights & Biases agents for sweeps
        - Execute custom functions with sweep configurations
        - Automatically handle initialization and teardown of Weights & Biases runs

    Usage:
        ```python
        from framework3.utils.wandb import WandbAgent

        def custom_function(config):
            # Your sweep run logic here
            result = ...
            return result

        agent = WandbAgent()
        agent("your_sweep_id", "your_project_name", custom_function)
        ```

    Methods:
        __call__(sweep_id: str, project: str, function: Callable) -> None:
            Create and run a Weights & Biases agent for a specified sweep.
    """

    @staticmethod
    def __call__(sweep_id: str, project: str, function: Callable) -> None:
        """
        Create and run a Weights & Biases agent for a specified sweep.

        This method initializes a Weights & Biases agent, executes the provided function
        with the sweep configuration, and handles the teardown of the Weights & Biases run.

        Args:
            sweep_id (str): The ID of the sweep to run.
            project (str): The name of the Weights & Biases project.
            function (Callable): A function that takes a configuration dictionary and returns a result to be logged.

        Returns:
            (None)
        """
        wandb.agent(  # type: ignore
            sweep_id,
            function=lambda: {
                wandb.init(reinit="finish_previous"),  # type: ignore
                wandb.log(function(dict(wandb.config))),  # type: ignore
            },
            project=project,
        )  # type: ignore
        wandb.teardown()  # type: ignore


class WandbRunLogger: ...
