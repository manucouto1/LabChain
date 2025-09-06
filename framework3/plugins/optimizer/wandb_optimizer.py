from typing import Any, Dict, cast

import wandb
from framework3 import Container
from framework3.base import BaseMetric
from framework3.base.base_clases import BaseFilter, BasePlugin
from framework3.base.base_optimizer import BaseOptimizer
from framework3.base.base_types import XYData
from framework3.utils.wandb import WandbAgent, WandbSweepManager

from rich import print

__all__ = ["WandbOptimizer"]


@Container.bind()
class WandbOptimizer(BaseOptimizer):
    """
    Weights & Biases (wandb) based optimizer for hyperparameter tuning.

    This class implements hyperparameter optimization using Weights & Biases' sweep functionality.
    It allows for efficient searching of hyperparameter spaces for machine learning models
    within the Framework3 pipeline system.

    Key Features:
        - Integrates with Weights & Biases for distributed hyperparameter optimization
        - Supports various types of hyperparameters
        - Allows for customizable scoring metrics
        - Integrates with the Framework3 pipeline system

    Usage:
        The WandbOptimizer can be used to optimize hyperparameters of a machine learning pipeline:

        ```python
        from framework3.plugins.optimizer import WandbOptimizer
        from framework3.base import XYData, F1

        # Assuming you have a pipeline and data
        pipeline = ...
        x_data = XYData(...)
        y_data = XYData(...)

        optimizer = WandbOptimizer(project="my_project", scorer=F1(), pipeline=pipeline)
        optimizer.fit(x_data, y_data)

        best_pipeline = optimizer.pipeline
        ```

    Attributes:
        project (str): The name of the Weights & Biases project.
        scorer (BaseMetric): The scoring metric for evaluation.
        sweep_id (str | None): The ID of the Weights & Biases sweep.
        pipeline (BaseFilter | None): The pipeline to be optimized.

    Methods:
        optimize(pipeline: BaseFilter) -> None: Set up the optimization process for a given pipeline.
        fit(x: XYData, y: XYData | None) -> None: Perform the hyperparameter optimization.
        predict(x: XYData) -> XYData: Make predictions using the best pipeline found.
        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData) -> Dict[str, Any]:
            Evaluate the optimized pipeline.
    """

    def __init__(
        self,
        project: str,
        scorer: BaseMetric,
        pipeline: BaseFilter | None = None,
        sweep_id: str | None = None,
    ):
        """
        Initialize the WandbOptimizer.

        Args:
            project (str): The name of the Weights & Biases project.
            scorer (BaseMetric): The scoring metric for evaluation.
            pipeline (BaseFilter | None): The pipeline to be optimized. Defaults to None.
            sweep_id (str | None): The ID of an existing Weights & Biases sweep. Defaults to None.
        """
        super().__init__()
        self.project = project
        self.scorer = scorer
        self.sweep_id = sweep_id
        self.pipeline = pipeline

    def optimize(self, pipeline: BaseFilter) -> None:
        """
        Set up the optimization process for a given pipeline.

        This method prepares the pipeline for optimization by Weights & Biases.

        Args:
            pipeline (BaseFilter): The pipeline to be optimized.
        """
        self.pipeline = pipeline
        self.pipeline.verbose(False)

    def get_grid(self, aux: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Recursively process the grid configuration of a pipeline or filter.

        This method traverses the configuration dictionary and updates the parameters
        based on the Weights & Biases configuration.

        Args:
            aux (Dict[str, Any]): The configuration dictionary to process.
            config (Dict[str, Any]): The Weights & Biases configuration.

        Note:
            This method modifies the input dictionary in-place.
        """
        match aux["params"]:
            case {"filters": filters, **r}:
                for filter_config in filters:
                    self.get_grid(filter_config, config)
            case {"pipeline": pipeline, **r}:  # noqa: F841
                self.get_grid(pipeline, config)
            case {"filter": cached_filter, **r}:  # noqa: F841
                self.get_grid(cached_filter, config)
            case p_params:
                if "_grid" in aux:
                    for param, value in aux["_grid"].items():
                        p_params.update({param: config[aux["clazz"]][param]})

    def exec(
        self, config: Dict[str, Any], x: XYData, y: XYData | None = None
    ) -> Dict[str, float]:
        """
        Execute a single run of the pipeline with a given configuration.

        This method is called by the Weights & Biases agent for each hyperparameter configuration.

        Args:
            config (Dict[str, Any]): The hyperparameter configuration to test.
            x (XYData): The input features.
            y (XYData | None): The target values.

        Returns:
            Dict[str, float]: A dictionary containing the score for the current configuration.

        Raises:
            ValueError: If the pipeline is not properly configured or returns unexpected results.
        """
        if self.pipeline is None and self.sweep_id is None or self.project == "":
            raise ValueError("Either pipeline or sweep_id must be provided")

        self.get_grid(config["pipeline"], config["filters"])

        pipeline: BaseFilter = cast(
            BaseFilter, BasePlugin.build_from_dump(config["pipeline"], Container.pif)
        )

        pipeline.verbose(False)

        match pipeline.fit(x, y):
            case None:
                losses = pipeline.evaluate(x, y, pipeline.predict(x))
                loss = losses.pop(self.scorer.__class__.__name__, 0.0)
                wandb.log(dict(losses))  # type: ignore[attr-defined]

                return {self.scorer.__class__.__name__: float(loss)}
            case float() as loss:
                return {self.scorer.__class__.__name__: loss}
            case _:
                raise ValueError("Unexpected return type from pipeline.fit()")

    def fit(self, x: XYData, y: XYData | None = None) -> None:
        """
        Perform the hyperparameter optimization.

        This method creates a Weights & Biases sweep if necessary, runs the optimization,
        and fits the best pipeline found.

        Args:
            x (XYData): The input features.
            y (XYData | None): The target values.

        Raises:
            ValueError: If neither pipeline nor sweep_id is provided.
        """
        if self.sweep_id is None and self.pipeline is not None:
            self.sweep_id = WandbSweepManager().create_sweep(
                self.pipeline, self.project, scorer=self.scorer, x=x, y=y
            )

        if self.sweep_id is not None:
            sweep = WandbSweepManager().get_sweep(self.project, self.sweep_id)
            sweep_state = sweep.state.lower()
            if sweep_state not in ("finished", "cancelled", "crashed"):
                WandbAgent()(
                    self.sweep_id, self.project, lambda config: self.exec(config, x, y)
                )
        else:
            raise ValueError("Either pipeline or sweep_id must be provided")

        winner = WandbSweepManager().get_best_config(
            self.project, self.sweep_id, self.scorer.__class__.__name__
        )

        print(winner)

        self.get_grid(winner["pipeline"], winner["filters"])
        self.pipeline = cast(
            BaseFilter, BasePlugin.build_from_dump(winner["pipeline"], Container.pif)
        )

        self.pipeline.unwrap().fit(x, y)

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using the best pipeline found.

        Args:
            x (XYData): The input features.

        Returns:
            XYData: The predicted values.

        Raises:
            ValueError: If the pipeline has not been fitted.
        """
        if self.pipeline is not None:
            return self.pipeline.predict(x)
        else:
            raise ValueError("Pipeline must be fitted before predicting")

    def start(self, x: XYData, y: XYData | None, X_: XYData | None) -> XYData | None:
        """
        Start the pipeline execution.

        Args:
            x (XYData): Input data for fitting.
            y (XYData | None): Target data for fitting.
            X_ (XYData | None): Data for prediction (if different from x).

        Returns:
            XYData | None: Prediction results if X_ is provided, else None.

        Raises:
            ValueError: If the pipeline has not been fitted.
        """
        if self.pipeline is not None:
            return self.pipeline.start(x, y, X_)
        else:
            raise ValueError("Pipeline must be fitted before starting")

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Dict[str, Any]:
        """
        Evaluate the optimized pipeline.

        Args:
            x_data (XYData): Input data.
            y_true (XYData | None): True target data.
            y_pred (XYData): Predicted target data.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation results.
        """
        return (
            self.pipeline.evaluate(x_data, y_true, y_pred)
            if self.pipeline is not None
            else {}
        )
