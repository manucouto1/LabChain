from typing import Any, Dict, Literal, Optional, cast

import wandb
from labchain import Container
from labchain.base import BaseMetric
from labchain.base.base_clases import BaseFilter, BasePlugin
from labchain.base.base_optimizer import BaseOptimizer
from labchain.base.base_types import XYData
from labchain.utils.wandb import WandbAgent, WandbSweepManager

from rich import print

__all__ = ["WandbOptimizer"]


@Container.bind()
class WandbOptimizer(BaseOptimizer):
    """
        Weights & Biases optimizer for hyperparameter tuning.

        Supports multiple search strategies:
            - grid: Exhaustive search (all combinations)
            - random: Random sampling
            - bayes: Bayesian optimization (recommended for DL)

        Usage:
    ```python
            # Bayesian optimization (recommended)
            optimizer = WandbOptimizer(
                project="my-project",
                scorer=F1(),
                pipeline=my_pipeline,
                method="bayes",
                n_trials=20,
                early_terminate={"type": "hyperband", "min_iter": 5}
            )
            optimizer.fit(x_train, y_train)
            predictions = optimizer.predict(x_test)
    ```

        Hyperparameter syntax in filters:
    ```python
            # Grid search (backward compatible)
            self._grid = {"lr": [1e-5, 1e-4, 1e-3], "dropout": [0.1, 0.2, 0.3]}

            # Bayesian optimization (new)
            self._grid = {
                "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
                "dropout": {"distribution": "uniform", "min": 0.1, "max": 0.3}
            }
    ```
    """

    def __init__(
        self,
        project: str,
        scorer: BaseMetric,
        pipeline: BaseFilter | None = None,
        sweep_id: str | None = None,
        method: Literal["grid", "random", "bayes"] = "grid",
        n_trials: Optional[int] = None,
        early_terminate: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the WandbOptimizer.

        Args:
            project: W&B project name
            scorer: Metric to optimize (maximize if higher_better=True)
            pipeline: Pipeline to optimize (must have _grid defined)
            sweep_id: Existing sweep ID to resume (optional)
            method: Search strategy - "grid", "random", or "bayes" (default: "bayes")
            n_trials: Max trials (None = unlimited for grid, required for random/bayes)
            early_terminate: Early stopping config, e.g.:
                {"type": "hyperband", "min_iter": 5, "s": 2, "eta": 3}
        """
        super().__init__()
        self.project = project
        self.scorer = scorer
        self.sweep_id = sweep_id
        self.pipeline = pipeline
        self.method = method
        self.n_trials = n_trials
        self.early_terminate = early_terminate

    def optimize(self, pipeline: BaseFilter) -> None:
        """
        Set up optimization for the given pipeline.

        Args:
            pipeline: Pipeline to optimize (sets verbose=False)
        """
        self.pipeline = pipeline
        self.pipeline.verbose(False)

    def get_grid(self, aux: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Recursively update pipeline params with W&B config.

        Args:
            aux: Pipeline configuration dict
            config: W&B sampled hyperparameters
        """
        match aux["params"]:
            case {"filters": filters, **r}:
                for filter_config in filters:
                    self.get_grid(filter_config, config)
            case {"pipeline": pipeline, **r}:
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
        Execute a single trial with given hyperparameters.

        Called by W&B agent for each configuration.

        Args:
            config: Sampled hyperparameters from W&B
            x: Training features
            y: Training targets

        Returns:
            Dict with scorer metric: {scorer_name: score}
        """
        if self.pipeline is None and self.sweep_id is None or self.project == "":
            raise ValueError("Either pipeline or sweep_id must be provided")

        # Update config with sampled params
        self.get_grid(config["pipeline"], config["filters"])

        # Rebuild pipeline
        pipeline: BaseFilter = cast(
            BaseFilter, BasePlugin.build_from_dump(config["pipeline"], Container.ppif)
        )
        pipeline.verbose(False)

        # Fit and evaluate
        match pipeline.fit(x, y):
            case None:
                losses = pipeline.evaluate(x, y, pipeline.predict(x))
                loss = losses.pop(self.scorer.__class__.__name__, 0.0)
                wandb.log(dict(losses))  # type: ignore
                return {self.scorer.__class__.__name__: float(loss)}

            case float() as loss:
                return {self.scorer.__class__.__name__: loss}

            case dict() as losses:
                loss = losses.pop(self.scorer.__class__.__name__, 0.0)
                wandb.log(dict(losses))  # type: ignore
                return {self.scorer.__class__.__name__: loss}

            case _:
                raise ValueError("Unexpected return type from pipeline.fit()")

    def fit(self, x: XYData, y: XYData | None = None) -> None:
        """
        Run hyperparameter optimization.

        Process:
            1. Create W&B sweep (or resume existing)
            2. Run optimization trials
            3. Load best config
            4. Fit final pipeline with best hyperparameters

        Args:
            x: Training features
            y: Training targets (optional for unsupervised)

        Raises:
            ValueError: If neither pipeline nor sweep_id provided
        """
        # Create sweep if needed
        if self.sweep_id is None and self.pipeline is not None:
            self.sweep_id = WandbSweepManager().create_sweep(
                pipeline=self.pipeline,
                project_name=self.project,
                scorer=self.scorer,
                x=x,
                y=y,
                method=self.method,  # type: ignore
                n_trials=self.n_trials,
                early_terminate=self.early_terminate,
            )

        # Run optimization
        if self.sweep_id is not None:
            sweep = WandbSweepManager().get_sweep(self.project, self.sweep_id)
            sweep_state = sweep.state.lower()

            if sweep_state not in ("finished", "cancelled", "crashed"):
                print(f"🚀 Starting sweep: {self.sweep_id}")
                print(
                    f"🔗 https://wandb.ai/citius-irlab/{self.project}/sweeps/{self.sweep_id}"
                )

                WandbAgent()(
                    self.sweep_id, self.project, lambda config: self.exec(config, x, y)
                )
            else:
                print(f"⚠️  Sweep {sweep_state}, loading best config")
        else:
            raise ValueError("Either pipeline or sweep_id must be provided")

        # Get best config
        print("📊 Retrieving best configuration...")
        winner = WandbSweepManager().get_best_config(
            self.project,
            self.sweep_id,
            order="descending" if self.scorer.higher_better else "ascending",
        )
        print(winner)

        # Rebuild and fit final pipeline
        self.get_grid(winner["pipeline"], winner["filters"])
        self.pipeline = cast(
            BaseFilter, BasePlugin.build_from_dump(winner["pipeline"], Container.ppif)
        )

        print("🎯 Fitting final pipeline...")
        self.pipeline.unwrap().fit(x, y)
        print("✅ Optimization complete!")

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions with optimized pipeline.

        Args:
            x: Input features

        Returns:
            Predictions

        Raises:
            ValueError: If not fitted
        """
        if self.pipeline is not None:
            return self.pipeline.predict(x)
        else:
            raise ValueError("Pipeline must be fitted before predicting")

    def start(self, x: XYData, y: XYData | None, X_: XYData | None) -> XYData | None:
        """
        Unified fit/predict interface.

        Args:
            x: Data for fitting
            y: Targets for fitting
            X_: Data for prediction (if different from x)

        Returns:
            Predictions if X_ provided, else None
        """
        if self.pipeline is not None:
            return self.pipeline.start(x, y, X_)
        else:
            raise ValueError("Pipeline must be fitted before starting")

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Dict[str, Any]:
        """
        Evaluate optimized pipeline.

        Args:
            x_data: Input data
            y_true: True targets
            y_pred: Predictions

        Returns:
            Evaluation metrics dict
        """
        return (
            self.pipeline.evaluate(x_data, y_true, y_pred)
            if self.pipeline is not None
            else {}
        )
