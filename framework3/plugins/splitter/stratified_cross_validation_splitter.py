from typing import Any, Dict, Optional, cast
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from framework3 import Container
from framework3.base.base_clases import BaseFilter, BasePlugin, rprint
from framework3.base.base_pipelines import BasePipeline
from framework3.base.base_splitter import BaseSplitter
from framework3.base.base_types import XYData


@Container.bind()
class StratifiedKFoldSplitter(BaseSplitter):
    """
    A Stratified K-Fold cross-validation splitter for evaluating classification models.

    This class implements Stratified K-Fold cross-validation, which splits the dataset into K folds
    while preserving the percentage of samples for each class. It is particularly useful for imbalanced datasets.

    Key Features:
        - Preserves label distribution across folds
        - Configurable number of splits
        - Option to shuffle data before splitting
        - Supports custom pipelines for model training and evaluation
        - Provides mean loss across all folds

    Usage:
        ```python
        from framework3.plugins.splitter import StratifiedKFoldSplitter
        from framework3.plugins.pipelines.sequential import F3Pipeline
        from framework3.base import XYData
        import numpy as np

        pipeline = F3Pipeline(filters=[...], metrics=[...])
        splitter = StratifiedKFoldSplitter(n_splits=5, shuffle=True, random_state=42, pipeline=pipeline)

        X = XYData(value=np.random.rand(100, 10))
        y = XYData(value=np.random.randint(0, 2, 100))

        mean_loss = splitter.fit(X, y)
        print(f"Mean loss across folds: {mean_loss}")
        ```

    Attributes:
        n_splits (int): Number of folds.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (int): Random seed for reproducibility.
        pipeline (BaseFilter | None): The pipeline to be used for training and evaluation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        pipeline: BaseFilter | None = None,
    ):
        """
        Initialize the StratifiedKFoldSplitter.

        Args:
            n_splits (int): Number of folds. Must be at least 2.
            shuffle (bool): Whether to shuffle the data before splitting.
            random_state (int): Controls the shuffling applied to the data before splitting.
            pipeline (BaseFilter | None): The pipeline used for model training and evaluation.
        """
        super().__init__(pipeline=pipeline)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self._skf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
        self.pipeline = pipeline

    def split(self, pipeline: BaseFilter):
        """
        Set the pipeline for the splitter and disable its verbosity.

        Args:
            pipeline (BaseFilter): The pipeline used for training and evaluation.
        """
        self.pipeline = pipeline
        self.pipeline.verbose(False)

    def fit(self, x: XYData, y: XYData | None) -> Optional[float | dict]:
        """
        Perform Stratified K-Fold cross-validation on the given data.

        Args:
            x (XYData): Input features.
            y (XYData | None): Target labels.

        Returns:
            Optional[float]: Mean loss across all folds.

        Raises:
            ValueError: If y is None or the pipeline is not set.
        """
        self._print_acction("Fitting with StratifiedKFold Splitter...")
        if self._verbose:
            rprint(self.pipeline)

        if y is None:
            raise ValueError("y must be provided for Stratified K-Fold split")

        if self.pipeline is None:
            raise ValueError("Pipeline must be fitted before splitting")

        X = x.value
        Y = y.value

        losses: dict = {}
        splits = self._skf.split(X, Y)
        for train_idx, val_idx in tqdm(
            splits, total=self._skf.get_n_splits(X, Y), disable=not self._verbose
        ):
            X_train = x.split(train_idx)
            X_val = x.split(val_idx)
            y_train = y.split(train_idx)
            y_val = y.split(val_idx)

            pipeline = cast(
                BasePipeline,
                BasePlugin.build_from_dump(self.pipeline.item_dump(), Container.pif),
            )

            pipeline.fit(X_train, y_train)

            _y = pipeline.predict(X_val)

            loss = pipeline.evaluate(X_val, y_val, _y)
            for metric, value in loss.items():
                v = losses.get(metric, [])
                v.append(value)
                losses[metric] = v

            self.clear_memory()

        means = dict(map(lambda item: (item[0], np.mean(item[1])), losses.items()))
        stds = dict(
            map(lambda item: (f"{item[0]}_std", np.std(item[1])), losses.items())
        )
        scores = dict(map(lambda item: (f"{item[0]}_scores", item[1]), losses.items()))

        return means | stds | scores

    def start(
        self, x: XYData, y: Optional[XYData], X_: Optional[XYData]
    ) -> Optional[XYData]:
        """
        Start the cross-validation process and optionally make predictions.

        Args:
            x (XYData): Input training features.
            y (Optional[XYData]): Target labels.
            X_ (Optional[XYData]): Optional input data for prediction.

        Returns:
            Optional[XYData]: Predictions if X_ is provided, else predictions on training data.

        Raises:
            Exception: If any error occurs during execution.
        """
        try:
            self.fit(x, y)
            if X_ is not None:
                return self.predict(X_)
            else:
                return self.predict(x)
        except Exception as e:
            print(f"Error during pipeline execution: {e}")
            raise e

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using the fitted pipeline.

        Args:
            x (XYData): Input data for prediction.

        Returns:
            XYData: Predictions from the trained pipeline.

        Raises:
            ValueError: If pipeline is not fitted.
        """
        self._print_acction("Predicting with StratifiedKFold Splitter...")
        if self._verbose:
            rprint(self.pipeline)

        if self.pipeline is None:
            raise ValueError("Pipeline must be fitted before prediction")

        return self.pipeline.predict(x)

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Dict[str, Any]:
        """
        Evaluate the pipeline using the provided data.

        Args:
            x_data (XYData): Input features.
            y_true (XYData | None): Ground truth labels.
            y_pred (XYData): Predictions from the pipeline.

        Returns:
            Dict[str, Any]: Evaluation metrics.

        Raises:
            ValueError: If the pipeline is not fitted.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be fitted before evaluation")
        return self.pipeline.evaluate(x_data, y_true, y_pred)
