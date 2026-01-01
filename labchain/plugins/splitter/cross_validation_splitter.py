from typing import Any, Dict, Optional, cast
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from labchain import BasePipeline, BasePlugin, Container
from labchain.base.base_clases import BaseFilter, rprint
from labchain.base.base_splitter import BaseSplitter
from labchain.base.base_types import XYData


@Container.bind()
class KFoldSplitter(BaseSplitter):
    """
    A K-Fold cross-validation splitter for evaluating machine learning models.

    This class implements K-Fold cross-validation, which splits the dataset into K equally sized folds.
    The model is trained on K-1 folds and validated on the remaining fold. This process is repeated K times,
    with each fold serving as the validation set once.

    Key Features:
        - Configurable number of splits
        - Option to shuffle data before splitting
        - Supports custom pipelines for model training and evaluation
        - Provides mean loss across all folds

    Usage:
        ```python
        from framework3.plugins.splitter import KFoldSplitter
        from framework3.plugins.pipelines.sequential import F3Pipeline
        from framework3.base import XYData
        import numpy as np

        # Create a dummy pipeline
        pipeline = F3Pipeline(filters=[...], metrics=[...])

        # Create the KFoldSplitter
        splitter = KFoldSplitter(n_splits=5, shuffle=True, random_state=42, pipeline=pipeline)

        # Prepare some dummy data
        X = XYData(value=np.random.rand(100, 10))
        y = XYData(value=np.random.randint(0, 2, 100))

        # Fit and evaluate the model using cross-validation
        mean_loss = splitter.fit(X, y)
        print(f"Mean loss across folds: {mean_loss}")

        # Make predictions on new data
        X_new = XYData(value=np.random.rand(20, 10))
        predictions = splitter.predict(X_new)
        ```

    Attributes:
        n_splits (int): Number of folds. Must be at least 2.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (int): Controls the shuffling applied to the data before applying the split.
        pipeline (BaseFilter | None): The pipeline to be used for training and evaluation.

    Methods:
        split(pipeline: BaseFilter): Set the pipeline for the splitter.
        fit(x: XYData, y: XYData | None) -> Optional[float]: Perform K-Fold cross-validation.
        predict(x: XYData) -> XYData: Make predictions using the fitted pipeline.
        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData) -> Dict[str, Any]:
            Evaluate the pipeline using the last fold.
        start(x: XYData, y: Optional[XYData], X_: Optional[XYData]) -> Optional[XYData]:
            Start the cross-validation process and optionally make predictions.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        pipeline: BaseFilter | None = None,
        # evaluator: BaseMetric | None = None
    ):
        """
        Initialize the KFoldSplitter.

        Args:
            n_splits (int, optional): Number of folds. Must be at least 2. Defaults to 5.
            shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to True.
            random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to 42.
            pipeline (BaseFilter | None, optional): The pipeline to be used for training and evaluation. Defaults to None.
        """
        super().__init__(pipeline=pipeline)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self._kfold = KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
        self.pipeline = pipeline
        # self.evaluator = evaluator

    def split(self, pipeline: BaseFilter):
        """
        Set the pipeline for the splitter and disable its verbosity.

        Args:
            pipeline (BaseFilter): The pipeline to be used for training and evaluation.
        """
        self.pipeline = pipeline
        self.pipeline.verbose(False)

    def fit(self, x: XYData, y: XYData | None) -> Optional[float | dict]:
        """
        Perform K-Fold cross-validation on the given data.

        This method splits the data into K folds, trains the pipeline on K-1 folds,
        and evaluates it on the remaining fold. This process is repeated K times.

        Args:
            x (XYData): The input features.
            y (XYData | None): The target values.

        Returns:
            Optional[float]: The mean loss across all folds, or None if no losses were calculated.

        Raises:
            ValueError: If y is None or if the pipeline is not set.
        """
        self._print_acction("Fitting with KFold Splitter...")
        if self._verbose:
            rprint(self.pipeline)

        X = x.value
        if y is None:  # type: ignore
            raise ValueError("y must be provided for KFold split")

        if self.pipeline is None:
            raise ValueError("Pipeline must be fitted before splitting")

        losses: dict = {}
        splits = self._kfold.split(X)
        for train_idx, val_idx in tqdm(
            splits, total=self._kfold.get_n_splits(X), disable=not self._verbose
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

        This method performs cross-validation by fitting the model and then
        makes predictions if X_ is provided.

        Args:
            x (XYData): The input features for training.
            y (Optional[XYData]): The target values for training.
            X_ (Optional[XYData]): The input features for prediction, if different from x.

        Returns:
            Optional[XYData]: Prediction results if X_ is provided, else None.

        Raises:
            Exception: If an error occurs during the process.
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

        This method uses the pipeline that was fitted during cross-validation
        to make predictions on new data.

        Args:
            x (XYData): The input features for prediction.

        Returns:
            XYData: The predictions made by the pipeline.

        Raises:
            ValueError: If the pipeline has not been fitted.
        """
        self._print_acction("Predicting with KFold Splitter...")
        if self._verbose:
            rprint(self.pipeline)

        # X = x.value
        if self.pipeline is None:
            raise ValueError("Pipeline must be fitted before prediction")

        return self.pipeline.predict(x)

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Dict[str, Any]:
        """
        Evaluate the pipeline using the provided data.

        This method uses the pipeline's evaluate method to assess its performance
        on the given data.

        Args:
            x_data (XYData): The input features.
            y_true (XYData | None): The true target values.
            y_pred (XYData): The predicted target values.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation metrics.

        Raises:
            ValueError: If the pipeline has not been fitted.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be fitted before evaluation")
        return self.pipeline.evaluate(x_data, y_true, y_pred)
