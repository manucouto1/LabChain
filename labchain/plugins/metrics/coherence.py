from typing import Unpack, cast
from labchain.base.base_clases import BaseMetric
from labchain.base.base_types import Float
from labchain.base.base_types import XYData
from labchain.container.container import Container
from labchain.plugins.metrics.utils.coherence import Coherence
import numpy as np
import pandas as pd

from labchain.plugins.metrics.utils.types import CoherenceEvaluateKwargs


__all__ = ["NPMI", "UMASS", "V", "UCI"]


@Container.bind()
class NPMI(BaseMetric):
    """
    Normalized Pointwise Mutual Information (NPMI) coherence metric for topic modeling evaluation.

    This class calculates the NPMI coherence score, which measures the coherence of topics based on
    the normalized pointwise mutual information of word pairs.

    Key Features:
        - Measures topic coherence using normalized pointwise mutual information
        - Suitable for evaluating topic models
        - Handles input data as pandas DataFrames

    Usage:
        The NPMI metric can be used to evaluate topic modeling results:

        ```python
        from framework3.plugins.metrics.coherence import NPMI
        from framework3.base.base_types import XYData
        import pandas as pd
        import numpy as np

        # Assuming you have a DataFrame 'df' with your document-term matrix
        x_data = XYData(value=df)
        y_pred = np.array([['word1', 'word2', 'word3'], ['word4', 'word5', 'word6']])  # Example topics

        npmi_metric = NPMI()
        score = npmi_metric.evaluate(x_data, None, y_pred, f_vocab=df.columns)
        print(f"NPMI Score: {score}")
        ```

    Methods:
        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData, **kwargs) -> Float | np.ndarray:
            Calculate the NPMI coherence score.

    Note:
        This metric requires the input data to be a pandas DataFrame. Ensure that your data
        is properly formatted before using this metric.
    """

    def evaluate(
        self,
        x_data: XYData,
        y_true: XYData | None,
        y_pred: XYData,
        **kwargs: Unpack[CoherenceEvaluateKwargs],
    ) -> Float | np.ndarray:
        """
        Calculate the NPMI coherence score.

        This method computes the NPMI coherence score for the given topics.

        Args:
            x_data (XYData): The input data, expected to be a pandas DataFrame.
            y_true (XYData | None): Not used for this metric, but required by the interface.
            y_pred (XYData): The predicted topics, typically a list of lists of words.
            **kwargs (Unpack[EvaluateKwargs]): Additional keyword arguments:
                - f_vocab (list): The vocabulary of the corpus.
                - topk (int): The number of top words to consider for each topic (default: 10).
                - processes (int): The number of processes to use for parallel computation (default: 1).

        Returns:
            Float | np.ndarray: The NPMI coherence score.

        Raises:
            Exception: If x_data is not a pandas DataFrame.

        Note:
            This method uses the Coherence class from framework3.plugins.metrics.utils.coherence internally.
        """
        f_vocab = kwargs.get("f_vocab")
        topk = kwargs.get("topk", 10)
        processes = kwargs.get("processes", 1)
        coherence = Coherence(
            f_vocab=f_vocab, topk=topk, processes=processes, measure="c_npmi"
        )
        if isinstance(x_data.value, pd.DataFrame):
            return coherence.evaluate(df=x_data.value, predicted=y_pred)
        else:
            raise Exception("x_data must be a pandas DataFrame")


@Container.bind()
class UMASS(BaseMetric):
    """
    UMass coherence metric for topic modeling evaluation.

    This class calculates the UMass coherence score, which is based on document co-occurrence counts
    and a sliding window.

    Key Features:
        - Measures topic coherence using document co-occurrence
        - Suitable for evaluating topic models
        - Handles input data as pandas DataFrames

    Usage:
        The UMASS metric can be used to evaluate topic modeling results:

        ```python
        from framework3.plugins.metrics.coherence import UMASS
        from framework3.base.base_types import XYData
        import pandas as pd
        import numpy as np

        # Assuming you have a DataFrame 'df' with your document-term matrix
        x_data = XYData(value=df)
        y_pred = np.array([['word1', 'word2', 'word3'], ['word4', 'word5', 'word6']])  # Example topics

        umass_metric = UMASS()
        score = umass_metric.evaluate(x_data, None, y_pred, f_vocab=df.columns)
        print(f"UMass Score: {score}")
        ```

    Methods:
        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData, **kwargs) -> Float | np.ndarray:
            Calculate the UMass coherence score.

    Note:
        This metric requires the input data to be a pandas DataFrame. Ensure that your data
        is properly formatted before using this metric.
    """

    def evaluate(
        self,
        x_data: XYData,
        y_true: XYData | None,
        y_pred: XYData,
        **kwargs: Unpack[CoherenceEvaluateKwargs],
    ) -> Float | np.ndarray:
        """
        Calculate the UMass coherence score.

        This method computes the UMass coherence score for the given topics.

        Args:
            x_data (XYData): The input data, expected to be a pandas DataFrame.
            y_true (XYData | None): Not used for this metric, but required by the interface.
            y_pred (XYData): The predicted topics, typically a list of lists of words.
            **kwargs (Unpack[EvaluateKwargs]): Additional keyword arguments:
                - f_vocab (list): The vocabulary of the corpus.
                - topk (int): The number of top words to consider for each topic (default: 10).
                - processes (int): The number of processes to use for parallel computation (default: 1).

        Returns:
            Float | np.ndarray: The UMass coherence score.

        Raises:
            Exception: If x_data is not a pandas DataFrame.

        Note:
            This method uses the Coherence class from framework3.plugins.metrics.utils.coherence internally.
        """
        f_vocab = kwargs.get("f_vocab")
        topk = kwargs.get("topk", 10)
        processes = kwargs.get("processes", 1)
        coherence = Coherence(
            f_vocab=f_vocab, topk=topk, processes=processes, measure="u_mass"
        )
        if isinstance(x_data.value, pd.DataFrame):
            return coherence.evaluate(df=x_data.value, predicted=y_pred)
        else:
            raise Exception("x_data must be a pandas DataFrame")


@Container.bind()
class V(BaseMetric):
    """
    V-measure coherence metric for topic modeling evaluation.

    This class calculates the V-measure coherence score, which is based on a combination of
    homogeneity and completeness.

    Key Features:
        - Measures topic coherence using a combination of homogeneity and completeness
        - Suitable for evaluating topic models
        - Handles input data as pandas DataFrames

    Usage:
        The V-measure metric can be used to evaluate topic modeling results:

        ```python
        from framework3.plugins.metrics.coherence import V
        from framework3.base.base_types import XYData
        import pandas as pd
        import numpy as np

        # Assuming you have a DataFrame 'df' with your document-term matrix
        x_data = XYData(value=df)
        y_pred = np.array([['word1', 'word2', 'word3'], ['word4', 'word5', 'word6']])  # Example topics

        v_metric = V()
        score = v_metric.evaluate(x_data, None, y_pred, f_vocab=df.columns)
        print(f"V-measure Score: {score}")
        ```

    Methods:
        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData, **kwargs) -> Float | np.ndarray:
            Calculate the V-measure coherence score.

    Note:
        This metric requires the input data to be a pandas DataFrame. Ensure that your data
        is properly formatted before using this metric.
    """

    def evaluate(
        self,
        x_data: XYData,
        y_true: XYData | None,
        y_pred: XYData,
        **kwargs: Unpack[CoherenceEvaluateKwargs],
    ) -> Float | np.ndarray:
        """
        Calculate the V-measure coherence score.

        This method computes the V-measure coherence score for the given topics.

        Args:
            x_data (XYData): The input data, expected to be a pandas DataFrame.
            y_true (XYData | None): Not used for this metric, but required by the interface.
            y_pred (XYData): The predicted topics, typically a list of lists of words.
            **kwargs (Unpack[EvaluateKwargs]): Additional keyword arguments:
                - f_vocab (list): The vocabulary of the corpus.
                - topk (int): The number of top words to consider for each topic (default: 10).
                - processes (int): The number of processes to use for parallel computation (default: 1).

        Returns:
            Float | np.ndarray: The V-measure coherence score.

        Raises:
            Exception: If x_data is not a pandas DataFrame.

        Note:
            This method uses the Coherence class from framework3.plugins.metrics.utils.coherence internally.
        """
        f_vocab = kwargs.get("f_vocab")
        topk: int = cast(int, kwargs.get("topk", 10))
        processes: int = cast(int, kwargs.get("processes", 1))
        coherence = Coherence(
            f_vocab=f_vocab, topk=topk, processes=processes, measure="c_v"
        )
        if isinstance(x_data.value, pd.DataFrame):
            return coherence.evaluate(df=x_data.value, predicted=y_pred)
        else:
            raise Exception("x_data must be a pandas DataFrame")


@Container.bind()
class UCI(BaseMetric):
    """
    UCI coherence metric for topic modeling evaluation.

    This class calculates the UCI coherence score, which is based on pointwise mutual information (PMI)
    of all word pairs in a topic.

    Key Features:
        - Measures topic coherence using pointwise mutual information of word pairs
        - Suitable for evaluating topic models
        - Handles input data as pandas DataFrames

    Usage:
        The UCI metric can be used to evaluate topic modeling results:

        ```python
        from framework3.plugins.metrics.coherence import UCI
        from framework3.base.base_types import XYData
        import pandas as pd
        import numpy as np

        # Assuming you have a DataFrame 'df' with your document-term matrix
        x_data = XYData(value=df)
        y_pred = np.array([['word1', 'word2', 'word3'], ['word4', 'word5', 'word6']])  # Example topics

        uci_metric = UCI()
        score = uci_metric.evaluate(x_data, None, y_pred, f_vocab=df.columns)
        print(f"UCI Score: {score}")
        ```

    Methods:
        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData, **kwargs) -> Float | np.ndarray:
            Calculate the UCI coherence score.

    Note:
        This metric requires the input data to be a pandas DataFrame. Ensure that your data
        is properly formatted before using this metric.
    """

    def evaluate(
        self,
        x_data: XYData,
        y_true: XYData | None,
        y_pred: XYData,
        **kwargs: Unpack[CoherenceEvaluateKwargs],
    ) -> Float | np.ndarray:
        """
        Calculate the UCI coherence score.

        This method computes the UCI coherence score for the given topics.

        Args:
            x_data (XYData): The input data, expected to be a pandas DataFrame.
            y_true (XYData | None): Not used for this metric, but required by the interface.
            y_pred (XYData): The predicted topics, typically a list of lists of words.
            **kwargs (Unpack[EvaluateKwargs]): Additional keyword arguments:
                - f_vocab (list): The vocabulary of the corpus.
                - topk (int): The number of top words to consider for each topic (default: 10).
                - processes (int): The number of processes to use for parallel computation (default: 1).

        Returns:
            Float | np.ndarray: The UCI coherence score.

        Raises:
            Exception: If x_data is not a pandas DataFrame.

        Note:
            This method uses the Coherence class from framework3.plugins.metrics.utils.coherence internally.
        """
        f_vocab = kwargs.get("f_vocab")
        topk = cast(int, kwargs.get("topk", 10))
        processes = cast(int, kwargs.get("processes", 1))
        coherence = Coherence(
            f_vocab=f_vocab, topk=topk, processes=processes, measure="c_uci"
        )
        if isinstance(x_data.value, pd.DataFrame):
            return coherence.evaluate(df=x_data.value, predicted=y_pred)
        else:
            raise Exception("x_data must be a pandas DataFrame")
