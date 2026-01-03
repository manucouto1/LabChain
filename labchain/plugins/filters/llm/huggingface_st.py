from labchain.base.base_types import XYData
from labchain.container.container import Container
from labchain.base.base_clases import BaseFilter, BasePlugin
from sentence_transformers import SentenceTransformer
import torch

__all__ = ["HuggingFaceSentenceTransformerPlugin"]


@Container.bind()
class HuggingFaceSentenceTransformerPlugin(BaseFilter, BasePlugin):
    """
    A plugin for generating sentence embeddings using Hugging Face's Sentence Transformers.

    This plugin integrates Sentence Transformers from Hugging Face into the framework3 ecosystem,
    allowing for easy generation of sentence embeddings within pipelines.

    Key Features:
        - Utilizes pre-trained Sentence Transformer models from Hugging Face
        - Supports custom model selection
        - Generates embeddings for input text data
        - Integrates seamlessly with framework3's BaseFilter interface

    Usage:
        The HuggingFaceSentenceTransformerPlugin can be used to generate embeddings for text data:

        ```python
        from framework3.plugins.filters.llm.huggingface_st import HuggingFaceSentenceTransformerPlugin
        from framework3.base.base_types import XYData

        # Create an instance of the plugin
        st_plugin = HuggingFaceSentenceTransformerPlugin(model_name="all-MiniLM-L6-v2")

        # Prepare input data
        input_texts = ["This is a sample sentence.", "Another example text."]
        x_data = XYData(_hash='input_data', _path='/tmp', _value=input_texts)

        # Generate embeddings
        embeddings = st_plugin.predict(x_data)
        print(embeddings.value)
        ```

    Attributes:
        model_name (str): The name of the Sentence Transformer model to use.
        _model (SentenceTransformer): The underlying Sentence Transformer model.

    Methods:
        fit(x: XYData, y: XYData | None) -> float | None:
            Placeholder method for compatibility with BaseFilter interface.
        predict(x: XYData) -> XYData:
            Generate embeddings for the input text data.

    Note:
        This plugin requires the `sentence-transformers` library to be installed.
        Ensure that you have the necessary dependencies installed in your environment.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize a new HuggingFaceSentenceTransformerPlugin instance.

        This constructor sets up the plugin with the specified Sentence Transformer model.

        Args:
            model_name (str): The name of the Sentence Transformer model to use.
                              Defaults to "all-MiniLM-L6-v2".

        Note:
            The specified model will be downloaded and loaded upon initialization.
            Ensure you have a stable internet connection and sufficient disk space.
        """
        super().__init__()
        self.model_name = model_name
        self._model = SentenceTransformer(self.model_name)

    def fit(self, x: XYData, y: XYData | None) -> float | None:
        """
        Placeholder method for compatibility with BaseFilter interface.

        This method is not implemented as Sentence Transformers typically don't require fitting.

        Args:
            x (XYData): The input features (not used).
            y (XYData | None): The target values (not used).

        Returns:
            float | None: Always returns None.

        Note:
            This method is included for API consistency but does not perform any operation.
        """
        ...

    def predict(self, x: XYData) -> XYData:
        """
        Generate embeddings for the input text data.

        This method uses the loaded Sentence Transformer model to create embeddings
        for the input text.

        Args:
            x (XYData): The input text data to generate embeddings for.

        Returns:
            XYData: The generated embeddings wrapped in an XYData object.

        Note:
            The input text should be in a format compatible with the Sentence Transformer model.
            The output embeddings are converted to a PyTorch tensor before being wrapped in XYData.
        """
        embeddings = self._model.encode(x.value)
        return XYData.mock(torch.tensor(embeddings))
