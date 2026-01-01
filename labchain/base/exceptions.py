class NotTrainableFilterError(Exception):
    """
    Exception raised when attempting to fit a non-trainable filter.

    This exception is used to indicate that an attempt was made to train or fit
    a filter that is not designed to be trainable.

    Key Features:
        - Inherits from the built-in Exception class
        - Provides a clear error message for debugging and error handling

    Usage:
        This exception should be raised when a non-trainable filter's fit method
        is called or when any attempt is made to train a filter that doesn't
        support training.

    Example:
        ```python
        class NonTrainableFilter(BaseFilter):
            def fit(self, X, y=None):
                raise NotTrainableFilterError("This filter does not support training.")

            def transform(self, X):
                # transformation logic here
                pass
        ```

    Attributes:
        message (str): The error message passed when raising the exception.

    Note:
        When catching this exception, it's often useful to log the error or inform
        the user that the attempted operation is not supported for the given filter.
    """

    pass
