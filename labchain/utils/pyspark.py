from pyspark.sql import SparkSession
from typing import Callable, Any, cast

from labchain.base.base_map_reduce import MapReduceStrategy


class PySparkMapReduce(MapReduceStrategy):
    """
    A MapReduce strategy implementation using PySpark for distributed computing.

    This class provides methods to perform map and reduce operations on large datasets
    using Apache Spark's distributed computing capabilities.

    Key Features:
        - Initializes a Spark session with configurable parameters
        - Supports map, flatMap, and reduce operations
        - Allows for parallel processing of data across multiple workers
        - Provides a method to stop the Spark context when processing is complete

    Usage:
        ```python
        from framework3.utils.pyspark import PySparkMapReduce

        # Initialize the PySparkMapReduce
        spark_mr = PySparkMapReduce(app_name="MySparkApp", master="local[*]", num_workers=4)

        # Perform map operation
        data = [1, 2, 3, 4, 5]
        mapped_data = spark_mr.map(data, lambda x: x * 2)

        # Perform reduce operation
        result = spark_mr.reduce(lambda x, y: x + y)
        print(result)  # Output: 30

        # Stop the Spark context
        spark_mr.stop()
        ```

    Attributes:
        sc (pyspark.SparkContext): The Spark context used for distributed computing.

    Methods:
        map(data: Any, map_function: Callable[..., Any], numSlices: int | None = None) -> Any:
            Applies a map function to the input data in parallel.
        flatMap(data: Any, map_function: Callable[..., Any], numSlices: int | None = None) -> Any:
            Applies a flatMap function to the input data in parallel.
        reduce(reduce_function: Callable[..., Any]) -> Any:
            Applies a reduce function to the mapped data.
        stop() -> None:
            Stops the Spark context.
    """

    def __init__(self, app_name: str, master: str = "local", num_workers: int = 4):
        """
        Initialize the PySparkMapReduce with a Spark session.

        Args:
            app_name (str): The name of the Spark application.
            master (str, optional): The Spark master URL. Defaults to "local".
            num_workers (int, optional): The number of worker instances. Defaults to 4.
        """
        builder: SparkSession.Builder = cast(SparkSession.Builder, SparkSession.builder)
        spark: SparkSession = (
            builder.appName(app_name)
            .config("spark.master", master)
            .config("spark.executor.instances", str(num_workers))
            .config("spark.cores.max", str(num_workers * 2))
            .getOrCreate()
        )

        self.sc = spark.sparkContext

    def map(
        self, data: Any, map_function: Callable[..., Any], numSlices: int | None = None
    ) -> Any:
        """
        Apply a map function to the input data in parallel.

        Args:
            data (Any): The input data to be processed.
            map_function (Callable[..., Any]): The function to apply to each element of the data.
            numSlices (int | None, optional): The number of partitions to create. Defaults to None.

        Returns:
            Any: The result of the map operation as a PySpark RDD.
        """
        self.rdd = self.sc.parallelize(data, numSlices=numSlices)
        self.mapped_rdd = self.rdd.map(map_function)

        # Aplicar transformaciones map
        return self.mapped_rdd

    def flatMap(
        self, data: Any, map_function: Callable[..., Any], numSlices: int | None = None
    ) -> Any:
        """
        Apply a flatMap function to the input data in parallel.

        Args:
            data (Any): The input data to be processed.
            map_function (Callable[..., Any]): The function to apply to each element of the data.
            numSlices (int | None, optional): The number of partitions to create. Defaults to None.

        Returns:
            Any: The result of the flatMap operation as a PySpark RDD.
        """
        self.rdd = self.sc.parallelize(data, numSlices=numSlices)
        self.mapped_rdd = self.rdd.flatMap(map_function)

        # Aplicar transformaciones map
        return self.mapped_rdd

    def reduce(self, reduce_function: Callable[..., Any]) -> Any:
        """
        Apply a reduce function to the mapped data.

        Args:
            reduce_function (Callable[..., Any]): The function to reduce the mapped data.

        Returns:
            Any: The result of the reduce operation.
        """
        result = self.mapped_rdd.reduce(reduce_function)
        return result

    def stop(self) -> None:
        """
        Stop the Spark context.

        This method should be called when you're done with Spark operations to release resources.
        """
        self.sc.stop()
