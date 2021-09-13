"""Dummy dataset."""

from functools import partial
from glob import glob
from typing import Tuple, Union
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset


class Dummy(AnomalyDetectionDataset):
    """Dummy dataset.

    The class is an hint on what to do when using or providing a
    dataset that is not already shipped with the code. In any case, it is suggested to have the
    dataset in a tf.data.Dataset data format.

    There are three basic simple possible cases of usage:
    a. Use tfds to download a dataset
    b. Download the data directly from an online location
    c. Provide the dataset through the use of a local folder that contain all the data

    This Dummy class is set up to be used in case "a" and "b". Case "c" should not differ too
    much from these two cases.

    Note:
        The following is a good rule to follow:
            - The label 1 is for the ANOMALOUS class.
            - The label 0 is for the NORMAL class.
    """

    def __init__(self, path: Path = Path("surface_cracks")):
        """
        Args:
            path: Provide a location where to download the data. The string should be provided
            when in case "b". Otherwise, the user can remove this argument.
        """
        super().__init__()

        # ## The dataset can comes directly from tfds (case "a") ## #
        (self._train_raw, self._test_raw), info = tfds.load(
            "DATASET-NAME-HERE-FOR-EXAMPLE-mnist",
            split=["train", "test"],
            as_supervised=True,
            with_info=True,
        )

        self._num_classes = info.features["label"].num_classes

        # ## Download the data and convert it into a Dataset structure (case "b) ## #

        # Provide the url from where to download the data
        self._archive_url = (
            "https://"
            "md-datasets-cache-zipfiles-prod.s3.eu-west-1"
            ".amazonaws.com/5y9wdsg2zt-2.zip"
        )

        # Store the path to be used next where to save the downloaded data
        self._path = path

        # Call the function to download and extract the data, the implementation of this
        # function should be provided by the user.
        self._download_and_extract()

        def _read_and_map_fn(label):
            """Closure used in tf.data.Dataset.map for creating
            the correct pair (image, label).
            """

            def fn(filename):
                binary = tf.io.read_file(filename)
                image = tf.image.decode_jpeg(binary)
                return image, label

            return fn

        # Provide here all the necessary remaining code to get from the self._path all the needed
        # dataset images. Note that self._path has been filled using self._download_and_extract()
        # function call above. What you should do is something as the following:

        glob_ext = "*.jpg"
        all_normal = glob(str(self._path / "Negative" / glob_ext))
        all_normal_train = all_normal[:10000]

        # ...

        all_anomalous = glob(str(self._path / "Positive" / glob_ext))
        all_anomalous_train = all_anomalous[:10000]

        # ... get a peak to surface_cracks.py to get a complete example.

        # Now, to better manage next the data from inside the configure function you can fill
        # these private variable. Instead of None you should put the previous extracted data.
        self._train_raw = tf.data.Dataset(None)
        self._test_raw = tf.data.Dataset(None)
        self._validation_raw = tf.data.Dataset(None)
        self._channels = 3

    def _download_and_extract(self) -> None:
        """Download and extract the dataset."""

        # To be implemented here, the code to download from self._archive_url and to extract the
        # data into the self._path. This is the code for the case "b".
        print("Raw dataset downloaded and extracted.")

    def configure(
        self,
        batch_size: int,
        new_size: Tuple[int, int],
        anomalous_label: Union[int, str, None] = None,
        class_label: Union[int, str, None] = None,
        shuffle_buffer_size: int = 10000,
        cache: bool = True,
        drop_remainder: bool = True,
        output_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        """Configure the dataset. This makes all the object properties valid (not None).
        Args:
            batch_size: The dataset batch size
            new_size: (H,W) of the input image.
            anomalous_label: If the raw dataset contains labels, all the elements with
                             "anomalous_label" are converted to element of
                             self.anomalous_label class.
            class_label: If the raw dataset contains different classes (each one
                         containing both positive and negative samples) we can select
                         only one class to focus on (e.g. a dataset of industrial
                         defects on industrial objects composed of transistors and
                         pills and we are interested only in transistors and not on pills).
            shuffle_buffer_size: Buffer size used during the tf.data.Dataset.shuffle call.
            cache: If True, cache the dataset
            drop_remainder: If True, when the dataset size is not a multiple of the dataset size,
                            the last batch will be dropped.
            output_range: A Tuple (min, max) containing the output range to use for
                          the processed images.
        """

        # This part should exists in both "a" and "b" cases.

        pipeline = partial(
            self.pipeline,
            new_size=new_size,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            cache=cache,
            drop_remainder=drop_remainder,
            output_range=output_range,
        )

        pipeline_train = partial(pipeline, is_training=True)
        pipeline_test = partial(pipeline, is_training=False)
        is_anomalous = lambda _, label: tf.equal(label, anomalous_label)
        is_normal = lambda _, label: tf.not_equal(label, anomalous_label)

        # You should fill every one of this variable with the corresponding data
        # Train-data
        self._train_anomalous = None
        self._train_normal = None
        self._train = None

        # Validation data
        self._validation_anomalous = None
        self._validation_normal = None
        self._validation = None

        # Test-data
        self._test_anomalous = None
        self._test_normal = None
        self._test = None
