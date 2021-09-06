# Addition of a (custom) datasets

To add a dataset you should follow some rules. These rules exist to ensure the compatibility 
with the architecture of the toolbox as well as to ease the addition of your data.
You can find a comprehensive explanation of what you need to add directly inside the `dummy.py` 
module. Moreover, inside the same file, you can find a skeleton to quickly start your dataset 
implementation. 

In general, there are three basic simple possible cases of usage:
    
    a. Use tfds to add/use a new dataset
    b. Download the data directly from an online location (URL)
    c. Provide the dataset through the use of a local folder that contain all the data

A part from these three cases, what you need to do when adding a dataset through the use of a .
py file is to create a class that inherits from `datasets.dataset.AnomalyDetectionDataset`.

```python
from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset

class YourDataset(AnomalyDetectionDataset):
    #The body of your class
    pass
```

Basically, there are a set of properties coming from the `AnomalyDetectionDataset` base class 
and that should be fulfilled. To populate these properties, the user should implement the 
`configure` method.

The properties are the following: 

```python

@property
    def channels(self) -> int:
        """The last dimension of the elements in the dataset.
        e.g. 3 if the dataset is a dataset of RGB images or 1
        if they are grayscale."""
        return self._channels

    @property
    def anomalous_label(self) -> tf.Tensor:
        """Return the constant tensor used for anomalous label (1)."""
        return self._anomalous_label

    @property
    def normal_label(self) -> tf.Tensor:
        """Return the constant tensor used for normal data label (0)."""
        return self._normal_label

    @property
    def train_normal(self) -> tf.data.Dataset:
        """Subset of the training dataset: only positive."""
        return self._train_normal

    @property
    def train_anomalous(self) -> tf.data.Dataset:
        """Subset of the training dataset: only negative."""
        return self._train_anomalous

    @property
    def train(self) -> tf.data.Dataset:
        """The complete training dataset with both positive and negatives.
        The labels are always 2."""
        return self._train

    @property
    def test(self) -> tf.data.Dataset:
        """The complete test dataset with both positive and negatives.
        The labels are always 2."""
        return self._test

    @property
    def test_normal(self) -> tf.data.Dataset:
        """Subset of the test dataset: only positive."""
        return self._test_normal

    @property
    def test_anomalous(self) -> tf.data.Dataset:
        """Subset of the test dataset: only negative."""
        return self._test_anomalous

    @property
    def validation(self) -> tf.data.Dataset:
        """The complete test dataset with both positive and negatives.
        The labels are always 2."""
        return self._validation

    @property
    def validation_normal(self) -> tf.data.Dataset:
        """Subset of the validation dataset: only positive."""
        return self._validation_normal

    @property
    def validation_anomalous(self) -> tf.data.Dataset:
        """Subset of the validation dataset: only negative."""
        return self._validation_anomalous

```

The function to be used, i.e., the `configure` function has the following signature:

```python 
def configure(
        self,
        batch_size: int,
        new_size: Tuple[int, int],
        anomalous_label: Optional[int] = None,
        shuffle_buffer_size: int = 10000,
        cache: bool = True,
        drop_remainder: bool = True,
        output_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
    """Configure the dataset. This makes all the object properties valid (not None).
        Args:
            batch_size: The dataset batch size
            new_size: (H,W) of the input image.
            anomalous_label: If the raw dataset contains label, all the elements with
                             "anomalous_label" are converted to element of
                             self.anomalous_label class.
            shuffle_buffer_size: Buffer size used during the tf.data.Dataset.shuffle call.
            cache: If True, cache the dataset
            drop_remainder: If True, when the dataset size is not a multiple of the dataset size,
                            the last batch will be dropped.
            output_range: A Tuple (min, max) containing the output range to use for
                          the processed images.
        """
```

Let's see the different cases.

## Case a -- Use tfds

This is the case where you want to add a dataset directly using tfds. You can find the list of 
already available datasets here: https://www.tensorflow.
org/datasets/catalog/overview#image_classification .

In this case, your `__init__` function can load directly from tfds the desired dataset.

For example, for the _MNIST_ case:

```python 
    class MNIST(AnomalyDetectionDataset):
    """MNIST dataset, split to be used for anomaly detection.
    Note:
        The label 1 is for the ANOMALOUS class.
        The label 0 is for the NORMAL class.
    """

    def __init__(self):
        super().__init__()
        (self._train_raw, self._test_raw), info = tfds.load(
            "mnist",
            split=["train", "test"],
            as_supervised=True,
            with_info=True,
        )

        self._num_classes = info.features["label"].num_classes

```

Next, the user should implement the `configure` method. For example, for the _MNIST_ case, what 
happens is the following:

```python 
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

        # 60000 train images -> 6000 per class -> 600 per class in validation set
        # do not overlap wih train images -> 6000 - 600 per class in training set
        per_class_dataset = [
            self._train_raw.filter(lambda _, y: tf.equal(y, label))
            for label in tf.range(self._num_classes, dtype=tf.int64)
        ]

        validation_raw = per_class_dataset[0].take(600)
        train_raw = per_class_dataset[0].skip(600)
        for i in range(1, self._num_classes):
            validation_raw = validation_raw.concatenate(per_class_dataset[i].take(600))
            train_raw = train_raw.concatenate(per_class_dataset[i].skip(600))

        # Train-data
        self._train_anomalous = (
            train_raw.filter(is_anomalous)
            .map(lambda x, _: (x, self.anomalous_label))
            .apply(pipeline_train)
        )
        self._train_normal = (
            train_raw.filter(is_normal)
            .map(lambda x, _: (x, self.normal_label))
            .apply(pipeline_train)
        )
        self._train = train_raw.map(
            lambda x, label: (
                x,
                tf.cast(tf.equal(label, anomalous_label), tf.int32),
            )
        ).apply(pipeline_train)

        # Validation data
        self._validation_anomalous = (
            validation_raw.filter(is_anomalous)
            .map(lambda x, _: (x, self.anomalous_label))
            .apply(pipeline_test)
        )
        self._validation_normal = (
            validation_raw.filter(is_normal)
            .map(lambda x, _: (x, self.normal_label))
            .apply(pipeline_test)
        )
        self._validation = validation_raw.map(
            lambda x, label: (
                x,
                tf.cast(tf.equal(label, anomalous_label), tf.int32),
            )
        ).apply(pipeline_test)

        # Test-data
        self._test_anomalous = (
            self._test_raw.filter(is_anomalous)
            .map(lambda x, _: (x, self.anomalous_label))
            .apply(pipeline_test)
        )
        self._test_normal = (
            self._test_raw.filter(is_normal)
            .map(lambda x, _: (x, self.normal_label))
            .apply(pipeline_test)
        )

        # Complete dataset with positive and negatives
        def _to_binary(x, y):
            if tf.equal(y, anomalous_label):
                return (x, self.anomalous_label)
            return (x, self.normal_label)

        self._test = self._test_raw.map(_to_binary).apply(pipeline_test)
```


Finally, the _MNIST_ class/name should be imported and added to the `datasets.__init__.py` file:

## Case b -- Download some data given a URL

This case differs from the previous one because you would probably need to process the files 
in order to create the correct train/validation/test structure of the dataset. The URL would 
likely contain a list of images that need to be downloaded to a local location and sorted out. 
However, the general structure of the code and what the user is intended to do remain the same.

To note: in the following lines, we will present a sample case. What user will need may differ 
from the following. Moreover, the user could decide to implement functions or some other 
utilities as needed. However, the presented structure can be of some help for the user.

In general, in your dataset class `__init__` function, you should provide the url from where the 
data is going to be downloaded, the path where to store the downloaded data and possibly some 
mapping function to be used to elaborate the data, if needed. In addition, the user can provide 
any code to further process the data. 

The important thing is to complete the code by providing all the requested basic 
`AnomalyDetectionDataset` properties (the ones regarding the Dataset itself) as `tf.data.
Dataset` instances. To note: in the previously described "case a" this step was not necessary 
because the `tfds` module already provides `tf.data.Dataset` objects.

For example, for the _Surface Cracks_ case:

```python 
        def __init__(self, path: Path = Path("surface_cracks")):
        super().__init__()

        self._archive_url = (
            "https://"
            "md-datasets-cache-zipfiles-prod.s3.eu-west-1"
            ".amazonaws.com/5y9wdsg2zt-2.zip"
        )
        self._path = path

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

        glob_ext = "*.jpg"
        all_normal = glob(str(self._path / "Negative" / glob_ext))
        all_normal_train = all_normal[:10000]
        all_normal_test = all_normal[10000:15000]
        all_normal_validation = all_normal[15000:]

        all_anomalous = glob(str(self._path / "Positive" / glob_ext))
        all_anomalous_train = all_anomalous[:10000]
        all_anomalous_test = all_anomalous[10000:15000]
        all_anomalous_validation = all_anomalous[15000:]

        self._train_raw = (
            tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(all_normal_train))
            .map(_read_and_map_fn(self.normal_label))
            .concatenate(
                tf.data.Dataset.from_tensor_slices(
                    tf.convert_to_tensor(all_anomalous_train)
                ).map(_read_and_map_fn(self.anomalous_label))
            )
        )

        self._test_raw = (
            tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(all_normal_test))
            .map(_read_and_map_fn(self.normal_label))
            .concatenate(
                tf.data.Dataset.from_tensor_slices(
                    tf.convert_to_tensor(all_anomalous_test)
                ).map(_read_and_map_fn(self.anomalous_label))
            )
        )

        self._validation_raw = (
            tf.data.Dataset.from_tensor_slices(
                tf.convert_to_tensor(all_normal_validation)
            )
            .map(_read_and_map_fn(self.normal_label))
            .concatenate(
                tf.data.Dataset.from_tensor_slices(
                    tf.convert_to_tensor(all_anomalous_validation)
                ).map(_read_and_map_fn(self.anomalous_label))
            )
        )

        # RGB dataset
        self._channels = 3
```

Here, we have included a `_download_and_extract` function to actually fill the `self._path` path 
with the data downloaded by using the `self._archive_url` address:

```python 
    def _download_and_extract(self) -> None:
        """Download and extract the dataset."""

        if self._path.exists():
            print(self._path, " already exists. Skipping dataset download.")
            return
        self._path.mkdir()

        # Download a zip file
        print("Downloading dataset from: ", self._archive_url)
        request = requests.get(self._archive_url)
        print("Unzipping...")
        with zipfile.ZipFile(BytesIO(request.content)) as zip_archive:
            # The zip file contains a rar file :\
            print(
                "Unrarring... "
                "(this may take up to 15 minutes because python 'rarfile' is slow)"
            )
            rar_archive = zip_archive.read(
                "Concrete Crack Images for Classification.rar"
            )
            rar_path = self._path / "cracks.rar"
            with open(str(rar_path), "wb") as fp:
                fp.write(rar_archive)

            rar = rarfile.RarFile(str(rar_path))
            rar.extractall(str(self._path))
        print("Raw dataset downloaded and extracted.")
```

The `configure` method has been next fulfilled as the following:

```python 
    def configure(
        self,
        batch_size: int,
        new_size: Tuple[int, int],
        anomalous_label: Optional[int] = None,
        shuffle_buffer_size: int = 10000,
        cache: bool = True,
        drop_remainder: bool = True,
        output_range: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """Configure the dataset. This makes all the object properties valid (not None).
        Args:
            batch_size: The dataset batch size
            new_size: (H,W) of the input image.
            anomalous_label: If the raw dataset contains label, all the elements with
                             "anomalous_label" are converted to element of
                             self.anomalous_label class.
            shuffle_buffer_size: Buffer size used during the tf.data.Dataset.shuffle call.
            cache: If True, cache the dataset
            drop_remainder: If True, when the dataset size is not a multiple of the dataset size,
                            the last batch will be dropped.
            output_range: A Tuple (min, max) containing the output range to use
                          for the processed images.
        """

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

        # Train-data
        self._train_anomalous = (
            self._train_raw.filter(is_anomalous)
            .map(lambda x, _: (x, self.anomalous_label))
            .apply(pipeline_train)
        )
        self._train_normal = (
            self._train_raw.filter(is_normal)
            .map(lambda x, _: (x, self.normal_label))
            .apply(pipeline_train)
        )
        self._train = self._train_raw.apply(pipeline_train)

        # Test-data
        self._validation_anomalous = (
            self._validation_raw.filter(is_anomalous)
            .map(lambda x, _: (x, self.anomalous_label))
            .apply(pipeline_test)
        )
        self._validation_normal = (
            self._validation_raw.filter(is_normal)
            .map(lambda x, _: (x, self.normal_label))
            .apply(pipeline_test)
        )
        self._validation = self._validation_raw.apply(pipeline_test)

        # Test-data
        self._test_anomalous = (
            self._test_raw.filter(is_anomalous)
            .map(lambda x, _: (x, self.anomalous_label))
            .apply(pipeline_test)
        )
        self._test_normal = (
            self._test_raw.filter(is_normal)
            .map(lambda x, _: (x, self.normal_label))
            .apply(pipeline_test)
        )
        self._test = self._test_raw.apply(pipeline_test)
```

The last "Case c", is very similar to "Case b". 


