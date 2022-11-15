# Copyright 2021 Zuru Tech HK Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MVTEC ANOMALY DETECTION DATASET

ABSTRACT
MVTec AD is a dataset for benchmarking anomaly detection methods with a
focus on industrial inspection. It contains over 5000 high-resolution images
divided into fifteen different object and texture categories.

Each category comprises a set of defect-free training images and a
test set of images with various kinds of defects as well as images without defects.

Pixel-precise annotations of all anomalies are also provided.
More information can be found in our corresponding paper
(https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_ad.pdf)
"""

import ftplib
import tarfile
from functools import partial
from glob import glob
from pathlib import Path
from typing import Callable, Tuple, Union

import tensorflow as tf

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset


class MVTecAD(AnomalyDetectionDataset):
    """The MVTec Anomaly Detection dataset comprises 15 categories with
    3629 images for training and validation and 1725 images for testing

    The training set contains only images without defects.
    The test set contains both: images containing various types of
    defects and defect-free images.

    The validation set is a subset of the training set with 1725/2 images.
    The test set is the remaining part of the set (no overlap between
    validation and test samples).
    """

    def __init__(self, path: Path = Path("mvtec_ad")):
        super().__init__()

        self._archive_user = "guest"
        self._archive_password = "GU.205dldo"
        self._archive_host = "ftp.softronics.ch"
        self._archive_path = "mvtec_anomaly_detection/"
        self._archive_filename = "mvtec_anomaly_detection.tar.xz"

        self._path = path

        self._download_and_extract()

        # RGB dataset
        self._channels = 3

    def _download_and_extract(self) -> None:
        """Download and extract the dataset."""

        if self._path.exists():
            print(self._path, " already exists. Skipping dataset download.")
            return
        self._path.mkdir()

        # Download a zip file
        print("Downloading dataset from ftp host: ", self._archive_host)
        ftp = ftplib.FTP(self._archive_host)
        ftp.login(self._archive_user, self._archive_password)
        ftp.cwd(self._archive_path)

        with open(self._archive_filename, "wb") as fp:
            ftp.retrbinary("RETR " + self._archive_filename, fp.write)
        ftp.quit()
        print("Unxz and untar (it may take a long time)...")
        with tarfile.open(self._archive_filename, mode="r:xz") as tar_archive:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar_archive, str(self._path))
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
        output_range: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """Configure the dataset. This makes all the object properties valid (not None).
        Args:
            batch_size: The dataset batch size
            new_size: (H,W) of the input image.
            anomalous_label: If the raw dataset contains label, all the elements with
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
            output_range: A Tuple (min, max) containing the output range to use
                          for the processed images.
        """

        if not class_label or anomalous_label:
            raise RuntimeError(
                "MVTec AD dataset requires the class_label to be selected, only."
            )

        def _read_and_map_fn(label: tf.Tensor) -> Callable:
            """Closure used in tf.data.Dataset.map for creating
            the correct pair (image, label).
            """

            def fn(filename) -> Tuple[tf.Tensor, tf.Tensor]:
                binary = tf.io.read_file(filename)
                image = tf.image.decode_png(binary)
                return image, label

            return fn

        # Train dataset of only positives
        train_pattern = self._path / class_label / "train" / "good" / "*.png"
        train_files = glob(str(train_pattern))
        if not train_files:
            raise RuntimeError(f"Empty training list for pattern {train_pattern}")

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

        self._train_normal = (
            tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(train_files))
            .map(lambda filename: _read_and_map_fn(self.normal_label)(filename))
            .apply(pipeline_train)
        )

        self._train = self._train_normal

        test_files_positive_all = glob(
            str(self._path / class_label / "test" / "good" / "*.png")
        )
        test_files_negative_all = list(
            set(glob(str(self._path / class_label / "test" / "**" / "*.png")))
            - set(test_files_positive_all)
        )

        test_files_negative = test_files_negative_all[
            : len(test_files_negative_all) // 2
        ]
        validation_files_negative = test_files_negative_all[
            len(test_files_negative_all) // 2 :
        ]

        test_files_positive = test_files_positive_all[
            : len(test_files_positive_all) // 2
        ]
        validation_files_positive = test_files_positive_all[
            len(test_files_positive_all) // 2 :
        ]

        self._test_normal = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(test_files_positive)
        ).map(lambda filename: _read_and_map_fn(self.normal_label)(filename))

        self._test_anomalous = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(test_files_negative)
        ).map(lambda filename: _read_and_map_fn(self.anomalous_label)(filename))

        self._validation_normal = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(validation_files_positive)
        ).map(lambda filename: _read_and_map_fn(self.normal_label)(filename))

        self._validation_anomalous = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(validation_files_negative)
        ).map(lambda filename: _read_and_map_fn(self.anomalous_label)(filename))

        self._validation = self._validation_normal.concatenate(
            self._validation_anomalous
        ).apply(pipeline_test)
        self._test = self._test_normal.concatenate(self._test_anomalous).apply(
            pipeline_test
        )

        self._validation_anomalous = self._validation_anomalous.apply(pipeline_test)
        self._validation_normal = self._validation_normal.apply(pipeline_test)

        self._test_anomalous = self._test_anomalous.apply(pipeline_test)
        self._test_normal = self._test_normal.apply(pipeline_test)
