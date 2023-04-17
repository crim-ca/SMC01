"""A set of classes that encapsulate a Dataset that outputs DataFrames. The
encapsulation is used to change the number of row in the DataFranes of the dataset."""

import math
import torch
import pandas as pd


class PandasBatchIteratorDataset(torch.utils.data.IterableDataset):
    """Iterate over a dataset that provides dataframes and outputs a sequence
    of dataframes of a fixed size. Can either build bigger dataframes from a
    sequence of smaller dataframes, or make smaller dataframes out of a bigger
    one. The last dataframe of the iterator could be of a smaller size than the
    other ones. Usage::

        iterator_dataset = PandasBatchIteratorDataset(my_other_dataset, 1000)

        for i in iterator_dataset:
            # The dataframes in i should have a size of 1000 except the last one.

    Args:
        input_dataset: An iterable dataset which returns DataFrames.
        df_size: The size of the output dataframes."""

    def __init__(self, input_dataset, df_size, transform=None):
        self.input_dataset = input_dataset
        self.df_size = df_size
        self.transform = transform

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            input_dataset_it = iter(self.input_dataset)
        else:
            # Taken from:
            # https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
            per_worker = int(
                math.ceil(len(self.input_dataset) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.input_dataset))

            input_dataset_it = (
                self.input_dataset[i] for i in range(iter_start, iter_end)
            )

        return PandasBatchIterator(
            input_dataset_it, self.df_size, transform=self.transform
        )


class PandasBatchIterator:
    def __init__(self, source_iterator, batch_size, transform=None):
        self.source_iterator = source_iterator
        self.batch_size = batch_size
        self.transform = transform

        self.current_df = pd.DataFrame()

    def __iter__(self):
        return self

    def __next__(self):
        next_example = pd.DataFrame()

        while len(next_example.index) < self.batch_size:
            n_missing = self.batch_size - len(next_example.index)

            if len(self.current_df.index) > 0:
                # Current dataframe still has examples in it. Use them to build
                # the next example.
                to_grab = min(n_missing, len(self.current_df.index))
                next_example = pd.concat([next_example, self.current_df.head(to_grab)])

                # Remove used rows from the current file.
                self.current_df = self.current_df.tail(-to_grab)
            else:
                # No examples in the current dataframe. Try to get a new large
                # file from the source iterator.
                try:
                    self.current_df = next(self.source_iterator)
                except StopIteration:
                    if len(next_example.index) == 0:
                        # We do not have a partial example to send and there
                        # are no files left. Stop iteration.
                        raise StopIteration
                    else:
                        # We do have a partial batch to send. Return it.
                        break

        next_example.reset_index(drop=True, inplace=True)

        if self.transform:
            return self.transform(next_example)
        else:
            return next_example
