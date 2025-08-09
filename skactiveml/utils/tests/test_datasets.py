import unittest
import numpy as np
import os

from pathlib import Path

successful_skorch_torch_import = False
try:
    import torch
    import torch.multiprocessing as mp
    from torch import nn
    from torch.utils.data import Dataset
    from skactiveml.utils import cache_numpy_dataset

    successful_skorch_torch_import = True
except ImportError:
    pass  # pragma: no cover


if successful_skorch_torch_import:
    class TestDatasets(unittest.TestCase):

        def setUp(self):
            mp.set_start_method("spawn", force=True)

        def test_cache_numpy_dataset(self):
            for n_samples in [1, 10, 100]:

                with self.subTest(msg="raw-sample mode"):
                    raw_ds = TensorPairDataset(n_samples)
                    X_raw_1, y_raw_1 = cache_numpy_dataset(
                        dataset=raw_ds,
                        model_fn=None,
                        batch_size=4,
                        cache_dir=".",
                        verbose=False,
                        overwrite=True,
                    )
                    np.testing.assert_array_equal(X_raw_1.shape, (n_samples, 1))
                    np.testing.assert_array_equal(
                        X_raw_1.squeeze(), np.arange(n_samples, dtype=np.float32)
                    )
                    np.testing.assert_array_equal(
                        y_raw_1, np.arange(n_samples, dtype=np.int64)
                    )
                    X_raw_2, y_raw_2 = cache_numpy_dataset(
                        dataset=TensorPairDataset(
                            n_samples
                        ),  # fresh dataset instance
                        model_fn=None,
                        batch_size=4,
                        cache_dir=".",
                        verbose=False,
                    )
                    np.testing.assert_array_equal(X_raw_1, X_raw_2)
                    np.testing.assert_array_equal(y_raw_1, y_raw_2)

                with self.subTest(msg="embedding mode"):
                    model = CountingIdentity()
                    X_emb_1, y_emb_1 = cache_numpy_dataset(
                        dataset=raw_ds,
                        model_fn=model,
                        model_name="identity",
                        batch_size=4,
                        cache_dir=".",
                        verbose=True,
                        overwrite=True,
                    )
                    self.assertGreater(model.calls, 0)
                    np.testing.assert_array_equal(
                        X_emb_1.shape, (n_samples, 1)
                    )
                    np.testing.assert_array_equal(y_emb_1, np.arange(n_samples))
                    fresh_model = CountingIdentity()
                    X_emb_2, y_emb_2 = cache_numpy_dataset(
                        dataset=raw_ds,
                        model_fn=fresh_model,
                        model_name="identity",
                        batch_size=4,
                        cache_dir=".",
                        verbose=True,
                    )
                    self.assertEqual(fresh_model.calls, 0)  # loaded from cache
                    np.testing.assert_array_equal(X_emb_1, X_emb_2)
                    np.testing.assert_array_equal(y_emb_1, y_emb_2)

                with self.subTest(msg="hugginface-like mode"):
                    dict_ds = DictDataset(n_samples)
                    X_dict, y_dict = cache_numpy_dataset(
                        dataset=dict_ds,
                        model_fn=None,
                        sample_label_keys=("text", "label"),
                        batch_size=4,
                        cache_dir=".",
                        verbose=False,
                        overwrite=True,
                        num_workers=0,
                    )
                    np.testing.assert_array_equal(X_dict.shape, (n_samples, 1))
                    np.testing.assert_array_equal(y_dict, np.arange(n_samples))
                    self.assertRaises(
                        TypeError,
                        cache_numpy_dataset,
                        dataset=dict_ds,
                        model_fn=None,
                        sample_label_keys="not-a-tuple",
                        cache_dir=".",
                        verbose=False,
                    )

                with self.subTest(msg="list dataset"):
                    list_ds = ListPairDataset(n_samples)
                    X_list_1, y_list_1 = cache_numpy_dataset(
                        dataset=list_ds,
                        batch_size=4,
                        cache_dir=".",
                        overwrite=True,
                    )
                    np.testing.assert_array_equal(
                        X_list_1.shape, n_samples
                    )  # identity keeps original shape
                    np.testing.assert_array_equal(y_list_1, np.arange(n_samples))
                    X_list_2, y_list_2 = cache_numpy_dataset(
                        dataset_name=f"ListPairDataset-{n_samples}",
                        batch_size=4,
                        cache_dir=".",
                    )
                    np.testing.assert_array_equal(X_list_1, X_list_2)
                    np.testing.assert_array_equal(y_list_1, y_list_2)

                with self.subTest(msg="heterogeneous dataset"):
                    heterogeneous_ds = HeterogeneousPairDataset(n_samples)
                    X_heterogeneous_1, y_heterogeneous_1 = cache_numpy_dataset(
                        dataset=heterogeneous_ds,
                        batch_size=1,
                        cache_dir=".",
                        overwrite=True,
                    )
                    if n_samples > 1:
                        self.assertEqual(
                            X_heterogeneous_1.dtype, object
                        )  # identity keeps original shape
                    X_heterogeneous_2, y_heterogeneous_2 = cache_numpy_dataset(
                        dataset=heterogeneous_ds,
                        batch_size=1,
                        cache_dir=".",
                    )
                    for i in range(len(X_heterogeneous_2)):
                        np.testing.assert_array_equal(
                            X_heterogeneous_1[i], X_heterogeneous_2[i]
                        )

            with self.subTest(msg="check number of cached files"):
                # finally, ensure exactly two cache files exist (raw + identity)
                cached_files = [
                    p for p in Path(".").iterdir() if p.suffix == ".npz"
                ]
                self.assertEqual(len(cached_files), 15)
                self.assertTrue(all(os.path.getsize(p) > 0 for p in cached_files))


    class TensorPairDataset(Dataset):
        """Returns (sample, label) tensor pairs."""

        def __init__(self, n=10):
            self.X = torch.arange(n, dtype=torch.float32).unsqueeze(
                1
            )  # shape (n, 1)
            self.y = torch.arange(n, dtype=torch.long)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


    class HeterogeneousPairDataset(Dataset):
        """Returns (sample, label) pairs of list elements."""

        def __init__(self, n=5):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            tokens = list(range(idx + 1))
            return [tokens], idx


    class ListPairDataset(Dataset):
        """Returns (sample, label) pairs of list elements."""

        def __init__(self, n=10):
            self.X = ["i" * i for i in range(n)]  # shape (n, 1)
            self.y = list(range(n))

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


    class DictDataset(Dataset):
        """Returns dicts -> exercises _WrappedDataset."""

        def __init__(self, n=10):
            self.rows = [
                {"text": torch.tensor([i], dtype=torch.float32), "label": i}
                for i in range(n)
            ]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]


    class CountingIdentity(torch.nn.Module):
        """Identity model that counts forward calls."""

        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x
