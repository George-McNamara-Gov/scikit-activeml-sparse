import numpy as np
import os
import time
import sys

from pathlib import Path
from types import NoneType
from ._validation import check_type, check_scalar

successful_skorch_torch_import = False
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    successful_skorch_torch_import = True
except ImportError:  # pragma: no cover
    pass


if successful_skorch_torch_import:

    def _as_numpy(x):
        """
        Best‑effort conversion of *x* to a NumPy array.

        * **torch.Tensor** – returns ``x.cpu().numpy()``
        * **numpy.ndarray** – returned unchanged
        * anything else – wrapped via ``np.asarray``

        Parameters
        ----------
        x
            Object to convert.

        Returns
        -------
        numpy.ndarray
            NumPy representation of *x*.
        """
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x)

    def _concat_numpy(chunks):
        """
        Concatenate a list of NumPy arrays along the first axis.  If shapes are
        incompatible, return an *object‑dtype* array of the chunks.

        Parameters
        ----------
        chunks : list of numpy.ndarray
            Pieces to concatenate.

        Returns
        -------
        numpy.ndarray
            Concatenated result or object‑dtype fallback.
        """
        try:
            return np.concatenate(chunks, axis=0)
        except ValueError:
            out = np.empty(len(chunks), dtype=object)
            out[:] = [np.asarray(c) for c in chunks]
            return out

    class _WrappedDataset(Dataset):
        """
        Lightweight adapter that turns a *column-oriented* Hugging Face
        ``datasets.Dataset`` (or any mapping-like dataset) into the
        ``(sample, label)`` tuple interface expected by a PyTorch
        ``DataLoader``.

        Parameters
        ----------
        dataset : Dataset
            Source dataset whose ``__getitem__`` returns a *dict*-like row.
        sample_key : str
            Column name that contains the **input sample** (e.g. `"text"`,
            `"audio"`, `"image"`).
        label_key : str
            Column name that contains the **target label**.

        Notes
        -----
        The wrapper does **not** copy data; it simply delegates look-ups to
        the underlying dataset and pulls out the two requested fields.

        Examples
        --------
        >>> from datasets import load_dataset
        >>> ds_hf = load_dataset("ag_news", split="train")       # dict rows
        >>> ds = _WrappedDataset(ds_hf, sample_key="text", label_key="label")
        >>> sample, label = ds[0]
        >>> print(sample[:60] + "...", label)
        """

        def __init__(self, dataset, sample_key, label_key):
            super().__init__()
            self.dataset = dataset
            self.sample_key = sample_key
            self.label_key = label_key

        def __len__(self):
            """
            Return the number of examples in the wrapped dataset.

            Returns
            -------
            int
                ``len(self.dataset)``.
            """
            return len(self.dataset)

        def __getitem__(self, idx):
            """
            Fetch *idx*-th example as a ``(sample, label)`` tuple.

            Parameters
            ----------
            idx : int
                Sample index.

            Returns
            -------
            tuple
                ``(row[sample_key], row[label_key])`` where *row* is the dict
                produced by the underlying dataset.
            """
            row = self.dataset[idx]
            return row[self.sample_key], row[self.label_key]

    def cache_numpy_dataset(
        dataset=None,
        dataset_name=None,
        model_fn=None,
        model_name=None,
        batch_size=32,
        device="cpu",
        cache_dir=".",
        num_workers=None,
        pin_memory=None,
        overwrite=False,
        sample_label_keys=None,
        verbose=True,
    ):
        """
        Compute or load cached data for a (*dataset*, *model*) pair.

        Two modes
        ---------
        1. **Embedding mode** – *model_fn* is a callable whose forward pass returns
           embeddings for a batch of samples.
        2. **Raw‑sample mode** – *model_fn* is ``None``; the samples themselves are
           converted to NumPy (when possible) and cached.

        Parameters
        ----------
        dataset : Dataset, default=None
            A PyTorch‑style dataset instance whose (embedded) samples and labels
            are saved as numpy arrays. If `dataset=None`, the dataset must have
            been cached before and `dataset_name` must be given. If this caching
            was performed using a model, the `model_name` must be also provided.
        dataset_name : str, default=None
            Identifier for the dataset.  Defaults to
            ``"<ClassName>-<len(dataset)>"``.
        model_fn : nn.Module, default=None
            Embedding model.  Pass ``None`` to activate raw‑sample mode.
        model_name : str, default=None
            Identifier used in the cache filename.  Ignored in raw‑sample mode.
            Defaults to ``model_fn.__class__.__name__`` when not given.
        batch_size : int, default=32
            Batch size for the ``DataLoader``.
        device : str, default="cpu"
            Device string for model inference (``"cuda:0"``, ``"mps"``, …).
        cache_dir : str or pathlib.Path, default="."
            Directory in which ``.npz`` cache files are stored.
        num_workers : int, default=None
            Number of worker processes for the ``DataLoader``.  Defaults to
            half the available CPU cores.
        pin_memory : bool, default=None
            Whether to pin memory in the ``DataLoader``. Defaults to
            ``device.startswith("cuda")``.
        overwrite : bool, default=False
            If ``True`` the cache is ignored and data are recomputed.
        sample_label_keys : tuple, default=None
            Column name that contains the *input* sample when *dataset* is a dict-style
            Hugging Face dataset.  If ``None`` the function tries the first non-label
            field.
        verbose : bool, default=True
            If ``True``, the progress bar will be displayed.


        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Tuple ``(X, y)`` containing either the embeddings or raw samples, and
            the labels.

        Usage examples
        --------------
        >>> from torchvision import datasets
        >>> ds = datasets.CIFAR10("data", train=False, download=True)
        >>> # --- embedding mode --------------------------------------------------
        >>> import torch
        >>> dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        >>> X, y = cache_numpy_dataset(
        ...     dataset=ds,
        ...     model_fn=dinov2,
        ...     model_name="dinov2_vitb14",
        ...     batch_size=128,
        ...     device="cuda:0"
        ... )
        >>> # --- raw‑sample mode -------------------------------------------------
        >>> X_raw, y = cache_numpy_dataset(dataset=ds, model_fn=None, batch_size=128)

        Notes
        -----
        * The cache file also stores a ``created`` timestamp for inspection.
        """
        # Check input parameters.
        check_type(dataset_name, "dataset_name", str, NoneType)
        check_type(model_name, "model_name", str, NoneType)
        check_type(device, "device", str)
        check_type(overwrite, "overwrite", bool)
        check_type(verbose, "verbose", bool)
        check_type(pin_memory, "pin_memory", bool, NoneType)
        check_type(sample_label_keys, "sample_label_keys", tuple, NoneType)
        check_scalar(batch_size, "batch_size", min_val=1, target_type=int)
        if num_workers is not None:
            check_scalar(
                num_workers, "num_workers", min_val=0, target_type=int
            )

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        if dataset_name is None:
            dataset_name = f"{dataset.__class__.__name__}-{len(dataset)}"

        if model_fn is None:
            model_tag = "raw"  # raw‑sample mode
        else:
            model_tag = model_name or model_fn.__class__.__name__

        cache_file = cache_dir / f"{dataset_name}-{model_tag}.npz"

        if cache_file.exists() and not overwrite:
            if verbose:
                print(
                    f"[cache] Load numpy arrays in {cache_file}.",
                    file=sys.stderr,
                    flush=True,
                )
            with np.load(cache_file, allow_pickle=True) as npz:
                return npz["X"], npz["y"]

        if isinstance(sample_label_keys, tuple):
            dataset = _WrappedDataset(
                dataset=dataset,
                sample_key=sample_label_keys[0],
                label_key=sample_label_keys[1],
            )

        loader = DataLoader(
            dataset,
            # Use dict as input.
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers or os.cpu_count() // 2,
            pin_memory=(
                pin_memory
                if pin_memory is not None
                else device.startswith("cuda")
            ),
        )

        sample_chunks, label_chunks = [], []
        iterator = tqdm(loader, desc=dataset_name) if verbose else loader
        if model_fn is not None:  # embedding mode
            model = model_fn.to(device).eval()
            with torch.no_grad():
                for samples, labels in iterator:
                    sample_chunks.append(model(samples.to(device)).cpu())
                    label_chunks.append(labels.cpu())
            X = torch.cat(sample_chunks).numpy()
        else:  # raw‑sample mode
            for samples, labels in iterator:
                sample_chunks.append(_as_numpy(samples))
                label_chunks.append(_as_numpy(labels))
            X = _concat_numpy(sample_chunks)

        y = _concat_numpy(label_chunks)

        np.savez_compressed(cache_file, X=X, y=y, created=time.time())
        return X, y
