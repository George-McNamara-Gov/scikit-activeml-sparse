import numpy as np

from abc import ABC, abstractmethod

from ...utils import (
    MISSING_LABEL,
    is_labeled,
    check_classes,
)

successful_skorch_torch_import = False
try:
    import torch

    from skorch import NeuralNet
    from skorch.utils import to_numpy

    from torch import nn
    from torch.nn import CrossEntropyLoss
    from torch.nn import functional as F
    from torch.utils.data import default_collate

    from ...classifier import SkorchClassifier

    successful_skorch_torch_import = True
except ImportError:
    pass  # pragma: no cover

if successful_skorch_torch_import:

    class _MultiAnnotatorClassifier(SkorchClassifier, ABC):

        def __init__(
            self,
            multi_annotator_module,
            criterion,
            clf_module,
            n_annotators=None,
            neural_net_param_dict=None,
            sample_dtype=None,
            classes=None,
            cost_matrix=None,
            missing_label=MISSING_LABEL,
            random_state=None,
        ):
            super(_MultiAnnotatorClassifier, self).__init__(
                module=multi_annotator_module,
                criterion=criterion,
                classes=classes,
                missing_label=missing_label,
                cost_matrix=cost_matrix,
                random_state=random_state,
                neural_net_param_dict=neural_net_param_dict,
                sample_dtype=sample_dtype,
            )
            self.clf_module = clf_module
            self.n_annotators = n_annotators

        def _net_parts(self, X, y):
            # Check module parameters.
            if self.neural_net_param_dict is None:
                neural_net_param_dict = {}
                clf_module_param_dict = {}
            else:
                neural_net_param_dict = self.neural_net_param_dict.copy()
                prefix = "module__"
                clf_module_param_dict = {
                    k[len(prefix) :]: neural_net_param_dict.pop(k)
                    for k in list(neural_net_param_dict)
                    if k.startswith(prefix)
                }

            # Check `classes` parameter.
            if not hasattr(self, "classes_") and self.classes is not None:
                self.classes_ = check_classes(self.classes)
            if not hasattr(self, "classes_") and self.classes is None:
                raise RuntimeError(
                    "Number of classes must be known before init."
                )

            # Check `n_annotators` parameter.
            if self.n_annotators is None and y is None:
                raise ValueError("Provide n_annotators or pass y at init.")
            if (
                self.n_annotators is not None
                and isinstance(y, np.ndarray)
                and self.n_annotators != y.shape[1]
            ):
                raise ValueError("n_annotators mismatch.")
            self.n_annotators_ = (
                self.n_annotators
                if self.n_annotators is not None
                else y.shape[1]
            )

            # Build `neural_net_param_dict`.
            invariant = {
                "module__clf_module": self.clf_module,
                "module__clf_module_param_dict": clf_module_param_dict,
            }
            override = self._build_neural_net_param_overrides(X=X, y=y)
            if not isinstance(override, dict):
                raise TypeError(
                    "`build_neural_net_param_overrides` must return a `dict`."
                )
            illegal = invariant.keys() & override.keys()
            if illegal:
                raise ValueError(
                    f"Do not set these in overrides: {sorted(illegal)}"
                )
            for k, v in (invariant | override).items():
                if (
                    k in neural_net_param_dict
                    and neural_net_param_dict[k] != v
                ):
                    raise ValueError(
                        f"`neural_net_param_dict[{k!r}]` must be {v!r} "
                        f"or unset."
                    )
                neural_net_param_dict[k] = v

            return self.module, self.criterion, neural_net_param_dict

        def _return_labeled_data(self, X, y):
            """
            Return only labeled samples.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Input samples.
            y : array-like of shape (n_samples, ...)
                Targets with unlabeled entries following the subclass'
                convention.

            Returns
            -------
            X_lbld : ndarray or None
                Labeled inputs or ``None`` if none exist.
            y_lbld : ndarray or None
                Corresponding labeled targets or ``None`` if none exist.
            """
            X_lbld, y_lbld = None, None
            is_lbld = is_labeled(y, missing_label=-1).any(axis=1)
            if np.sum(is_lbld) > 0:
                net = self.neural_net_.module_
                net.set_forward_return()
                X_lbld = X[is_lbld]
                y_lbld = y[is_lbld].astype(np.int64)
            return X_lbld, y_lbld

        def _validate_data_kwargs(self):
            """
            Return kwargs forwarded to ``_validate_data``.

            Returns
            -------
            kwargs : dict or None
                Keyword arguments consumed by ``_validate_data``.
            """
            vd_kwargs = super()._validate_data_kwargs()
            vd_kwargs["y_ensure_1d"] = False
            return vd_kwargs

        @abstractmethod
        def _build_neural_net_param_overrides(self, X, y):
            """Subclasses must return the enforced skorch param overrides."""

    class _MultiAnnotatorClassificationModule(nn.Module):
        """
        Auxiliary module for Annot-Mix [1]_ that produces class logits and
        annotator-conditioned outputs, while training with MixUp [2]_.

        Parameters
        ----------
        clf_module : nn.Module or type
            Classifier backbone/head that maps `x -> logits_class` or
            `(logits_class, x_embed)`. If it returns only logits, `x_embed` is
            set to the input `x` (or to `None` if `x` is not an embedding).
        clf_module_param_dict : dict
            Keyword args for constructing `clf_module` if a class is passed.
        """

        def __init__(
            self,
            clf_module,
            clf_module_param_dict,
            default_forward_outputs,
            full_forward_outputs,
        ):
            super().__init__()
            self.clf_module = self._as_module(
                clf_module, clf_module_param_dict
            )
            self.default_forward_outputs = default_forward_outputs
            self.full_forward_outputs = full_forward_outputs
            self.set_forward_return()

        def set_forward_return(self, values=None):
            """
            Select tensors to return.

            Parameters
            ----------
            values : str or sequence of str
                Any subset of {"logits_class", "x_embed", "a_embed",
                "log_p_annotator_class", "p_perf"}.

            Returns
            -------
            self : _AnnotMixModule
                The module itself for chaining.

            Raises
            ------
            ValueError
                If an unknown name is requested.
            """
            if values is None:
                values = self.default_forward_outputs
            if isinstance(values, str):
                values = [values]
            unknown = set(values) - set(self.full_forward_outputs)
            if unknown:
                raise ValueError(f"Unknown forward return(s): {unknown}")
            self.forward_return = set(values)
            return self

        def clf_module_forward(self, x):
            """
            Parameters
            ----------
            x : torch.Tensor of shape (batch_size, ...)
                Input batch. Shape depends on `clf_module`.

            Returns
            -------
            out : torch.Tensor or tuple
                Given `set_forward_return`, tensors are appended in the order:
                `"logits_class"`, `"x_embed"`, `"a_embed"`,
                `"log_p_annotator_class"`, `"p_perf"`.

            Raises
            ------
            ValueError
                If AP outputs are requested but `a`/`combs` are missing or
                shapes mismatch.
            """
            cls_out = self.clf_module(x)
            if isinstance(cls_out, tuple):
                logits_class, x_embed = cls_out
            else:
                logits_class, x_embed = cls_out, x  # fallback
            return logits_class, x_embed

        @staticmethod
        def _as_module(maybe_cls_or_mod, kwargs):
            if isinstance(maybe_cls_or_mod, nn.Module):
                return maybe_cls_or_mod
            if isinstance(maybe_cls_or_mod, type):
                return maybe_cls_or_mod(**(kwargs or {}))
            raise TypeError(
                "Expected nn.Module instance or class for a submodule."
            )


    class _MultiAnnotatorCollate:
        """
        Collate that expands a batch into all (sample, annotator) pairs and
        optionally applies MixUp jointly to samples, annotators, and labels.

        Parameters
        ----------
        n_classes : int
            Number of classes (for one-hot encoding).
        a : torch.Tensor or array-like of shape (n_annotators, ...)\
                or (n_annotators,)
            Annotator representations/features. Will be converted to a CPU tensor
            once and reused across batches.
        alpha : float, default=1.0
            MixUp Beta(alpha, alpha) parameter. If <= 0, no MixUp is applied.
        missing_label : int or float, default=-1
            Value in `y` indicating an unlabeled sample. Rows whose sample label
            equals `missing_label` are excluded from the (sample, annotator)
            pairs. If set to `float('nan')` or `numpy.nan`, NaN labels are
            treated as missing.

        Notes
        -----
        - This collate runs on CPU (inside DataLoader workers). For maximum
          speed, keep heavy augmentations inside the model on GPU and use this
          collate only if you truly need CPU-side MixUp across
          `(sample, annotator)` pairs.
        - Labels are returned as one-hot vectors of length `n_classes`.
        """

        def __init__(
            self, missing_label=-1
        ):
            self.missing_label = missing_label

        def __call__(self, batch):
            # Basic collation (supports tensors/ndarrays/nested dicts of X, y).
            x = default_collate([b[0] for b in batch])
            y = default_collate([b[1] for b in batch])

            # Flatten labels to (n_samples * n_annotators,)
            n_samples, n_annotators = y.shape
            y = y.view(-1)

            # Sample indices: 0..B-1 repeated for each annotator.
            idx_s = torch.arange(
                n_samples, dtype=torch.long
            ).repeat_interleave(n_annotators)

            # Annotator indices: 0..A-1 repeated for each annotator.
            idx_a = torch.arange(n_annotators, dtype=torch.long).repeat(
                n_samples
            )

            # Mask out pairs whose sample is unlabeled.
            if isinstance(self.missing_label, float) and (
                self.missing_label != self.missing_label
            ):  # NaN
                mask = ~torch.isnan(y.to(torch.float32))
            else:
                mask = y != self.missing_label
            idx_s = idx_s[mask]
            idx_a = idx_a[mask]
            y = y[mask]

            # Return batches including sample and annotator indices.
            x_out = {"x": x, "input_ids": torch.column_stack((idx_s, idx_a))}
            return x_out, y