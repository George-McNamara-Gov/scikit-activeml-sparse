import math
import numpy as np
from sklearn.utils.validation import check_array

successful_skorch_torch_import = False
try:
    import torch

    from skorch import NeuralNet
    from skorch.utils import to_numpy

    from torch import nn
    from torch.nn import KLDivLoss
    from torch.nn import functional as F
    from torch.utils.data import default_collate

    from ._utils import (
        _MultiAnnotatorClassificationModule,
        _MultiAnnotatorClassifier,
    )

    successful_skorch_torch_import = True
except ImportError:
    pass  # pragma: no cover

from ...utils import (
    MISSING_LABEL,
    check_n_features,
    check_scalar,
)

if successful_skorch_torch_import:

    class AnnotatorMixClassifier(_MultiAnnotatorClassifier):
        """AnnotMixClassifier

        AnnotMix [1]_ rains a multi-annotator classifier using an extension of
        MixUp [2]_.

        Parameters
        ----------
        clf_module : nn.Module or nn.Module.__class__
            A PyTorch module as classification model outputting logits for
            samples as input. In general, the uninstantiated class should
            be passed, although instantiated modules will also work.
        n_annotators : int
            Number of annotators.
        neural_net_param_dict : dict, default=None
            Additional arguments for `skorch.net.NeuralNet`. If
            `neural_net_param_dict` is None, no additional arguments
             are added.
        sample_dtype : str or type, default=None
            The type or typecode all data is casted to. If `sample_dtype` is `None`,
            the datatype is preserved.
        classes : array-like of shape (n_classes,), default=None
            Holds the label for each class. If `None`, the classes are determined
            during the fit.
        missing_label : scalar or string or np.nan or None, default=np.nan
            Value to represent a missing label.
        cost_matrix : array-like of shape (n_classes, n_classes), default=None
            Cost matrix with `cost_matrix[i,j]` indicating cost of predicting
            class `classes[j]` for a sample of class `classes[i]`. Can be only
            set, if `classes` is not `None`.
        random_state : int or RandomState instance or None, default=None
            Determines random number for 'predict' method. Pass an int for
            reproducible results across multiple method calls.

        References
        ----------
        .. [1] Herde, M., Lührs, L., Huseljic, D., & Sick, B. (2024). Annot-Mix:
           Learning with Noisy Class Labels from Multiple Annotators via a
           Mixup Extension. Eur. Conf. Artif. Intell.
        .. [2] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018).
           mixup: Beyond Empirical Risk Minimization. Int. Conf. Learn.
           Represent.
        """

        def __init__(
            self,
            clf_module,
            alpha=1.0,
            sample_embed_dim=None,
            annotator_embed_dim=128,
            hidden_dim=None,
            n_hidden_layers=1,
            hidden_dropout=0.5,
            eta=0.9,
            n_annotators=None,
            neural_net_param_dict=None,
            sample_dtype=None,
            classes=None,
            cost_matrix=None,
            missing_label=MISSING_LABEL,
            random_state=None,
        ):
            super(AnnotatorMixClassifier, self).__init__(
                multi_annotator_module=_AnnotMixModule,
                clf_module=clf_module,
                criterion=KLDivLoss,
                classes=classes,
                missing_label=missing_label,
                cost_matrix=cost_matrix,
                random_state=random_state,
                neural_net_param_dict=neural_net_param_dict,
                sample_dtype=sample_dtype,
            )
            self.clf_module = clf_module
            self.alpha = alpha
            self.sample_embed_dim = sample_embed_dim
            self.annotator_embed_dim = annotator_embed_dim
            self.hidden_dim = hidden_dim
            self.n_hidden_layers = n_hidden_layers
            self.hidden_dropout = hidden_dropout
            self.eta = eta
            self.n_annotators = n_annotators

        def predict_proba(
            self,
            X,
            return_embeddings=False,
            return_logits=False,
            return_annotator_perf=False,
            return_annotator_class=False,
            return_annotator_embeddings=False,
        ):
            """Returns class-membership probability estimates for the test data
            `X`. Optionally, a tuple is returned whose elements appear
            **in this exact order** *if* they were requested:

            (0) `P_class` – always returned,
            (1) `X_embed` – if `return_embeddings`,
            (2) `P_perf`  – if `return_annotator_perf`,
            (3) `P_annot` – if `return_annotator_class`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Test samples.

            Returns
            -------
            P_class : numpy.ndarray of shape (n_samples, classes)
                `p_class[n, c]` is the probability, that instance `X[n]`
                belongs to the `classes_[c]`.
            X_embed : numpy.ndarray of shape (n_samples, ...)
                `X_embed[n]` refers to the learned embedding for sample `X[n]`.
                Only returned, if `return_embeddings=True`.
            P_perf : numpy.ndarray of shape (n_samples, n_annotators)
                `P_perf[n, m]` refers to the estimated correct probability
                (performance) of annotator `m` when labeling sample `X[n]`.
                Only returned, if `return_annotator_perf=True`.
            P_annot : numpy.ndarray of shape (n_samples, n_classes, n_annotators)
                `P_annot[n, c, m]` refers to the probability that annotator
                `m` provides the class label `c` for instance `X[n]`.
                Only returned, if `return_annotator_class=True`.
            """
            # Check input parameters.
            self._validate_data_kwargs()
            X = check_array(X, **self.check_X_dict_)
            check_n_features(
                self, X, reset=not hasattr(self, "n_features_in_")
            )
            check_scalar(
                return_embeddings, name="return_embeddings", target_type=bool
            )
            check_scalar(
                return_annotator_perf,
                name="return_annotator_perf",
                target_type=bool,
            )
            check_scalar(
                return_annotator_class,
                name="return_annotator_class",
                target_type=bool,
            )

            # Initialize module, if not done yet.
            if not hasattr(self, "neural_net_"):
                self.initialize()

            # Set forward options to obtain the different outputs required
            # by the input parameters.
            net = self.neural_net_.module_
            old_forward_return = net.forward_return
            forward_options = ["logits_class"]
            if return_embeddings:
                forward_options.append("x_embed")
            if return_annotator_perf:
                forward_options.append("log_p_annotator_perf")
            if return_annotator_class:
                forward_options.append("log_p_annotator_class")
            if return_annotator_embeddings:
                forward_options.append("a_embed")
            net.set_forward_return(forward_options)

            # Compute predictions for the different outputs required
            # by the input parameters.
            try:
                out_torch = self.neural_net_.forward(X)
                out_idx = 0
                if isinstance(out_torch, tuple):
                    P_class = to_numpy(out_torch[out_idx].softmax(dim=-1))
                    out_numpy = [P_class]
                    out_idx += 1
                else:
                    P_class = to_numpy(out_torch.softmax(dim=-1))
                    out_numpy = P_class
                if return_embeddings:
                    X_embed = to_numpy(out_torch[out_idx])
                    out_numpy.append(X_embed)
                    out_idx += 1
                if return_logits:
                    L_class = to_numpy(out_torch[0])
                    out_numpy.append(L_class)
                if return_annotator_perf:
                    P_annotator_perf = to_numpy(out_torch[out_idx].exp())
                    P_annotator_perf = P_annotator_perf.reshape(
                        -1, self.n_annotators_
                    )
                    out_numpy.append(P_annotator_perf)
                    out_idx += 1
                if return_annotator_class:
                    P_annotator_class = to_numpy(out_torch[out_idx].exp())
                    P_annotator_class = P_annotator_class.reshape(
                        -1, self.n_annotators_, len(self.classes_)
                    )
                    out_numpy.append(P_annotator_class)
                    out_idx += 1
                if return_annotator_class:
                    A_embed = to_numpy(
                        out_torch[out_idx][: self.n_annotators_]
                    )
                    out_numpy.append(A_embed)
                    out_idx += 1
            finally:
                net.set_forward_return(old_forward_return)

            # Initialize fallbacks if the classifier hasn't been fitted before.
            self._initialize_fallbacks(P=P_class)
            if isinstance(out_numpy, np.ndarray):
                return out_numpy
            else:
                return tuple(out_numpy)

        def _build_neural_net_param_overrides(self, X, y):
            """Initialize the internal `sklearn` wrapper from `skorch`."""
            # Check parameters specific to `AnnotMixClassifier`.
            check_scalar(
                self.alpha,
                name="alpha",
                target_type=float,
                min_val=0.0,
                min_inclusive=True,
            )
            if self.sample_embed_dim is None and (X is None or X.ndim > 2):
                raise ValueError(
                    "`sample_embed_dim` must be specified, "
                    "if no `X` is given or `X.ndim > 2`."
                )
            sample_embed_dim = self.sample_embed_dim or X.shape[-1]
            check_scalar(
                sample_embed_dim,
                name="sample_embed_dim",
                target_type=int,
                min_val=1,
                min_inclusive=True,
            )
            check_scalar(
                self.annotator_embed_dim,
                name="annotator_embed_dim",
                target_type=int,
                min_val=1,
                min_inclusive=True,
            )
            hidden_dim = self.hidden_dim or min(
                4 * len(self.classes_),
                max(
                    128,
                    2 * (self.annotator_embed_dim + sample_embed_dim),
                ),
            )
            check_scalar(
                hidden_dim,
                name="hidden_dim",
                target_type=int,
                min_val=1,
                min_inclusive=True,
            )
            check_scalar(
                self.n_hidden_layers,
                name="n_hidden_layers",
                target_type=int,
                min_val=1,
                min_inclusive=True,
            )
            check_scalar(
                self.hidden_dropout,
                name="hidden_dropout",
                target_type=float,
                min_val=0.0,
                min_inclusive=True,
                max_val=1.0,
                max_inclusive=False,
            )
            check_scalar(
                self.eta,
                name="eta",
                target_type=float,
                min_val=1 / len(self.classes_),
                min_inclusive=False,
                max_val=1.0,
                max_inclusive=False,
            )
            collate_fn = _MixUpCollate(
                n_classes=len(self.classes_),
                n_annotators=self.n_annotators_,
                alpha=self.alpha,
                missing_label=-1,
            )
            return {
                "criterion__reduction": "batchmean",
                "module__n_classes": len(self.classes_),
                "module__n_annotators": self.n_annotators_,
                "module__sample_embed_dim": sample_embed_dim,
                "module__annotator_embed_dim": self.annotator_embed_dim,
                "module__hidden_dim": self.hidden_dim,
                "module__n_hidden_layers": self.n_hidden_layers,
                "module__hidden_dropout": self.hidden_dropout,
                "module__eta": self.eta,
                "iterator_train__collate_fn": collate_fn,
                "predict_nonlinearity": None,
            }

    class _AnnotMixModule(_MultiAnnotatorClassificationModule):
        """
        Auxiliary module for Annot-Mix [1]_ that produces class logits and
        annotator-conditioned outputs, while training with MixUp [2]_.

        Parameters
        ----------
        n_classes : int
            Number of classes.
        clf_module : nn.Module or type
            Classifier backbone/head that maps `x -> logits_class` or
            `(logits_class, x_embed)`. If it returns only logits, `x_embed` is
            set to the input `x` (or to `None` if `x` is not an embedding).
        clf_module_param_dict : dict
            Keyword args for constructing `clf_module` if a class is passed.
        annotator_encoder : nn.Module or type or None, default=None
            Maps annotator features/IDs `a -> a_embed`. If ``None``, identity.
        annotator_encoder_param_dict : dict
            Keyword args for constructing `annotator_encoder` if a class is passed.
        sample_encoder : nn.Module or type or None, default=None
            Maps sample embeddings `x_embed -> x_embed_ap` used by the AP head.
            If ``None``, identity.
        sample_encoder_param_dict : dict
            Keyword args for constructing `sample_encoder` if a class is passed.
        annotator_confusion_head : nn.Module or type
            Maps concatenated `[x_embed_ap, a_embed] -> logits_conf` with shape
            `(batch_size, n_classes * n_classes)`.
        annotator_confusion_head_param_dict : dict
            Keyword args for constructing `annotator_confusion_head` if a class is
            passed.
        freeze_clf_for_ap : bool, default=True
            If True, detach `x_embed` before feeding the AP branch (no gradient
            from AP back into the classifier). Set False to train jointly.

        Notes
        -----
        - Return order is **fixed**: elements appear in the order
          `("logits_class", "x_embed", "a_embed", "log_p_annotator_class", "p_perf")`
          if requested via `set_forward_return`.
        - `log_p_annotator_class` has shape `(batch_size, n_classes)` and represents
          `log p(y_annot | x, a)`.
        - `log_p_annotator_perf` has shape `(batch_size,)` and equals
          `log sum_c p(y_true=c | x) * p(y_annot=c | y_true=c, a)`.

        References
        ----------
        .. [1] Herde, M., Lührs, L., Huseljic, D., & Sick, B. (2024). Annot-Mix:
           Learning with Noisy Class Labels from Multiple Annotators via a
           Mixup Extension. Eur. Conf. Artif. Intell.
        .. [2] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018).
           mixup: Beyond Empirical Risk Minimization. Int. Conf. Learn.
           Represent.
        """

        # Optional names that can be returned *after* logits_class
        OUTPUTS = (
            "logits_class",
            "x_embed",
            "a_embed",
            "log_p_annotator_class",
            "log_p_annotator_perf",
        )

        def __init__(
            self,
            n_classes,
            n_annotators,
            clf_module,
            clf_module_param_dict,
            sample_embed_dim,
            annotator_embed_dim,
            hidden_dim,
            n_hidden_layers,
            hidden_dropout,
            eta,
        ):
            super().__init__(
                clf_module=clf_module,
                clf_module_param_dict=clf_module_param_dict,
                default_forward_outputs="log_p_annotator_class",
                full_forward_outputs=[
                    "logits_class",
                    "x_embed",
                    "a_embed",
                    "log_p_annotator_class",
                    "log_p_annotator_perf",
                ],
            )
            # Define integer variables.
            self.n_classes = n_classes
            self.annotator_embed_dim = annotator_embed_dim

            # Set up layer to learn annotator embeddings.
            self.register_buffer(
                "a", torch.eye(n_annotators, dtype=torch.float32)
            )
            self.annotator_embed = nn.Linear(
                in_features=n_annotators,
                out_features=annotator_embed_dim,
            )

            # Post-scale diagonal bump as inductive bias.
            eta = math.log(eta / (1.0 - eta)) + math.log(n_classes - 1.0)
            prior_conf = nn.Parameter(
                eta * torch.eye(n_classes, dtype=torch.float32).flatten()
            )

            # Set up annotator confusion head.
            full_dim = sample_embed_dim + annotator_embed_dim
            blocks, dim = [], full_dim
            for _ in range(n_hidden_layers):
                blocks += [
                    nn.Dropout(hidden_dropout),
                    nn.Linear(dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),
                ]
                dim = hidden_dim
            out = nn.Linear(dim, n_classes * n_classes)
            out.bias = prior_conf
            blocks += [out]
            self.annotator_confusion_head = nn.Sequential(*blocks)

        def forward(self, x, a=None):
            """
            Parameters
            ----------
            x : torch.Tensor of shape (batch_size, ...)
                Input batch. Shape depends on `clf_module`.
            a : torch.Tensor or None
                Annotator features/IDs. Needed if any of {"a_embed",
                "log_p_annotator_class", "p_perf"} are requested. Shape
                `(n_annotators, ...)` if using `combs`, or `(batch_size, ...)`
                when predicting per-batch annotators without `combs`.

            Returns
            -------
            out : torch.Tensor or tuple
                Given `set_forward_return`, tensors are appended in the order:
                `"logits_class"`, `"x_embed"`, `"a_embed"`, `"log_p_annotator_class"`,
                `"p_perf"`.

            Raises
            ------
            ValueError
                If AP outputs are requested but `a`/`combs` are missing or
                shapes mismatch.
            """
            # Obtain classifier outputs.
            logits_class, x_embed = self.clf_module_forward(x)

            # Append classifier output if required.
            out = []
            if "logits_class" in self.forward_return:
                out.append(logits_class)
            if "x_embed" in self.forward_return:
                out.append(x_embed)

            need_annotator_output = any(
                k in self.forward_return
                for k in (
                    "a_embed",
                    "log_p_annotator_class",
                    "log_p_annotator_perf",
                )
            )
            if need_annotator_output:
                a = a if a is not None else self.a

                # Sample/annotator embeddings for annotator head.
                x_embed = x_embed.detach()
                a_embed = self.annotator_embed(a)

                # Generate pairs of samples and annotator if not done yet.
                if len(x_embed) != len(a_embed):
                    combs = torch.cartesian_prod(
                        torch.arange(len(x_embed), device=x_embed.device),
                        torch.arange(len(a_embed), device=a_embed.device),
                    )
                    x_embed = x_embed[combs[:, 0]]
                    a_embed_return = a_embed.clone().detach()
                    a_embed = a_embed[combs[:, 1]]
                    logits_class = logits_class[combs[:, 0]]
                else:
                    a_embed_return = a_embed

                # Compute confusion matrix logits per sample-annotator pair.
                logits_conf = self.annotator_confusion_head(
                    torch.cat([x_embed, a_embed], dim=-1)
                )
                logits_conf = logits_conf.view(
                    -1, self.n_classes, self.n_classes
                )

                # Compute log-probabilities for class and confusion matrices.
                p_conf_log = F.log_softmax(logits_conf, dim=-1)
                p_class_log = F.log_softmax(logits_class, dim=-1)

                # Compute and append annotator correctness log-probabilities.
                if "log_p_annotator_perf" in self.forward_return:
                    log_diag_conf = torch.diagonal(
                        p_conf_log, dim1=-2, dim2=-1
                    )
                    p_perf = torch.logsumexp(
                        p_class_log + log_diag_conf, dim=-1
                    )
                    out.append(p_perf)

                # Compute and append annotator class log-probabilities.
                if "log_p_annotator_class" in self.forward_return:
                    log_p_annotator_class = torch.logsumexp(
                        p_class_log[:, :, None] + p_conf_log, dim=1
                    )
                    out.append(log_p_annotator_class)

                if "a_embed" in self.forward_return:
                    out.append(a_embed_return)

            return out[0] if len(out) == 1 else tuple(out)

    class _MixUpCollate:
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
            self, n_classes, n_annotators, alpha=1.0, missing_label=-1
        ):
            self.n_classes = n_classes
            self.a = torch.eye(n_annotators, dtype=torch.float32)
            self.alpha = float(alpha)
            self.missing_label = missing_label

        def __call__(self, batch):
            # 1) Basic collation (supports tensors/ndarrays/nested dicts of X, y)
            x = default_collate([b[0] for b in batch])
            y = default_collate([b[1] for b in batch])

            # Flatten labels to (n_samples * n_annotators,)
            y = y.view(-1)

            # 2) Build all (sample, annotator) combinations
            n_samples = x.shape[0]
            n_annotators = self.a.shape[0]

            # sample indices: 0..B-1 repeated for each annotator
            idx_s = torch.arange(
                n_samples, dtype=torch.long
            ).repeat_interleave(n_annotators)
            # annotator indices: 0..A-1 tiled B times
            idx_a = torch.arange(n_annotators, dtype=torch.long).repeat(
                n_samples
            )
            # mask out pairs whose sample is unlabeled
            if isinstance(self.missing_label, float) and (
                self.missing_label != self.missing_label
            ):  # NaN
                mask = ~torch.isnan(y.to(torch.float32))
            else:
                mask = y != self.missing_label

            idx_s = idx_s[mask]
            idx_a = idx_a[mask]

            # 3) Select data per pair
            # x_pairs: (N_pairs, ...)  a_pairs: (N_pairs, ...)
            x_pairs = x.index_select(0, idx_s)
            a_pairs = self.a.index_select(0, idx_a)
            y_pairs = y[mask]  # integer class ids

            # One-hot labels
            y_oh = F.one_hot(y_pairs, num_classes=self.n_classes).to(
                dtype=torch.float32
            )

            # 4) Optional MixUp across pairs (jointly mixing x, a, and y)
            if self.alpha > 0:
                x_pairs, a_pairs, y_oh, _, _ = _mix_up(
                    x_pairs, a_pairs, y_oh, alpha=self.alpha
                )

            x_out = {"x": x_pairs, "a": a_pairs}
            return x_out, y_oh

    def _mix_up(*arrays, alpha=1.0, lmbda=None, permute_indices=None):
        """
        MixUp multiple arrays in lockstep using the same permutation and lambdas.

        Parameters
        ----------
        arrays : sequence of torch.Tensor
            Tensors with the same length `N` along the first dimension. Each will
            be mixed with the same permutation and mixing coefficients.
        alpha : float, default=1.0
            Beta(alpha, alpha) parameter. Used only if `lmbda is None`. If `alpha <= 0`,
            returns inputs unchanged along with generated permutation and lambda (1s).
        lmbda : torch.Tensor of shape (N,), optional
            Precomputed mixing coefficients in [0, 1]. If not provided, sampled
            from `Beta(alpha, alpha)` on the same device as the first array.
        permute_indices : torch.Tensor of shape (N,), optional
            Precomputed permutation indices. If not provided, a random permutation
            is generated on the same device as the first array.

        Returns
        -------
        outputs : tuple
            Tuple of mixed tensors in the same order as `arrays`, followed by
            `(lmbda, permute_indices)`.

        References
        ----------
        Zhang, H., Cissé, M., Dauphin, Y. N., & Lopez-Paz, D. (2018).
        mixup: Beyond Empirical Risk Minimization. ICLR.
        """
        if len(arrays) == 0:
            raise ValueError("At least one array must be provided to _mixup.")

        # All arrays must share the same leading dimension
        N = arrays[0].shape[0]
        for arr in arrays[1:]:
            if arr.shape[0] != N:
                raise ValueError(
                    "All arrays must have the same length in dim 0."
                )

        first = arrays[0]
        device = first.device

        if lmbda is None:
            if alpha > 0:
                lmbda = (
                    torch.distributions.Beta(alpha, alpha)
                    .sample((N,))
                    .to(device)
                )
            else:
                lmbda = torch.ones(N, device=device)
        else:
            lmbda = torch.as_tensor(lmbda, device=device, dtype=first.dtype)

        if permute_indices is None:
            permute_indices = torch.randperm(N, device=device)
        else:
            permute_indices = torch.as_tensor(
                permute_indices, device=device, dtype=torch.long
            )

        # Broadcast lmbda to array shapes and mix
        outputs = []
        for arr in arrays:
            # shape: (N, 1, 1, ...) to broadcast to arr
            view_shape = (N,) + (1,) * (arr.dim() - 1)
            lam_view = lmbda.view(view_shape)
            outputs.append(
                lam_view * arr
                + (1.0 - lam_view) * arr.index_select(0, permute_indices)
            )
        outputs.extend([lmbda, permute_indices])
        return tuple(outputs)
