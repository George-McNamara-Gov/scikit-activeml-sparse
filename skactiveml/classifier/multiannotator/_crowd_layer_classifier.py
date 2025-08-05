from inspect import isclass

import numpy as np
from sklearn.utils.validation import check_array

successful_skorch_torch_import = False
try:
    import torch

    from skorch import NeuralNet
    from skorch.utils import to_numpy

    from torch import nn
    from torch.nn import CrossEntropyLoss
    from torch.nn import functional as F

    successful_skorch_torch_import = True
except ImportError:
    pass  # pragma: no cover

from ...base import AnnotatorModelMixin
from ...classifier import SkorchClassifier
from ...utils import (
    MISSING_LABEL,
    is_labeled,
    check_random_state,
    check_n_features,
    check_scalar,
)

if successful_skorch_torch_import:

    class CrowdLayerClassifier(SkorchClassifier, AnnotatorModelMixin):
        """Crowd Layer

        Crowd Layer [1]_ is a layer added at the end of a classifying neural
        network and allows us to train deep neural networks end-to-end,
        directly from the noisy labels of multiple annotators, using only
        backpropagation.

        Parameters
        ----------
        classification_module : nn.Module or nn.Module.__class__
            A PyTorch module as classification model outputting logits for
            samples as input. In general, the uninstantiated class should
            be passed, although instantiated modules will also work.
        n_annotators : int
            Number of annotators.
        neural_net_param_dict : dict, default=None
            Additional arguments for `skorch.net.NeuralNet`. If
            `neural_net_param_dict` is None, no additional arguments
             are added.
        X_dtype : str or type, default=None
            The type or typecode all data is casted to. If `X_dtype` is `None`,
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
        .. [1] Rodrigues, Filipe, and Francisco Pereira. "Deep Learning from
            Crowds." AAAI Conference on Artificial Intelligence, 2018.

        """

        def __init__(
            self,
            classification_module,
            n_annotators=None,
            neural_net_param_dict=None,
            X_dtype=None,
            classes=None,
            cost_matrix=None,
            missing_label=MISSING_LABEL,
            random_state=None,
        ):
            super(CrowdLayerClassifier, self).__init__(
                module=_CrowdLayerModule,
                criterion=CrossEntropyLoss,
                classes=classes,
                missing_label=missing_label,
                cost_matrix=cost_matrix,
                random_state=random_state,
                neural_net_param_dict=neural_net_param_dict,
                X_dtype=X_dtype,
            )
            self.classification_module = classification_module
            self.n_annotators = n_annotators

        def _fit(self, fit_function, X, y, **fit_params):
            """Initialize and fit the module.

            If the module was already initialized, by calling fit, the module
            will be re-initialized (unless ``warm_start`` is True).

            Parameters
            ----------
            X : matrix-like, shape (n_samples, ...)
                Training data set, usually complete, i.e. including the labeled
                and unlabeled samples
            y : array-like of shape (n_samples,)
                Labels of the training data set (possibly including unlabeled
                ones indicated by self.missing_label)
            fit_params : dict-like
                Further parameters as input to the 'fit' method of
                `skorch.net.NeuralNet`.

            Returns
            -------
            self: SkorchClassifier,
                The SkorchClassifier is fitted on the training data.
            """
            # Check input parameters
            X, y, sample_weight = self._validate_data(
                X=X, y=y, check_X_dict=self._check_X_dict, y_ensure_1d=False
            )

            # Optional initialization.
            if (
                not hasattr(self, "initialized_")
                or not self.initialized_
                or fit_function == "fit"
            ):
                self.initialize(n_annotators=y.shape[-1])

            # Fit on labeled data.
            is_lbld = is_labeled(y, missing_label=-1).any(axis=1)
            if np.sum(is_lbld) > 0:
                net = self.neural_net_.module_
                old_forward_return = net.forward_return
                try:
                    net.set_forward_return("logits_annot")
                    X_lbld = X[is_lbld]
                    y_lbld = y[is_lbld].astype(np.int64)
                    self.neural_net_.partial_fit(X_lbld, y_lbld, **fit_params)
                    self.is_fitted_ = True
                finally:
                    net.set_forward_return(old_forward_return)
            else:
                self.is_fitted_ = False
            return self

        def predict_annotator_perf(self, X, return_confusion_matrix=False):
            """Calculates the probability that an annotator provides the true
            label for a given sample.

            Parameters
            ----------
            X : matrix-like of shape (n_samples, n_features)
                Test samples.
            return_confusion_matrix : bool, default=False
                If `return_confusion_matrix=True`, the entire confusion matrix
                per annotator is returned.

            Returns
            -------

            """
            if not hasattr(self, "initialized_") or not self.initialized_:
                self.initialize()
            n_annotators = self.n_annotators
            net = self.neural_net_
            p_class, logits_annot = net.forward(X)
            p_class = p_class.numpy()
            P_annot = F.softmax(logits_annot, dim=-1)
            P_annot = P_annot.numpy()
            P_perf = np.array(
                [
                    np.einsum("ij,ik->ijk", p_class, P_annot[:, :, i])
                    for i in range(n_annotators)
                ]
            )
            P_perf = P_perf.swapaxes(0, 1)
            if return_confusion_matrix:
                return P_perf
            return P_perf.diagonal(axis1=-2, axis2=-1).sum(axis=-1)

        def predict_proba(
            self,
            X,
            return_embeddings=False,
            return_annot_perf=False,
            return_annot_proba=False,
        ):
            """Returns class-membership probability estimates for the test data
            `X`.

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
                Only returned, if `return_annot_perf=True`.
            P_annot : numpy.ndarray of shape (n_samples, n_classes, n_annotators)
                `P_annot[n, c, m]` refers to the probability that annotator
                `m` provides the class label `c` for instance `X[n]`.
                Only returned, if `return_annot_proba=True`.
            """
            # Check input parameters.
            if not hasattr(self, "random_state_"):
                self.random_state_ = check_random_state(self.random_state)
            X = check_array(X, **self._check_X_dict)
            check_n_features(
                self, X, reset=not hasattr(self, "n_features_in_")
            )
            check_scalar(
                return_embeddings, name="return_embeddings", target_type=bool
            )
            check_scalar(
                return_annot_perf, name="return_annot_perf", target_type=bool
            )
            check_scalar(
                return_annot_proba, name="return_annot_proba", target_type=bool
            )

            # Initialize module, if not done yet.
            if not hasattr(self, "initialized_") or not self.initialized_:
                self.initialize()

            # Set forward option.
            net = self.neural_net_.module_
            old_forward_return = net.forward_return
            forward_options = ["p_class"]
            if return_embeddings:
                forward_options.append("x_embed")
            if return_annot_perf or return_annot_proba:
                forward_options.append("logits_annot")
            net.set_forward_return(forward_options)

            try:
                out_torch = self.neural_net_.forward(X)
                if isinstance(out_torch, tuple):
                    P_class = to_numpy(out_torch[0])
                    out_numpy = [P_class]
                else:
                    P_class = to_numpy(out_torch)
                    out_numpy = P_class
                if return_embeddings:
                    X_embed = to_numpy(out_torch[1])
                    out_numpy.append(X_embed)
                if return_annot_perf or return_annot_proba:
                    L_annot = out_torch[-1]
                    P_annot = to_numpy(L_annot.softmax(dim=1))
                    if return_annot_perf:
                        P_perf = P_class[:, None] @ P_annot
                        out_numpy.append(P_perf[:, 0, :])
                    if return_annot_proba:
                        out_numpy.append(P_annot)
            finally:
                net.set_forward_return(old_forward_return)

            # Initialize fallbacks if the classifier hasn't been fitted before.
            self._initialize_fallbacks(P=P_class)
            return tuple(out_numpy)

        def initialize(self, n_annotators=None):
            """Initialize the internal `sklearn` wrapper from `skorch`."""
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

            if self.classes is not None:
                n_classes = len(self.classes)
            elif self.classes_ is not None:
                n_classes = len(self.classes_)
            else:
                raise RuntimeError(
                    "The number of classes needs to be known prior to the"
                    "initialization of the `CrowdLayerModule`. Either set"
                    "`self.classes` or call `self.fit` beforehand."
                )

            if (
                self.n_annotators is not None
                and n_annotators is not None
                and self.n_annotators != n_annotators
            ):
                raise ValueError(
                    "The number of annotators needs to be the same as given"
                    "for the initialization of the classifier object."
                )
            if self.n_annotators is None and n_annotators is None:
                raise ValueError(
                    "The number of annotators must be either given for the "
                    "initialization of the classifier object or the "
                    "initialization of the neural net."
                )
            self.n_annotators_ = (
                self.n_annotators
                if self.n_annotators is not None
                else n_annotators
            )

            neural_net_param_dict_override = {
                "criterion__reduction": "mean",
                "criterion__ignore_index": -1,
                "module__n_classes": n_classes,
                "module__n_annotators": self.n_annotators_,
                "module__clf_module": self.classification_module,
                "module__clf_module_param_dict": clf_module_param_dict,
            }
            for p_name, p_val in neural_net_param_dict_override.items():
                if p_name in neural_net_param_dict:
                    if p_val != neural_net_param_dict[p_name]:
                        raise ValueError(
                            f"The value for "
                            f"`neural_net_param_dict[{p_name}]` must either "
                            f"be left undefined or must be set to `{p_val}`."
                        )
            neural_net_param_dict.update(neural_net_param_dict_override)
            self.neural_net_ = NeuralNet(
                module=self.module,
                criterion=self.criterion,
                **neural_net_param_dict,
            )
            self.neural_net_.initialize()
            self.initialized_ = True

    class _CrowdLayerModule(nn.Module):
        """Crowd Layer Module

        Crowd Layer [1]_ is a layer added at the end of a classifying neural
        network and allows us to train deep neural networks end-to-end,
        directly from the noisy labels of multiple annotators, using only
        backpropagation.

        Parameters
        ----------
        n_classes : int
            Number of classes.
        n_annotators : int
            Number of annotators.
        clf_module : nn.Module
            Pytorch module of the classification module taking samples as
            input to predict class-membership logits.

        References
        ----------
        .. [1] Rodrigues, Filipe, and Francisco Pereira. "Deep Learning from
           Crowds." AAAI Conference on Artificial Intelligence, 2018.
        """

        # Names that may be *optionally* returned.
        OUTPUTS = {"p_class", "x_embed", "logits_annot"}

        def __init__(
            self, n_classes, n_annotators, clf_module, clf_module_param_dict
        ):
            super().__init__()
            self.n_classes = n_classes
            self.n_annotators = n_annotators
            if isclass(clf_module):
                self.classification_module = clf_module(
                    **clf_module_param_dict
                )
            else:
                self.classification_module = clf_module

            # By default, return only `logits_annot`.
            self.set_forward_return("logits_annot")

            # Setup crowd layer.
            self.annotator_layers = nn.ModuleList()
            for i in range(n_annotators):
                layer = nn.Linear(n_classes, n_classes, bias=False)
                layer.weight = nn.Parameter(torch.eye(n_classes))
                self.annotator_layers.append(layer)

        def set_forward_return(self, values):
            """
            Choose which *additional* tensors (besides `p_class`) you want
            `forward` to return. Valid names: "x_embed", "logits_annot".

            Parameters
            ----------
            values : str or array-like
                Accepts only "x_embed", "logits_annot" or
                ["x_embed", "logits_annot"].

            Returns
            -------
            self : nn.Module
                Crowd layer module.
            """
            if isinstance(values, str):
                values = [values]

            unknown = set(values) - self.OUTPUTS
            if unknown:
                raise ValueError(f"Unknown forward return(s): {unknown}")
            if len(values) == 0:
                raise ValueError(f"No forward return(s): {values}")

            self.forward_return = set(values)
            return self

        def forward(self, x: torch.Tensor):
            """
            Forward pass.

            Parameters
            ----------
            x : torch.Tensor of shape (batch_size, ...)
                Input samples.

            Returns
            -------
            p_class : torch.Tensor of shape (batch_size, n_classes)
                Class-membership probabilities.
            x_embed : torch.Tensor of shape (batch_size, ...)
                Learned embeddings of samples. Only returned if "x_embed" in
                self.forward_return.
            logits_annot : torch.Tensor of shape (batch_size, n_classes,\
                    n_annotators)
                Annotation logits for each sample-annotator pair. Only returned
                if "logits_annot" in self.forward_return.
            """
            # Inference of the classification module.
            cls_out = self.classification_module(x)
            if isinstance(cls_out, tuple):
                logits_class, x_embed = cls_out
            else:
                logits_class, x_embed = cls_out, None
            p_class = F.softmax(logits_class, dim=-1)

            # Check whether to add sampled embeddings to `outputs`.
            outputs = []
            if "p_class" in self.forward_return:
                outputs.append(p_class)

            if "x_embed" in self.forward_return:
                if x_embed is None:
                    raise RuntimeError(
                        "`x_embed` was requested, but the classification "
                        "module did not return it."
                    )
                outputs.append(x_embed)

            # Compute logits for the annotator labels and add them to `outputs`.
            if "logits_annot" in self.forward_return:
                logits_annot = []
                for layer in self.annotator_layers:
                    logits_annot.append(layer(p_class))
                logits_annot = torch.stack(logits_annot, dim=2)
                outputs.append(logits_annot)

            return outputs[0] if len(outputs) == 1 else tuple(outputs)
            # return {"logits_annot": logits_annot}
