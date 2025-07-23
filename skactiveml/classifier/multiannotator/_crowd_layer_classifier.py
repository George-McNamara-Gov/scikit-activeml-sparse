import numpy as np
from sklearn.utils.validation import check_array

successful_skorch_torch_import = False
try:
    from skorch import NeuralNet
    from skorch.dataset import unpack_data
    import torch
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
    ExtLabelEncoder,
    is_labeled,
    check_random_state,
    check_n_features,
)

if successful_skorch_torch_import:

    class CrowdLayerClassifier(SkorchClassifier, AnnotatorModelMixin):
        """
        Crowd Layer

        Crowd Layer [1]_ is a layer added at the end of a classifying neural
        network and allows us to train deep neural networks end-to-end,
        directly from the noisy labels of multiple annotators, using only
        backpropagation.

        Parameters
        ----------
        module__n_annotators : int
            Number of annotators.
        module__gt_net : nn.Module
            Pytorch module of the GT model taking samples as input to predict
            class-membership logits.
        *args :
            More possible arguments for initializing your neural network (cf.
            https://skorch.readthedocs.io/en/stable/net.html).
        classes : array-like of shape (n_classes,), default=None
            Holds the label for each class. If none, the classes are determined
            during the fit.
        missing_label : scalar or string or np.nan or None, default=np.nan
            Value to represent a missing label.
        cost_matrix : array-like of shape (n_classes, n_classes)
            Cost matrix with `cost_matrix[i,j]` indicating cost of predicting
            class `classes[j]` for a sample of class `classes[i]`. Can be only
            set, if `classes` is not none.
        random_state : int or RandomState instance or None, default=None
            Determines random number for 'predict' method. Pass an int for
            reproducible results across multiple method calls.
        **kwargs : keyword arguments
            More possible parameters for customizing your neural network (cf.
            https://skorch.readthedocs.io/en/stable/net.html). This class
            overrides the criterion and is not compatible with other criteria.

        References
        ----------
        .. [1] Rodrigues, Filipe, and Francisco Pereira. "Deep Learning from
            Crowds." AAAI Conference on Artificial Intelligence, 2018.

        """

        def __init__(
            self,
            n_annotators,
            gt_net,
            classes=None,
            missing_label=MISSING_LABEL,
            cost_matrix=None,
            random_state=None,
            neural_net_param_dict=None,
            X_dtype=None,
        ):
            super(CrowdLayerClassifier, self).__init__(
                module=CrowdLayerModule,
                criterion=CrossEntropyLoss,
                classes=classes,
                missing_label=missing_label,
                cost_matrix=cost_matrix,
                random_state=random_state,
                neural_net_param_dict=neural_net_param_dict,
                X_dtype=X_dtype,
            )
            self.n_annotators = n_annotators
            self.gt_net = gt_net

        def get_loss(self, y_pred, y_true, *args, **kwargs):
            """Return the loss for this batch.

            Parameters
            ----------
            y_pred : torch.Tensor
            Predicted target values
            y_true : torch.Tensor
            True target values.

            Returns
            ---------
            loss : torch.Tensor
                Loss for this batch
            """
            if not hasattr(self, "initialized_") or not self.initialized_:
                self.initialize()
            # unpack the tuple from the forward function
            p_class, logits_annot = y_pred
            loss = self.neural_net_.get_loss(
                logits_annot, y_true, *args, **kwargs
            )
            return loss

        def _fit(self, fit_function, X, y, **fit_params):
            """Initialize and fit the module.

            If the module was already initialized, by calling fit, the module
            will be re-initialized (unless ``warm_start`` is True).

            Parameters
            ----------
            X : matrix-like, shape (n_samples, n_features)
                Training data set, usually complete, i.e. including the labeled
                and unlabeled samples
            y : array-like of shape (n_samples, )
                Labels of the training data set (possibly including unlabeled
                ones indicated by self.missing_label)
            fit_params : dict-like
                Further parameters as input to the 'fit' method of the
                'estimator'.

            Returns
            -------
            self: SkorchClassifier,
                The SkorchClassifier is fitted on the training data.
            """
            # check input parameters
            self.check_X_dict_ = {
                "ensure_min_samples": 0,
                "ensure_min_features": 0,
                "allow_nd": True,
                "dtype": None,
            }
            X, y, sample_weight = self._validate_data(
                X=X, y=y, check_X_dict=self.check_X_dict_, y_ensure_1d=False
            )

            if self.X_dtype is not None:
                X = X.astype(self.X_dtype)

            is_lbld = is_labeled(y, missing_label=-1).any(axis=1)

            if (
                not hasattr(self, "initialized_")
                or not self.initialized_
                or fit_function == "fit"
            ):
                self.initialize()
            if np.sum(is_lbld) > 0:
                net = self.neural_net_.module_
                old_forward_return = net.forward_return
                try:
                    net.set_forward_return("logits_annot")
                    X_lbld = X[is_lbld]
                    y_lbld = y[is_lbld].astype(np.int64)
                    if fit_function == "fit":
                        self.neural_net_.partial_fit(
                            X_lbld, y_lbld, **fit_params
                        )
                    elif fit_function == "partial_fit":
                        self.neural_net_.partial_fit(
                            X_lbld, y_lbld, **fit_params
                        )
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
            P_perf : numpy.ndarray of shape (n_samples, n_annotators) or
            (n_samples, n_annotators, n_classes, n_classes)
                If `return_confusion_matrix=False`, `P_perf[n, m]` is the
                probability, that annotator `A[m]` provides the correct class
                label for sample `X[n]`. If `return_confusion_matrix=False`,
                `P_perf[n, m, c, j]` is the probability, that annotator `A[m]`
                provides the correct class label `classes_[j]` for sample
                `X[n]` and that this sample belongs to class `classes_[c]`. If
                `return_cond=True`, `P_perf[n, m, c, j]` is the probability
                that annotator `A[m]` provides the class label `classes_[j]`
                for sample `X[n]` conditioned that this sample belongs to class
                `classes_[c]`.
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

        def predict_proba(self, X):
            """Returns class-membership probability estimates for the test data
            `X`.

            Parameters
            ----------
            X : matrix-like, shape (n_samples, n_features)
                Test samples.

            Returns
            -------
            p_class : numpy.ndarray of shape (n_samples, classes)
                `p_class[n, c]` is the probability, that instance `X[n]`
                belongs to the `classes_[c]`.
            """
            if not hasattr(self, "random_state_"):
                self.random_state_ = check_random_state(self.random_state)
            if not hasattr(self, "check_X_dict_"):
                self.check_X_dict_ = {
                    "ensure_min_samples": 0,
                    "ensure_min_features": 0,
                    "allow_nd": True,
                    "dtype": None,
                }

            X = check_array(X, **self.check_X_dict_)
            if self.X_dtype is not None:
                X = X.astype(self.X_dtype)

            reset_n_features_in_ = not hasattr(self, "n_features_in_")
            check_n_features(self, X, reset=reset_n_features_in_)

            if not hasattr(self, "initialized_") or not self.initialized_:
                self.initialize()
            net = self.neural_net_.module_
            old_forward_return = net.forward_return
            try:
                net.set_forward_return("p_class")
                p_class = self.neural_net_.forward(X).numpy()
            finally:
                net.set_forward_return(old_forward_return)

            if not hasattr(self, "_le"):
                # initialize fallbacks if the classifier hasn't been fitted
                # before
                self._le = ExtLabelEncoder(
                    classes=self.classes, missing_label=self.missing_label
                )
                if self.classes is not None:
                    y_dummy = self.classes
                else:
                    y_dummy = np.arange(P.shape[-1], dtype=int)
                y_dummy = self._le.fit_transform(y_dummy)
                self.classes_ = self._le.classes_
            if not hasattr(self, "cost_matrix_"):
                self.cost_matrix_ = (
                    1 - np.eye(len(self.classes_))
                    if self.cost_matrix is None
                    else self.cost_matrix
                )
            return p_class

        def predict_proba_annot(self, X):
            """Predict the probabilities of annotator assign for a label for
            the given input data.

            Parameters
            ----------
            X : matrix-like of shape (n_samples, n_features)
                Test samples.

            Returns
            -------
            numpy.ndarray
                The predicted probabilities for each annotator, obtained by
                applying softmax to the logits.
            """
            if not hasattr(self, "initialized_") or not self.initialized_:
                self.initialize()
            with torch.no_grad():
                _, logits_annot = self.neural_net_.forward(X)
                P_annot = F.softmax(logits_annot, dim=-1).numpy()
            return P_annot

        def initialize(self):
            """initialize the internal `sklearn` wrapper from `skorch`."""
            neural_net_param_dict = self.neural_net_param_dict.copy()
            if neural_net_param_dict is None:
                neural_net_param_dict = {}

            if self.classes is not None:
                n_classes = len(self.classes)
            elif hasattr(self, "classes_"):
                n_classes = len(self.classes)
            else:
                raise RuntimeError(
                    "The number of classes needs to be known prior to the"
                    "initialization of the `CrowdLayerModule`. Either set"
                    "`self.classes` or call `self.fit` beforehand."
                )

            neural_net_param_dict_override = {
                "criterion__reduction": "mean",
                "criterion__ignore_index": -1,
                "module__n_classes": n_classes,
                "module__n_annotators": self.n_annotators,
                "module__gt_net": self.gt_net,
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

    class CrowdLayerModule(nn.Module):
        """
        Crowd Layer Module

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
        gt_net : nn.Module
            Pytorch module of the GT model taking samples as input to predict
            class-membership logits.

        References
        ----------
        .. [1] Rodrigues, Filipe, and Francisco Pereira. "Deep learning from
        crowds." AAAI Conference on Artificial Intelligence, 2018.
        """

        def __init__(
            self,
            n_classes,
            n_annotators,
            gt_net,
        ):
            super().__init__()
            self.n_classes = n_classes
            self.n_annotators = n_annotators
            self.gt_net = gt_net
            self.forward_return = "both"

            # Setup crowd layer.
            self.annotator_layers = nn.ModuleList()
            for i in range(n_annotators):
                layer = nn.Linear(n_classes, n_classes, bias=False)
                layer.weight = nn.Parameter(torch.eye(n_classes))
                self.annotator_layers.append(layer)

        def set_forward_return(self, value):
            self.forward_return = value
            return self

        def forward(self, x):
            """Forward propagation of samples through the GT and AP (optional)
            model.

            Parameters
            ----------
            x : torch.Tensor of shape (batch_size, *)
                Samples.

            Returns
            -------
            p_class : torch.Tensor of shape (batch_size, n_classes)
                Class-membership probabilities.
            logits_annot : torch.Tensor of shape (batch_size, n_classes,
            n_annotators)
                Annotation logits for each sample-annotator pair.
            """
            # Compute class-membership logits.
            logit_class = self.gt_net(x)

            # Compute class-membership probabilities.
            p_class = F.softmax(logit_class, dim=-1)

            if self.forward_return == "p_class":
                return p_class

            # Compute logits per annotator.
            logits_annot = []
            for layer in self.annotator_layers:
                logits_annot.append(layer(p_class))
            logits_annot = torch.stack(logits_annot, dim=2)

            if self.forward_return == "logits_annot":
                return logits_annot
            return p_class, logits_annot
