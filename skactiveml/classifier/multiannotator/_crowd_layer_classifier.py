import numpy as np
from sklearn.utils.validation import check_array
from ...utils import (
    MISSING_LABEL,
    check_n_features,
    check_scalar,
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
    from ._utils import (
        _MultiAnnotatorClassifier,
        _MultiAnnotatorClassificationModule,
        _MultiAnnotatorCollate,
    )

    successful_skorch_torch_import = True
except ImportError:
    pass  # pragma: no cover

if successful_skorch_torch_import:

    class CrowdLayerClassifier(_MultiAnnotatorClassifier):
        """Crowd Layer

        Crowd Layer [1]_ is a layer added at the end of a classifying neural
        network and allows us to train deep neural networks end-to-end,
        directly from the noisy labels of multiple annotators, using only
        backpropagation.

        Parameters
        ----------
        clf_module : nn.Module or nn.Module.__class__
            A PyTorch module as classification model outputting logits for
            samples as input. In general, the uninstantiated class should
            be passed, although instantiated modules will also work.
        n_annotators : int, default=None
            Number of annotators. If `n_annotators=None`, the number of
            annotators is inferred from `y` when calling `fit`.
        neural_net_param_dict : dict, default=None
            Additional arguments for `skorch.net.NeuralNet`. If
            `neural_net_param_dict` is None, no additional arguments
             are added.
        sample_dtype : str or type, default=None
            The type or typecode all data is casted to. If `sample_dtype` is
            `None`, the datatype is preserved.
        classes : array-like of shape (n_classes,), default=None
            Holds the label for each class. If `None`, the classes are
            determined during the fit.
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
            clf_module,
            n_annotators=None,
            neural_net_param_dict=None,
            sample_dtype=None,
            classes=None,
            cost_matrix=None,
            missing_label=MISSING_LABEL,
            random_state=None,
        ):
            super(CrowdLayerClassifier, self).__init__(
                multi_annotator_module=_CrowdLayerModule,
                clf_module=clf_module,
                criterion=CrossEntropyLoss,
                classes=classes,
                missing_label=missing_label,
                cost_matrix=cost_matrix,
                random_state=random_state,
                neural_net_param_dict=neural_net_param_dict,
                sample_dtype=sample_dtype,
            )
            self.clf_module = clf_module
            self.n_annotators = n_annotators

        def predict(
            self,
            X,
            return_logits=False,
            return_embeddings=False,
            return_annotator_perf=False,
            return_annotator_class=False,
        ):
            """Returns class predictions for the test data `X`. Optionally, a
            tuple is returned whose elements appear in this exact order if they
            were requested:

            - (0) `y_pred` – always returned,
            - (1) `L_class` - if `return_logits`,
            - (2) `X_embed` – if `return_embeddings`,
            - (3) `P_annotator_perf`  – if `return_annotator_perf`,
            - (4) `P_annotator_class` – if `return_annotator_class`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Test samples.
            return_logits : bool, default=False
                If `return_logits=True`, additionally return the
                class-membership logits for the samples in `X` as the second
                element of the output tuple.
            return_embeddings : bool, default=False
                If `return_embeddings=True`, additionally return the learned
                embeddings `X_embed` for the samples in `X` as the next
                element of the output tuple.
            return_annotator_perf : bool, default=False
                If `return_annotator_perf=True`, additionally return the
                estimated annotator performance probabilities `P_perf` for each
                sample–annotator pair as the next element of the output tuple.
            return_annotator_class : bool, default=False
                If `return_annotator_class=True`, additionally return the
                annotator–class probability estimates `P_annot` for each sample,
                class, and annotator as the last element of the output tuple.

            Returns
            -------
            y_pred : np.ndarray of shape (n_samples,)
                `y_pred[n]` is the predicted class label for sample `X[n]`.
            L_class : np.ndarray of shape (n_samples, n_classes)
                `L_class[n, c]` is the logit the class `classes_[c]` of sample
                `X[n]`.
            X_embed : np.ndarray of shape (n_samples, ...)
                `X_embed[n]` refers to the learned embedding for sample `X[n]`.
                Only returned, if `return_embeddings=True`.
            P_perf : np.ndarray of shape (n_samples, n_annotators)
                `P_perf[n, m]` refers to the estimated correct probability
                (performance) of annotator `m` when labeling sample `X[n]`.
                Only returned, if `return_annotator_perf=True`.
            P_annot : np.ndarray of shape (n_samples, n_annotators, n_classes)
                `P_annot[n, m, c]` refers to the probability that annotator
                `m` provides the class label `c` for instance `X[n]`.
                Only returned, if `return_annotator_class=True`.
            """
            predict_dict = {k: v for k, v in locals().items() if k != "self"}
            return self._transform_predict_proba_output(
                predict_dict=predict_dict
            )

        def predict_proba(
            self,
            X,
            return_logits=False,
            return_embeddings=False,
            return_annotator_perf=False,
            return_annotator_class=False,
        ):
            """Returns class-membership probability estimates for the test data
            `X`. Optionally, a tuple is returned whose elements appear in this
            exact order if they were requested:

            - (0) `P_class` – always returned,
            - (1) `L_class` - if `return_logits`,
            - (2) `X_embed` – if `return_embeddings`,
            - (3) `P_annotator_perf`  – if `return_annotator_perf`,
            - (4) `P_annotator_class` – if `return_annotator_class`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Test samples.
            return_logits : bool, default=False
                If `return_logits=True`, additionally return the
                class-membership logits for the samples in `X` as the second
                element of the output tuple.
            return_embeddings : bool, default=False
                If `return_embeddings=True`, additionally return the learned
                embeddings `X_embed` for the samples in `X` as the next
                element of the output tuple.
            return_annotator_perf : bool, default=False
                If `return_annotator_perf=True`, additionally return the
                estimated annotator performance probabilities `P_perf` for each
                sample–annotator pair as the next element of the output tuple.
            return_annotator_class : bool, default=False
                If `return_annotator_class=True`, additionally return the
                annotator–class probability estimates `P_annot` for each sample,
                class, and annotator as the last element of the output tuple.

            Returns
            -------
            P_class : np.ndarray of shape (n_samples, classes)
                `p_class[n, c]` is the probability, that sample `X[n]`
                belongs to the `classes_[c]`.
            L_class : np.ndarray of shape (n_samples, n_classes)
                `L_class[n, c]` is the logit the class `classes_[c]` of sample
                `X[n]`.
            X_embed : np.ndarray of shape (n_samples, ...)
                `X_embed[n]` refers to the learned embedding for sample `X[n]`.
                Only returned, if `return_embeddings=True`.
            P_perf : np.ndarray of shape (n_samples, n_annotators)
                `P_perf[n, m]` refers to the estimated correct probability
                (performance) of annotator `m` when labeling sample `X[n]`.
                Only returned, if `return_annotator_perf=True`.
            P_annot : np.ndarray of shape (n_samples, n_annotators, n_classes)
                `P_annot[n, m, c]` refers to the probability that annotator
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
                return_embeddings, name="return_logits", target_type=bool
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
            if return_annotator_perf or return_annotator_class:
                forward_options.append("logits_annot")
            net.set_forward_return(forward_options)

            # Compute predictions for the different outputs required
            # by the input parameters.
            try:
                out_torch = self.neural_net_.forward(X)
                if isinstance(out_torch, tuple):
                    P_class = to_numpy(out_torch[0].softmax(dim=-1))
                    out_numpy = [P_class]
                else:
                    P_class = to_numpy(out_torch.softmax(dim=-1))
                    out_numpy = P_class
                if return_logits:
                    L_class = to_numpy(out_torch[0])
                    out_numpy.append(L_class)
                if return_embeddings:
                    X_embed = to_numpy(out_torch[1])
                    out_numpy.append(X_embed)
                if return_annotator_perf or return_annotator_class:
                    L_annot = out_torch[-1]
                    P_annot = to_numpy(L_annot.softmax(dim=-1))
                    if return_annotator_perf:
                        P_perf = np.einsum("nc,nmc->nm", P_class, P_annot)
                        out_numpy.append(P_perf)
                    if return_annotator_class:
                        out_numpy.append(P_annot)
            finally:
                net.set_forward_return(old_forward_return)

            # Initialize fallbacks if the classifier hasn't been fitted before.
            self._initialize_fallbacks(P=P_class)
            if isinstance(out_numpy, np.ndarray):
                return out_numpy
            else:
                return tuple(out_numpy)

        def _build_neural_net_param_overrides(self, X, y):
            collate_fn = _MultiAnnotatorCollate(missing_label=-1)
            return {
                "criterion__reduction": "mean",
                "criterion__ignore_index": -1,
                "module__n_classes": len(self.classes_),
                "module__n_annotators": self.n_annotators_,
                "iterator_train__collate_fn": collate_fn,
                "predict_nonlinearity": None,
            }

    class _CrowdLayerModule(_MultiAnnotatorClassificationModule):
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
        clf_module_param_dict : dict
            Keyword args for constructing `clf_module` if a class is passed.

        References
        ----------
        .. [1] Rodrigues, Filipe, and Francisco Pereira. "Deep Learning from
           Crowds." AAAI Conference on Artificial Intelligence, 2018.
        """

        def __init__(
            self, n_classes, n_annotators, clf_module, clf_module_param_dict
        ):
            super().__init__(
                clf_module=clf_module,
                clf_module_param_dict=clf_module_param_dict,
                default_forward_outputs="logits_annot",
                full_forward_outputs=[
                    "logits_class",
                    "x_embed",
                    "logits_annot",
                ],
            )
            self.n_classes = n_classes
            self.n_annotators = n_annotators

            # Setup crowd layer.
            self.W_annot = torch.eye(n_classes).repeat(n_annotators, 1, 1)
            self.W_annot = nn.Parameter(self.W_annot)

        def forward(self, x, input_ids=None):
            """
            Forward pass through the classification module and optionally
            through the crowd layer.

            Parameters
            ----------
            x : torch.Tensor of shape (batch_size, ...)
                Input samples.
            input_ids : torch.Tensor of shape (batch_size, 2), default=None
                - If `isinstance(input_ids, torch.Tensor)=True`, the column
                  `input_ids[:, 0]` refers to the sample indices and the column
                  `input_ids[:, 1]` to the annotator indices to be propagated
                  through the crowd-layer.
                - If `input_ids=None`, all combinations of samples and
                  annotators are propagated through the crowd-layer.

            Returns
            -------
            logits_class : torch.Tensor of shape (batch_size, n_classes)
                Class-membership logits.
            x_embed : torch.Tensor of shape (batch_size, ...)
                Learned embeddings of samples. Only returned if "x_embed" in
                `self.forward_return`.
            logits_annot : torch.Tensor of shape (batch_size, n_annotators,\
                    n_classes) or (len(input_ids), n_annotators, n_classes)
                Annotation logits for sample-annotator pairs. Only returned
                if "logits_annot" in self.forward_return. Shape depends on
                whether `input_ids` is given or `None`.
            """
            # Inference of classification model.
            logits_class, x_embed = self.clf_module_forward(x)

            # Append classifier outputs to `out` if required.
            out = []
            if "logits_class" in self.forward_return:
                out.append(logits_class)
            if "x_embed" in self.forward_return:
                out.append(x_embed)

            # Add annotator logits to `out` if required.
            if "logits_annot" in self.forward_return:
                p_class = F.softmax(logits_class, dim=-1)
                if isinstance(input_ids, torch.Tensor):
                    x = p_class.index_select(0, input_ids[:, 0])
                    W_sel = self.W_annot.index_select(0, input_ids[:, 1])
                    logits_annot = torch.einsum("mi,moi->mo", x, W_sel)
                else:
                    logits_annot = torch.einsum(
                        "ni,aoi->nao", p_class, self.W_annot
                    )
                out.append(logits_annot)

            return out[0] if len(out) == 1 else tuple(out)
