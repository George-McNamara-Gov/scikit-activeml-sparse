import inspect
import warnings
from copy import deepcopy
from operator import attrgetter

import numpy as np
from scipy.stats import norm
from sklearn.base import MetaEstimatorMixin, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
)

from ..base import SkactivemlRegressor, ProbabilisticRegressor
from ..utils import (
    is_labeled,
    match_signature,
    check_n_features,
    check_type,
    check_scalar,
    MISSING_LABEL,
)

successful_skorch_torch_import = False
try:
    from torch import nn
    from skorch import NeuralNet
    from skorch.utils import to_numpy
    from skactiveml.base import SkorchMixin
    from skactiveml.utils import make_criterion_tuple_aware

    successful_skorch_torch_import = True
except ImportError:  # pragma: no cover
    pass


class SklearnRegressor(SkactivemlRegressor, MetaEstimatorMixin):
    """Sklearn Regressor

    Implementation of a wrapper class for scikit-learn regressors such that
    missing labels can be handled. Therefore, samples with missing values are
    filtered.

    Parameters
    ----------
    estimator : sklearn.base.RegressorMixin with predict method
        scikit-learn regressor.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.
    """

    def __init__(
        self, estimator, missing_label=MISSING_LABEL, random_state=None
    ):
        super().__init__(
            random_state=random_state, missing_label=missing_label
        )
        self.estimator = estimator

    @match_signature("estimator", "fit")
    def fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Fit the model using X as training data and y as labels.

        Parameters
        ----------
        X : matrix-like of shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like of shape (n_samples,)
            It contains the numeric target values of the training samples.
            Missing labels are represented as `self.missing_label`.
        sample_weight : array-like of shape (n_samples,), default=None
            It contains the weights of the training samples´ labels. It
            must have the same shape as y.
        fit_kwargs : dict-like
            Further parameters are passed as input to the `fit` method of the
            'estimator'.

        Returns
        -------
        self: SklearnRegressor,
            The SklearnRegressor is fitted on the training data.
        """
        return self._fit(
            fit_function="fit",
            X=X,
            y=y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )

    @match_signature("estimator", "partial_fit")
    def partial_fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Partially fitting the model using X as training data and y as class
        labels.

        Parameters
        ----------
        X : matrix-like of shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like of shape (n_samples,)
            It contains the numeric labels of the training samples.
            Missing labels are represented the attribute `self.missing_label`.
        sample_weight : array-like of shape (n_samples,)
            It contains the weights of the training samples' numeric labels. It
            must have the same shape as y.
        fit_kwargs : dict-like
            Further parameters as input to the `fit` method of the `estimator`.

        Returns
        -------
        self : SklearnRegressor,
            The `SklearnRegressor` is fitted on the training data.
        """
        return self._fit(
            fit_function="partial_fit",
            X=X,
            y=y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )

    def _fit(self, fit_function, X, y, sample_weight, **fit_kwargs):
        if not is_regressor(estimator=self.estimator):
            raise TypeError(
                "'{}' must be a scikit-learn "
                "regressor.".format(self.estimator)
            )

        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }

        X, y, sample_weight = self._validate_data(
            X,
            y,
            sample_weight,
            check_X_dict=self.check_X_dict_,
            reset=fit_function == "fit" or not hasattr(self, "n_features_in_"),
        )

        is_lbld = is_labeled(y, missing_label=self.missing_label_)
        X_labeled = X[is_lbld]
        y_labeled = y[is_lbld]
        estimator_params = dict(fit_kwargs) if fit_kwargs is not None else {}

        if sample_weight is not None:
            estimator_params["sample_weight"] = sample_weight[is_lbld]

        self._label_mean = np.mean(y[is_lbld]) if np.sum(is_lbld) > 0 else 0
        self._label_std = np.std(y[is_lbld]) if np.sum(is_lbld) > 1 else 1
        self.estimator_ = deepcopy(self.estimator)
        try:
            attrgetter(fit_function)(self.estimator_)(
                X_labeled, y_labeled, **estimator_params
            )
        except Exception as e:
            warnings.warn(
                f"The 'estimator' could not be fitted because of"
                f" '{e}'. Therefore, the empirical label mean "
                f"`_label_mean={self._label_mean}` and the "
                f"empirical label standard deviation "
                f"`_label_std={self._label_std}` will be used to make "
                f"predictions."
            )

        return self

    @match_signature("estimator", "predict")
    def predict(self, X, **predict_kwargs):
        """Return label predictions for the input data `X`.

        Parameters
        ----------
        X :  array-like of shape (n_samples, n_features)
            Input samples.
        predict_kwargs : dict-like
            Further parameters are passed as input to the `predict` method of
            the `estimator`. If the estimator could not be fitted, only
            `return_std` is supported as keyword argument.

        Returns
        -------
        y :  ndarray of shape (n_samples,)
            Predicted labels of the input samples.
        """
        check_is_fitted(self)
        predict_dict = {"ensure_min_samples": 1, "ensure_min_features": 1}
        X = check_array(X, **(self.check_X_dict_ | predict_dict))
        check_n_features(self, X, reset=False)
        try:
            return self.estimator_.predict(X, **predict_kwargs)
        except NotFittedError:
            warnings.warn(
                f"Since the 'estimator' could not be fitted when"
                f" calling the `fit` method, the label "
                f"mean `_label_mean={self._label_mean}` and optionally the "
                f"label standard deviation `_label_std={self._label_std}` is "
                f"used to make the predictions."
            )
            has_std = predict_kwargs.pop("return_std", False)
            if has_std:
                return (
                    np.full(len(X), self._label_mean),
                    np.full(len(X), self._label_std),
                )
            else:
                return np.full(len(X), self._label_mean)

    @match_signature("estimator", "sample_y")
    def sample_y(self, X, n_samples=1, **sample_kwargs):
        """Assumes a probabilistic regressor. Samples are drawn from a
        predicted target distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features)
            Input samples from which the target values are drawn.
        n_samples : int, default=1
            Number of random samples to be drawn.
        **sample_kwargs : dict
            Additional keyword arguments for sampling. For example:

            random_state : int, RandomState instance or None, default=None
                Determines the random number generation for drawing samples.
                Pass an int for reproducible results across multiple method
                calls.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples)
            Drawn random target samples.
        """
        return self._sample(
            sample_function="sample_y",
            X=X,
            n_samples=n_samples,
            **sample_kwargs,
        )

    @match_signature("estimator", "sample")
    def sample(self, X, n_samples=1, **sample_kwargs):
        """Assumes a probabilistic regressor. Samples are drawn from a
        predicted target distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features)
            Input samples from which the target values are drawn.
        n_samples : int, default=1
            Number of random samples to be drawn.
        **sample_kwargs : dict
            Additional keyword arguments for sampling. For example:

            random_state : int, RandomState instance or None, default=None
                Determines the random number generation for drawing samples.
                Pass an int for reproducible results across multiple method
                calls.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples)
            Drawn random target samples.
        """
        return self._sample(
            sample_function="sample", X=X, n_samples=n_samples, **sample_kwargs
        )

    def _sample(self, sample_function, X, n_samples=1, **sample_kwargs):
        check_is_fitted(self)
        predict_dict = {"ensure_min_samples": 1, "ensure_min_features": 1}
        X = check_array(X, **(self.check_X_dict_ | predict_dict))
        check_n_features(self, X, reset=False)
        try:
            return attrgetter(sample_function)(self.estimator_)(
                X, n_samples, **sample_kwargs
            )
        except NotFittedError:
            warnings.warn(
                f"Since the 'estimator' could not be fitted when"
                f" calling the `fit` method, the label "
                f"mean `_label_mean={self._label_mean}` and optionally the "
                f"label standard deviation `_label_std={self._label_std}` is "
                f"used to make the predictions."
            )
            random_state = sample_kwargs.get("random_state", None)
            random_state = check_random_state(random_state)
            check_scalar(
                n_samples,
                "n_samples",
                min_val=1,
                min_inclusive=True,
                target_type=int,
            )
            y_samples = random_state.randn(len(X), n_samples)
            y_samples *= self._label_std
            y_samples += self._label_mean
            return y_samples

    def __sklearn_is_fitted__(self):
        if hasattr(self, "_label_mean"):
            return True

        try:
            check_is_fitted(self.estimator)
        except NotFittedError:
            return False

        # set attributes that would be set by the fit function
        self._label_mean = 0
        self._label_std = 1
        self.estimator_ = deepcopy(self.estimator)
        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }

        return True

    def __getattr__(self, item):
        if "estimator_" in self.__dict__:
            return getattr(self.estimator_, item)
        else:
            return getattr(self.estimator, item)


class SklearnNormalRegressor(ProbabilisticRegressor, SklearnRegressor):
    """Sklearn Normal Regressor

    Implementation of a wrapper class for scikit-learn probabilistic regressors
    such that missing labels can be handled and the target distribution can be
    estimated. Therefore, samples with missing values are filtered and a normal
    distribution is fitted using the predicted means and standard deviations.

    The wrapped regressor of sklearn needs `return_std` as a key_word argument
    for `predict`.

    Parameters
    ----------
    estimator : sklearn.base.RegressorMixin with predict method
        scikit-learn regressor.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.
    """

    def __init__(
        self, estimator, missing_label=MISSING_LABEL, random_state=None
    ):
        super().__init__(
            estimator, missing_label=missing_label, random_state=random_state
        )

    def predict_target_distribution(self, X):
        """Returns the estimated target normal distribution conditioned on the
        test samples `X`.

        Parameters
        ----------
        X :  array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        dist : scipy.stats._distn_infrastructure.rv_frozen
            The distribution of the targets at the test samples.

        """
        check_is_fitted(self)

        if (
            "return_std"
            not in inspect.signature(self.estimator.predict).parameters.keys()
        ):
            raise ValueError(
                f"`{self.estimator}` must have key_word argument"
                f"`return_std` for predict."
            )

        loc, scale = SklearnRegressor.predict(self, X, return_std=True)
        return norm(loc=loc, scale=scale)


if successful_skorch_torch_import:

    class SkorchRegressor(SkactivemlRegressor, SkorchMixin):
        """SkorchRegressor

        Implement a wrapper class, to make it possible to use `torch` with
        `skactiveml`. This is achieved by providing a wrapper around `torch`
        that has a `skactiveml` interface and also be able to handle missing
        labels. This wrapper is based on the open-source library `skorch` [1]_.

        Notes
        -----
        Adjust your `criterion` with the outputs of your `nn.Module`.
        For example, if you use `criterion=nn.NLLLoss`, then your module is
        expected to output log-probabilities, which can be implemented through
        `nn.LogSoftmax(dim=1)`. To ensure that the `predict_proba` method can
        handle these log-probabilities, you need to set
        `"predict_nonlinearity": torch.exp` as part of the
        `neural_net_param_dict`, which then transforms the log-probabilities to
        actual probabilities.

        Parameters
        ----------
        module : torch module (class or instance)
            A PyTorch :class:`~torch.nn.Module`. In general, the uninstantiated
            class should be passed, although instantiated modules will also
            work.
        criterion : torch.nn.Module.__class__, default=torch.nn.NLLoss
            The uninitialized criterion (loss) used to optimize the module. By
            default, `torch.nn.NLLoss` is used as criterion.
        filter_criterion_input : bool, default=True
            - If True, this flag ensures criteria expecting tensors as input,
            e.g., `nn.CrossEntropyLoss`, work with implementations of the
            `module.forward` methods outputting tuples, e.g., where the first
            element corresponds to the class predictions (probabilities,
            logits, etc.) and the second element is a tensor of embeddings
            (cf. `return_embeddings` in `predict_proba`).
            - If False, the criterion is used as is and must be able to process
            the full output `module.forward`.
        neural_net_param_dict : dict, default=None
            Additional arguments for `skorch.net.NeuralNet`. If
            `neural_net_param_dict` is `None`, no additional arguments are added.
        sample_dtype : str or type, default=None
            The type or typecode all data is casted to. If `sample_dtype` is None,
            the datatype is preserved.
        missing_label : scalar or string or np.nan or None, default=np.nan
            Value to represent a missing label.
        random_state : int or RandomState instance or None, default=None
            Determines random number for 'predict' method. Pass an int for
            reproducible results across multiple method calls.

        References
        ----------
        .. [1] Marian Tietz, Thomas J. Fan, Daniel Nouri, Benjamin Bossan, and
           skorch Developers. skorch: A scikit-learn compatible neural network
           library that wraps PyTorch, July 2017.
        """

        def __init__(
            self,
            module,
            criterion=nn.NLLLoss,
            filter_criterion_input=True,
            neural_net_param_dict=None,
            sample_dtype=None,
            missing_label=MISSING_LABEL,
            random_state=None,
        ):
            super(SkorchRegressor, self).__init__(
                missing_label=missing_label,
                random_state=random_state,
            )
            self.module = module
            self.criterion = criterion
            self.filter_criterion_input = filter_criterion_input
            self.neural_net_param_dict = neural_net_param_dict
            self.sample_dtype = sample_dtype

        def fit(self, X, y, **fit_params):
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
                `skorch.net.NeuralNet`.

            Returns
            -------
            self: SkorchClassifier,
                The SkorchClassifier is fitted on the training data.
            """
            return self._fit("fit", X, y, **fit_params)

        def partial_fit(self, X, y, **fit_params):
            """Fit the module without re-initialization.

            If the module was already initialized, by calling partial_fit, the
            module will not be re-initialized again.

            Parameters
            ----------
            X : matrix-like, shape (n_samples, n_features)
                Training data set, usually complete, i.e. including the labeled
                and unlabeled samples
            y : array-like of shape (n_samples, )
                Labels of the training data set (possibly including unlabeled
                ones indicated by self.missing_label)
            fit_params : dict-like
                Further parameters as input to the 'partial_fit' method of the
                `skorch.net.NeuralNet`.

            Returns
            -------
            self: SkorchClassifier,
                The SkorchClassifier is fitted on the training data.
            """
            return self._fit("partial_fit", X, y, **fit_params)

        def _fit(self, fit_function, X, y, **fit_params):
            """Initialize and fit the module.

            If the module was already initialized, by calling fit, the module
            will be re-initialized
            (unless `neural_net_param_dict["warm_start"]=True`).

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
            # Initialize module, if not done yet
            # or if `fit` is called, while `warm_start` is deactivated.
            if not hasattr(self, "neural_net_") or (
                fit_function == "fit" and not self.neural_net_.warm_start
            ):
                X, y = self.initialize(X=X, y=y, enforce_check_X_y=True)
            else:
                X, y, _ = self._validate_data(
                    X=X, y=y, check_X_dict=self.check_X_dict_
                )

            # Fit only on labeled data.
            is_lbld = is_labeled(y, missing_label=self.missing_label_)
            if np.sum(is_lbld) > 0:
                X_lbld = X[is_lbld]
                y_lbld = y[is_lbld].astype(np.float32).reshape(-1, 1)
                self.neural_net_.partial_fit(X_lbld, y_lbld, **fit_params)
            return self

        def predict(self, X, return_embeddings=False, return_std=False):
            """Return probability estimates for the test data X.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Test samples.
            return_embeddings : boolean, default=False
                If `return_embeddings=True`, the forward method of the neural
                network module is expected to return multiple outputs,
                of which the first element corresponds to the class predictions
                (probabilities, logits, etc.) and the second element is a
                tensor of embeddings learned by the neural network.

            Returns
            -------
            P : numpy.ndarray of shape (n_samples, classes)
                The class probabilities of the test samples. Classes are
                ordered according to `self.classes_`.
            X_embed : numpy.ndarray of shape (n_samples, ...)
                Sample embeddings, which are only returned if
                `return_embeddings=True`.
            """
            # Initialize module, if not done yet.
            if not hasattr(self, "neural_net_"):
                self.initialize()

            # Check input parameters.
            X = check_array(X, **self.check_X_dict_)
            check_n_features(
                self, X, reset=not hasattr(self, "n_features_in_")
            )
            check_scalar(
                return_embeddings, name="return_embeddings", target_type=bool
            )

            if not return_embeddings:
                out = self.neural_net_.predict(X)
            else:
                out = self.neural_net_.forward(X)
                if not isinstance(out, tuple):
                    raise ValueError(
                        "`return_embeddings=True` only works when module is"
                        "expected to return multiple outputs, of which the"
                        "first element corresponds to the class predictions"
                        "(probabilities, logits, etc.) and the second element"
                        "is a tensor of embeddings."
                    )
                P = self.neural_net_._get_predict_nonlinearity()(out[0])
                P = to_numpy(P)
                X_embed = to_numpy(out[1])
                out = (P, X_embed)

            return out

        def initialize(self, X=None, y=None, enforce_check_X_y=False):
            """
            Initialize the internal ``sklearn`` wrapper built on ``skorch``.

            Optionally validates input data and instantiates
            ``self.neural_net_`` (a ``skorch.NeuralNet``) with the configured
            module, criterion, and parameters. If any data is provided (``X``
            or ``y``), the cleaned pair is returned. Otherwise, the estimator
            instance is returned.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...), default=None
                Input samples used for optional validation. If provided
                (together with or without ``y``), both ``X`` and ``y`` are
                passed to ``self._validate_data`` with ``check_X_dict_``.
            y : array-like of shape (n_samples,), default=None
                Target values used for optional validation. See notes on shape
                in the estimator's data interface.
            enforce_check_X_y : bool, default=False
                If ``True``, run input validation even when both ``X`` and ``y``
                are ``None``. Validation also runs automatically when either
                ``X`` or ``y`` is provided.

            Returns
            -------
            self : object
                The estimator instance, if no data was provided
                (both ``X`` and ``y`` are ``None``).
            X_out, y_out : ndarray
                Cleaned/validated versions of ``X`` and ``y``, returned as a
                tuple when any data was provided.
            """
            # Check input samples and class labels.
            self.check_X_dict_ = {
                "ensure_min_samples": 0,
                "ensure_min_features": 0,
                "allow_nd": True,
                "dtype": self.sample_dtype,
            }
            if enforce_check_X_y or X is not None or y is not None:
                X, y, _ = self._validate_data(
                    X=X, y=y, check_X_dict=self.check_X_dict_
                )

            # Check `__init__` parameters.
            if self.neural_net_param_dict is None:
                self.neural_net_param_dict_ = {}
            else:
                self.neural_net_param_dict_ = self.neural_net_param_dict
            check_type(
                self.neural_net_param_dict_, "neural_net_param_dict", dict
            )

            # Check criterion for loss computation.
            criterion = self.criterion
            if self.filter_criterion_input:
                criterion = make_criterion_tuple_aware(criterion)

            self.neural_net_ = NeuralNet(
                module=self.module,
                criterion=criterion,
                **self.neural_net_param_dict_,
            ).initialize()

            if X is not None or y is not None:
                return X, y
            else:
                return self
