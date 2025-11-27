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
    check_scalar,
    check_type,
    MISSING_LABEL,
)

successful_skorch_torch_import = False
try:
    from torch import nn
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
    include_unlabeled_samples : bool, default=False
        - If `False`, only labeled samples are passed to the `fit` method of
          the `estimator`.
        - If `True`, all samples including the unlabeled ones are passed to
          the `fit` method of the `estimator`. Ensure that your `estimator`
          is able to handle unlabeled samples marked by `missing_label`.
          Otherwise, `missing_label` is interpreted as a regular target value.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.
    """

    def __init__(
        self,
        estimator,
        include_unlabeled_samples=False,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            random_state=random_state, missing_label=missing_label
        )
        self.estimator = estimator
        self.include_unlabeled_samples = include_unlabeled_samples

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
        """Partially fitting the model using X as training data and y as
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
        check_type(
            self.include_unlabeled_samples, "include_unlabeled_samples", bool
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
        if self.include_unlabeled_samples:
            is_included = np.full_like(y, True, dtype=bool)
        else:
            is_included = is_lbld
        X_train = X[is_included]
        y_train = y[is_included]
        estimator_params = dict(fit_kwargs) if fit_kwargs is not None else {}

        if sample_weight is not None:
            estimator_params["sample_weight"] = sample_weight[is_included]

        self._label_mean = np.mean(y[is_lbld]) if np.sum(is_lbld) > 0 else 0
        self._label_std = np.std(y[is_lbld]) if np.sum(is_lbld) > 1 else 1
        self.estimator_ = deepcopy(self.estimator)
        try:
            attrgetter(fit_function)(self.estimator_)(
                X_train, y_train, **estimator_params
            )
            self.is_fitted_ = True
        except Exception as e:
            warnings.warn(
                f"The 'estimator' could not be fitted because of"
                f" '{e}'. Therefore, the empirical label mean "
                f"`_label_mean={self._label_mean}` and the "
                f"empirical label standard deviation "
                f"`_label_std={self._label_std}` will be used to make "
                f"predictions."
            )
            self.is_fitted_ = False

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
        if self.is_fitted_:
            return self.estimator_.predict(X, **predict_kwargs)

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
        if hasattr(self, "is_fitted_"):
            return True

        try:
            check_is_fitted(self.estimator)
        except NotFittedError:
            return False

        # set attributes that would be set by the fit function
        self.is_fitted_ = True
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

    The wrapped regressor of sklearn needs `return_std` as a keyword argument
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

    def _fit(self, fit_function, X, y, sample_weight, **fit_kwargs):
        if (
            hasattr(self.estimator, "predict")
            and "return_std"
            not in inspect.signature(self.estimator.predict).parameters.keys()
            and inspect.getfullargspec(self.estimator.predict).varkw is None
        ):
            raise ValueError(
                f"`{self.estimator}` must have keyword argument"
                f"`return_std` for predict."
            )

        return super()._fit(fit_function, X, y, sample_weight, **fit_kwargs)

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

        try:
            loc, scale = SklearnRegressor.predict(self, X, return_std=True)
            return norm(loc=loc, scale=scale)
        except TypeError as e:
            if (
                "predict() got an unexpected keyword argument 'return_std'"
                in str(e)
            ):
                raise ValueError(
                    "SklearnNormalRegressors require the Regressor from"
                    "`sklearn` to accept `return_std`."
                ) from e


if successful_skorch_torch_import:

    class SkorchRegressor(SkactivemlRegressor, SkorchMixin):
        """SkorchRegressor

        Implement a wrapper class, to make it possible to use `torch` with
        `skactiveml`. This is achieved by providing a wrapper around `torch`
        that has a `skactiveml` interface and also be able to handle missing
        labels. This wrapper is based on the open-source library `skorch` [1]_.

        Notes
        -----
        Adjust your `criterion` to match the outputs of your `nn.Module`.
        For example:
        - If you use the default regression loss `criterion=nn.MSELoss`,
          then your module is typically expected to output raw, unbounded
          continuous predictions. In this default case, the final layer usually
          has no activation function, and `predict_nonlinearity` should be set
          to `nn.Identity()`. Alternatively, `predict_nonlinearity=None` is
          also supported, but only for this default setting with
          `criterion=nn.MSELoss`.
        - If your module outputs transformed values (e.g., log-transformed
          targets or values passed through a bounded activation such as
          `torch.sigmoid`), then your `criterion` must be defined on the same
          transformed scale. To obtain predictions on the original target
          scale, you can set `predict_nonlinearity` to the corresponding
          inverse transformation (for example, `torch.exp` if the network
          outputs log-targets).

        Parameters
        ----------
        module : torch module (class or instance)
            A PyTorch :class:`~torch.nn.Module`. In general, the uninstantiated
            class should be passed, although instantiated modules will also
            work.
        criterion : torch.nn.Module.__class__, default=torch.nn.MSELoss
            The uninitialized criterion (loss) used to optimize the module. By
            default, `torch.nn.MSELoss` is used as criterion.
        predict_nonlinearity : Callable, default=None
            When calling `predict`, this is the nonlinearity
            to be applied to the output of your module's forward method or its
            first element, if the output is a tuple. In the default case,
            we set `predict_nonlinearity=torch.nn.Identity()`.
        criterion_input_index : int or array-like of int, default=0
            Index or indices of the output of `module.forward` that are
            passed to the loss / criterion. Use this when `module.forward`
            returns a tuple, e.g. `(raw outputs, embeddings, ...)`, but the
            criterion expects a single tensor input such as a numerical array
            (e.g. `nn.MSELoss`).

            - If an `int`, the corresponding element of the `module.forward`
              output is passed to the criterion (e.g. `0` to use only the
              first element of the `module.forward` output).
            - If an array-like of `int`, the selected elements are packed
              into a tuple and passed to the criterion in that order.
            - If `None`, the full output of `module.forward` is passed
              unchanged.
        neural_net_param_dict : dict, default=None
            Additional arguments for `skorch.net.NeuralNet`. If
            `neural_net_param_dict` is `None`, no additional arguments are
            added.
        sample_dtype : str or type, default=np.float32
            Dtype to which input samples are cast inside the estimator. If set
            to `None`, the input dtype is preserved.
        include_unlabeled_samples : bool, default=False
            - If `False`, only labeled samples are passed to the `fit` method
              of the `estimator`.
            - If `True`, all samples including the unlabeled ones are passed to
              the `fit` method of the `estimator`. Ensure that the `criterion`
              is able to handle unlabeled samples marked by `missing_label`.
              Otherwise, `missing_label` is interpreted as a regular target
              value.
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
            criterion=nn.MSELoss,
            predict_nonlinearity=None,
            criterion_input_index=0,
            neural_net_param_dict=None,
            sample_dtype=np.float32,
            include_unlabeled_samples=False,
            missing_label=MISSING_LABEL,
            random_state=None,
        ):
            super(SkorchRegressor, self).__init__(
                missing_label=missing_label,
                random_state=random_state,
            )
            self.module = module
            self.criterion = criterion
            self.predict_nonlinearity = predict_nonlinearity
            self.criterion_input_index = criterion_input_index
            self.neural_net_param_dict = neural_net_param_dict
            self.include_unlabeled_samples = include_unlabeled_samples
            self.sample_dtype = sample_dtype

        def fit(self, X, y, **fit_params):
            """Initialize and fit the module.

            If the module was already initialized, by calling fit, the module
            will be re-initialized (unless `warm_start` is True).

            Parameters
            ----------
            X : matrix-like, shape (n_samples, n_features)
                Training data set, usually complete, i.e. including the labeled
                and unlabeled samples
            y : array-like of shape (n_samples,)
                Labels of the training data set (possibly including unlabeled
                ones indicated by self.missing_label)
            fit_params : dict-like
                Further parameters as input to the 'fit' method of the
                `skorch.net.NeuralNet`.

            Returns
            -------
            self: SkorchRegressor,
                The SkorchRegressor is fitted on the training data.
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
            self: SkorchRegressor,
                The SkorchRegressor is fitted on the training data.
            """
            return self._fit("partial_fit", X, y, **fit_params)

        def predict(self, X, return_embeddings=False):
            """Return probability estimates for the test data X.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Test samples.
            return_embeddings : boolean, default=False
                If `return_embeddings=True`, the forward method of the neural
                network module is expected to return multiple outputs,
                of which the first element corresponds to the target
                predictions and the second element is a tensor of embeddings
                learned by the neural network.

            Returns
            -------
            y_pred : numpy.ndarray of shape (n_samples,)
                The target predictions for the test samples.
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
                out = self.neural_net_.predict(X).ravel()
            else:
                out = self.neural_net_.forward(X)
                if not isinstance(out, tuple):
                    raise ValueError(
                        "`return_embeddings=True` only works when module is "
                        "expected to return multiple outputs, of which the "
                        "first element corresponds to the target predictions "
                        "and the second element is a tensor of embeddings."
                    )
                y_pred = self._predict_nonlinearity(out[0])
                y_pred = to_numpy(y_pred).ravel()
                X_embed = to_numpy(out[1])
                out = (y_pred, X_embed)

            return out

        def _net_parts(self, X, y):
            """
            Assemble and validate network components.

            Implementations should perform any optional checks or normalization
            of constructor/init parameters (e.g., shape consistency, dtype
            checks, wrapping criteria), then return the ready-to-use pieces for
            `skorch.NeuralNet`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...), default=None
                Input samples for optional validation.
            y : array-like of shape (n_samples, ...), default=None
                Target values for optional validation.

            Returns
            -------
            module : torch.nn.Module.__class__ or torch.nn.Module
                A PyTorch `torch.nn.Module`. In general, the uninstantiated
                class should be passed, although instantiated modules will also
                work.
            criterion : torch.nn.Module.__class__
                The uninitialized criterion (loss) used to optimize the module.
            predict_nonlinearity : Callable
                The nonlinearity to be applied to the prediction.
            params : dict
                Keyword arguments (excluding `predict_non_linearity`) for
                `skorch.NeuralNet` construction. Must be a mapping and may be
                empty.
            """
            criterion = self.criterion
            if (
                self.criterion is not nn.MSELoss
                and not isinstance(self.criterion, nn.MSELoss)
            ) and self.predict_nonlinearity is None:
                raise ValueError(
                    "`predict_nonlinearity` must not be None, "
                    "if `criterion` is not torch.nn.MSELoss."
                )
            if self.criterion_input_index is not None:
                criterion = make_criterion_tuple_aware(
                    criterion, criterion_input_index=self.criterion_input_index
                )
            if self.predict_nonlinearity is None:
                self._predict_nonlinearity = nn.Identity()
            else:
                self._predict_nonlinearity = self.predict_nonlinearity
            return (
                self.module,
                criterion,
                self._predict_nonlinearity,
                self.neural_net_param_dict or {},
            )

        def _validate_data_kwargs(self):
            """
            Return kwargs forwarded to `_validate_data`.

            Returns
            -------
            kwargs : dict or None
                Keyword arguments consumed by `_validate_data`.
            """
            self.check_X_dict_ = {
                "ensure_min_samples": 0,
                "ensure_min_features": 0,
                "allow_nd": True,
                "dtype": self.sample_dtype,
            }
            check_type(
                self.include_unlabeled_samples,
                "include_unlabeled_samples",
                bool,
            )
            return {"check_X_dict": self.check_X_dict_}

        def _return_training_data(self, X, y):
            """
            Return only samples and labels required for training.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Input samples.
            y : array-like of shape (n_samples, ...)
                Targets with unlabeled entries following the subclass'
                convention.

            Returns
            -------
            X_train : ndarray or None
                Training samples or `None` if none exist.
            y_train : ndarray or None
                Training labels or `None` if none exist.
            """
            X_train, y_train = None, None
            if self.include_unlabeled_samples:
                is_included = np.full_like(y, fill_value=True, dtype=bool)
            else:
                is_included = is_labeled(y, missing_label=self.missing_label_)
            if np.sum(is_included) > 0:
                X_train = X[is_included]
                y_train = y[is_included]
            if y_train is not None:
                y_train = y_train.astype(np.float32, copy=True).reshape(-1, 1)
            return X_train, y_train
