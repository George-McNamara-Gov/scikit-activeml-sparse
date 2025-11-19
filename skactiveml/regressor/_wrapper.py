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
        """Partially fitting the model using X as training data and y as labels.

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
        except Exception:
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
                    "SklearnNormalRegressors require the Regressor from Sklearn to accept 'return_std'"
                ) from e


if successful_skorch_torch_import:

    class SkorchRegressor(SkactivemlRegressor, SkorchMixin):
        """SkorchRegressor

        Implement a wrapper class, to make it possible to use `torch` with
        `skactiveml`. This is achieved by providing a wrapper around `torch`
        that has a `skactiveml` interface and also be able to handle missing
        labels. This wrapper is based on the open-source library `skorch` [1]_.

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
            e.g., `nn.MSELoss`, work with implementations of the
            `module.forward` methods outputting tuples, e.g., where the first
            element corresponds to the target predictions and the second element
            is a tensor of embeddings (cf. `return_embeddings` in `predict`).
            - If False, the criterion is used as is and must be able to process
            the full output `module.forward`.
        neural_net_param_dict : dict, default=None
            Additional arguments for `skorch.net.NeuralNet`. If
            `neural_net_param_dict` is `None`, no additional arguments are
            added.
        sample_dtype : str or type, default=None
            The type or typecode all data is casted to. If `sample_dtype` is
            None, the datatype is preserved.
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
            criterion=nn.NLLLoss,
            filter_criterion_input=True,
            neural_net_param_dict=None,
            sample_dtype=None,
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
            self.filter_criterion_input = filter_criterion_input
            self.neural_net_param_dict = neural_net_param_dict
            self.include_unlabeled_samples = include_unlabeled_samples
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
                y_pred = self.neural_net_._get_predict_nonlinearity()(out[0])
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
            ``skorch.NeuralNet``.

            Returns
            -------
            module : torch.nn.Module or Callable[..., torch.nn.Module]
                The classification/regression module or a factory returning it.
            criterion : Callable or torch.nn.Module
                The loss used by the internal network. May be pre-wrapped to
                handle tuple targets or other conventions.
            net_params : dict
                Additional keyword arguments for ``skorch.NeuralNet``
                construction (e.g., ``optimizer``, ``callbacks``, ``device``).
                Empty if none.
            """
            criterion = self.criterion
            if self.filter_criterion_input:
                criterion = make_criterion_tuple_aware(criterion)
            return self.module, criterion, self.neural_net_param_dict or {}

        def _validate_data_kwargs(self):
            """
            Return kwargs forwarded to ``_validate_data``.

            Returns
            -------
            kwargs : dict or None
                Keyword arguments consumed by ``_validate_data``.
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
                Training samples or ``None`` if none exist.
            y_train : ndarray or None
                Training labels or ``None`` if none exist.
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

    class SkorchProbabilisticRegressor(
        SkorchRegressor, ProbabilisticRegressor
    ):
        """SkorchProbabilisticRegressor

        Implement a wrapper class, to make it possible to use `torch` with
        `skactiveml`. This is achieved by providing a wrapper around `torch`
        that has a `skactiveml` interface and also be able to handle missing
        labels. This wrapper is based on the open-source library `skorch` [1]_.
        In contrast to the `SkorchRegressor`, this class expects the neural
        network module to output standard deviation estimates next to the
        target estimates and the optional sample embeddings.

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
            e.g., `nn.MSELoss`, work with implementations of the
            `module.forward` methods outputting tuples, e.g., where the first
            element corresponds to the target predictions and the second element
            is a tensor of embeddings (cf. `return_embeddings` in `predict`).
            - If False, the criterion is used as is and must be able to process
            the full output `module.forward`.
        neural_net_param_dict : dict, default=None
            Additional arguments for `skorch.net.NeuralNet`. If
            `neural_net_param_dict` is `None`, no additional arguments are
            added.
        sample_dtype : str or type, default=None
            The type or typecode all data is casted to. If `sample_dtype` is
            None, the datatype is preserved.
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
            super(SkorchProbabilisticRegressor, self).__init__(
                missing_label=missing_label,
                random_state=random_state,
                module=module,
                criterion=criterion,
                filter_criterion_input=filter_criterion_input,
                neural_net_param_dict=neural_net_param_dict,
                sample_dtype=sample_dtype,
            )

        def predict_target_distribution(self, X, return_embeddings=False):
            """Returns the predicted target distribution conditioned on the
            test samples `X`. The module is expected to return at least two
            outputs, of which the first element corresponds to the target
            predictions and the second element to the standard deviation
            estimates. Optionally, the third element is a tensor of
            sample embeddings.

            Parameters
            ----------
            X :  array-like, shape (n_samples, n_features)
                Input samples.

            Returns
            -------
            dist : scipy.stats._distn_infrastructure.rv_frozen
                The distribution of the targets at the test samples.

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

            out = self.neural_net_.forward(X)
            if not isinstance(out, tuple):
                raise ValueError(
                    "`The module is expected to return at least two outputs,"
                    "of which the first element corresponds to the target"
                    "predictions, the second element to the standard deviation"
                    "estimates Optionally, the third element is a tensor of "
                    "sample embeddings."
                )
            if return_embeddings and len(out) != 3:
                raise ValueError(
                    "`return_embeddings=True` only works when the module "
                    "returns three outputs, of which the "
                    "first element corresponds to the target predictions,"
                    "the second element to the standard deviation estimates, "
                    "and the third element is a tensor of sample embeddings."
                )
            y_pred = self.neural_net_._get_predict_nonlinearity()(out[0])
            y_pred = to_numpy(y_pred).ravel()
            X_embed = to_numpy(out[1])
            out = (y_pred, X_embed)

            return out
