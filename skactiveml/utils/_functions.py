import inspect
import sys

from copy import deepcopy
from types import MethodType
from makefun import with_signature

successful_skorch_torch_import = False
try:
    from torch import nn

    successful_skorch_torch_import = True
except ImportError:  # pragma: no cover
    pass


def call_func(
    f_callable, only_mandatory=False, ignore_var_keyword=False, **kwargs
):
    """Calls a function with the given parameters given in `kwargs`, if they
    exist as parameters in `f_callable`.

    Parameters
    ----------
    f_callable : callable
        The function or object that is to be called.
    only_mandatory : boolean, default=False
        If `True`, only mandatory parameters are set.
    ignore_var_keyword : boolean, default=False
        If `False`, all kwargs are passed when `f_callable` uses a parameter
        that is of kind `Parameter.VAR_KEYWORD`, i.e., `**kwargs`. For further
        reference see the `inspect` package.
    kwargs : kwargs
        All parameters that could be used for calling f_callable.

    Returns
    -------
    f_callable_result : return type of `f_callable`
        The return value of f_callable.
    """
    params = inspect.signature(f_callable).parameters
    param_keys = params.keys()
    if only_mandatory:
        param_keys = list(
            filter(lambda k: params[k].default == params[k].empty, param_keys)
        )

    has_var_keyword = any(
        filter(lambda p: p.kind == p.VAR_KEYWORD, params.values())
    )
    if has_var_keyword and not ignore_var_keyword and not only_mandatory:
        vars = kwargs
    else:
        vars = dict(filter(lambda e: e[0] in param_keys, kwargs.items()))

    return f_callable(**vars)


def match_signature(wrapped_obj_name, func_name):
    """A decorator that matches the signature to a given method from a
    reference and hides it when the reference object does not have the wrapped
    function. This is especially helpful for wrapper classes whose functions
    should appear. This decorator is heavily inspired by the `available_if`
    decorator from `sklearn.utils.metaestimators`.

    Parameters
    ----------
    wrapped_obj_name : str
        The name of the object that will be wrapped.
    func_name : str
        The name of the function that will be wrapped.

    Returns
    -------
    wrapped_obj : callable
        The wrapped function.
    """

    class _MatchSignatureDescriptor:
        """_MatchSignatureDescriptor

        A descriptor that allows a wrapper to clone the signature of a
        method `func_name` from the wrapped object `wrapped_obj_name`.
        Furthermore, this extends upon the conditional property as implemented
        in `available_if` from `sklearn.utils.metaestimators`.

        Parameters
        ----------
        fn: MethodType
            The method that should be wrapped.
        wrapped_obj_name: str
            The name of the wrapped object within the wrapper class.
        func_name : str
            The method name of the function that should be wrapped.
        """

        def __init__(self, fn, wrapped_obj_name, func_name):
            self.fn = fn
            self.wrapped_obj_name = wrapped_obj_name
            self.func_name = func_name
            self.__name__ = func_name

        def __get__(self, obj, owner=None):
            """Wrap the method specified in `self.func_name` from the wrapped
            object `self.wrapped_obj_name` such that the signature will be the
            same.

            Parameters
            ----------
            obj : object
                The wrapper object. This parameter will be None, if the method
                is accessed via the class and not an instantiated object.
            owner : class, default=None
                The wrapper class.

            Returns
            -------
            The wrapped method.
            """
            if obj is not None:
                reference_object = getattr(obj, self.wrapped_obj_name)
                if not hasattr(reference_object, self.func_name):
                    raise AttributeError(
                        f"This {reference_object} has"
                        f" no method {self.func_name}."
                    )
                reference_function = getattr(reference_object, self.func_name)

                # Check if the refrenced function is a method of that object.
                # If it is, use `__func__` to copy the name of the `self`
                # parameter
                # If it is not, (i.e. it has been added as a lambda function),
                # add a provisory self argument in the first position
                if hasattr(reference_function, "__func__"):
                    new_sig = inspect.signature(reference_function.__func__)
                else:
                    reference_sig = inspect.signature(reference_function)
                    new_parameters = list(reference_sig.parameters.values())
                    new_parameters.insert(
                        0,
                        inspect.Parameter(
                            "self",
                            kind=inspect._ParameterKind.POSITIONAL_OR_KEYWORD,
                        ),
                    )
                    new_sig = inspect.Signature(
                        new_parameters,
                        return_annotation=reference_sig.return_annotation,
                    )

                # create a wrapper with the new signature and the correct
                # function name
                function_decorator = with_signature(
                    func_signature=new_sig,
                    func_name=self.fn.__name__,
                )
                fn = function_decorator(self.fn)
                out = MethodType(fn, obj)
            else:
                out = self.fn

            return out

    return lambda fn: _MatchSignatureDescriptor(
        fn, wrapped_obj_name, func_name=func_name
    )


if successful_skorch_torch_import:

    def make_criterion_tuple_aware(criterion, criterion_input_index=0):
        """
        Create a loss *class* (or wrap an existing instance) that selects part
        of a model's (possibly tuple-valued) output before passing it to the
        criterion. This utility generates (and caches) a dynamic subclass named
        `TupleAware<LossName>Idx<...>` of the given base loss class. The
        subclass overrides `forward` so that, if the `input` argument is a
        tuple (e.g., `(logits, embeddings, ...)`), only the element(s)
        specified by `criterion_input_index` are passed to the base class's
        `forward`. If `input` is not a tuple, it is forwarded unchanged.

        Parameters
        ----------
        criterion : type or nn.Module
            - Either a loss **class** (subclass of :class:`torch.nn.Module`),
              e.g., `nn.CrossEntropyLoss`, or
            - a loss **instance**, e.g. `nn.CrossEntropyLoss()`.

        criterion_input_index : int or array-like of int or None, default=0
            Index or indices of the output of `module.forward` to pass as the
            `input` argument of the criterion:

            - If an `int`, `input[criterion_input_index]` is used.
            - If an array-like of int, the selected elements are packed into a
              tuple `(input[i_1], ..., input[i_k])`.
            - If `None`, the full `input` is passed unchanged (wrapping is
              effectively a no-op for tuples).

        Returns
        -------
        type or nn.Module
            - If `criterion` is a **class**, returns the generated subclass
              `TupleAware<LossName>Idx<...>`.
            - If `criterion` is an **instance**, returns a **new** instance
              (deep copy) whose class is that subclass. The original instance
              is not modified.
        """
        import numbers
        from collections.abc import Sequence

        if isinstance(criterion, nn.Module):
            base_cls = criterion.__class__
        elif issubclass(criterion, nn.Module):
            base_cls = criterion
        else:
            raise TypeError(
                "criterion must be an nn.Module subclass or instance."
            )

        # Normalize the selector to something predictable for both logic
        # & naming.
        def _normalize_index(sel):
            msg = (
                "criterion_input_index must be an int, "
                "array-like of int, or None."
            )
            if sel is None:
                return None

            def _is_int_like(x):
                return isinstance(x, numbers.Integral) and not isinstance(
                    x, bool
                )

            if _is_int_like(sel):
                return int(sel)
            if isinstance(sel, Sequence) and not isinstance(sel, (str, bytes)):
                if not all(_is_int_like(i) for i in sel):
                    raise TypeError(msg)
                return tuple(int(i) for i in sel)
            raise TypeError(msg)

        selector = _normalize_index(criterion_input_index)

        if selector is None:
            idx_key = "All"
        elif isinstance(selector, int):
            idx_key = str(selector)
        else:  # tuple of ints
            idx_key = "_".join(str(i) for i in selector)

        cls_name = f"TupleAware{base_cls.__name__}Idx{idx_key}"

        # Build the wrapper `forward`.
        def forward(self, input, target, *args, **kwargs):
            x = input
            if isinstance(x, tuple):
                if selector is None:
                    # pass full tuple unchanged
                    x = x
                elif isinstance(selector, int):
                    x = x[selector]
                else:
                    # selector is a tuple of indices
                    x = tuple(x[i] for i in selector)
            return base_cls.forward(self, x, target, *args, **kwargs)

        # Keep signature/tool-tips identical.
        forward.__signature__ = inspect.signature(base_cls.forward)
        forward.__doc__ = base_cls.forward.__doc__

        # Create/reuse subclass and expose it under *this* module.
        mod = sys.modules[__name__]
        TupleAwareCls = getattr(mod, cls_name, None)
        if TupleAwareCls is None:
            TupleAwareCls = type(
                cls_name,
                (base_cls,),
                {
                    "forward": forward,
                    "__module__": mod.__name__,
                    "__criterion_input_index__": selector,
                },
            )
            setattr(mod, cls_name, TupleAwareCls)  # makes pickle happy

        if isinstance(criterion, nn.Module):
            wrapped = deepcopy(criterion)
            object.__setattr__(wrapped, "__class__", TupleAwareCls)
            return wrapped
        else:
            return TupleAwareCls
