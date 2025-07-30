import unittest

import inspect

from skactiveml.utils import call_func, match_signature

successful_skorch_torch_import = False
try:
    import torch
    from torch import nn
    from skactiveml.utils import make_criterion_tuple_aware

    successful_skorch_torch_import = True
except ImportError:
    pass  # pragma: no cover


class TestFunctions(unittest.TestCase):
    def test_call_func(self):
        def dummy_function(a, b=2, c=3):
            return a * b * c

        result = call_func(dummy_function, a=2, b=5, c=5)
        self.assertEqual(result, 50)
        result = call_func(dummy_function, only_mandatory=True, a=2, b=5, c=5)
        self.assertEqual(result, 12)

        # test kwargs
        def sum_all(*args, **kwargs):
            s = 0
            for v in args + tuple(kwargs.values()):
                s += v
            return s

        def test_func_0(**kwargs):
            return sum_all(**kwargs)

        def test_func_1(arg1, **kwargs):
            return sum_all(arg1, **kwargs)

        def test_func_2(kwarg1=0, **kwargs):
            return sum_all(kwarg1=kwarg1, **kwargs)

        result = call_func(test_func_0, arg1=1, arg2=2, arg3=3)
        self.assertEqual(result, 6)

        result = call_func(test_func_1, arg1=1, arg2=2, arg3=3)
        self.assertEqual(result, 6)

        result = call_func(test_func_2, kwarg1=1, arg2=2, arg3=3)
        self.assertEqual(result, 6)

        result = call_func(
            test_func_1, only_mandatory=True, arg1=1, arg2=2, arg3=3
        )
        self.assertEqual(result, 1)

        result = call_func(
            test_func_2, only_mandatory=True, kwarg1=1, arg2=2, arg3=3
        )
        self.assertEqual(result, 0)

        result = call_func(
            test_func_2, ignore_var_keyword=True, kwarg1=1, arg2=2, arg3=3
        )
        self.assertEqual(result, 1)

    def test_match_signature(self):
        class DummyA:
            def __init__(self, dummy_b):
                self.dummy_b = dummy_b

            # test case where c has to be in kwargs to not fail
            @match_signature("dummy_b", "test_me")
            def test_me(self, a, b=None, **kwargs):
                return self.dummy_b.test_me(a=a, b=b, **kwargs)

            # test case without kwargs
            @match_signature("dummy_b", "test_me_alt")
            def test_me_alt(self, a, b=None, **kwargs):
                return self.dummy_b.test_me_alt(a=a, **kwargs)

            # test case without kwargs
            @match_signature("dummy_b", "test_me_hidden")
            def test_me_hidden(self, a, b=None, **kwargs):
                return self.dummy_b.test_me_alt(a=a, **kwargs)

            # test case with type hinting
            @match_signature("dummy_b", "test_me_hint")
            def test_me_hint(self, a, b=None, **kwargs):
                return self.dummy_b.test_me_alt(a=a, **kwargs)

        class DummyB:
            def test_me(self, a, c, **kwargs):
                output = {"a": a, "c": c}
                output.update(kwargs)
                return output

            def test_me_alt(self, a, c):
                output = {"a": a, "c": c}
                return output

            def test_me_hint(self, a: int, c: str) -> dict:
                output = {"a": a, "c": c}
                return output

        dummy_b = DummyB()
        dummy_a = DummyA(dummy_b)

        # test default working case
        kwargs_1 = {
            "a": "p1",
            "b": "p2",
            "c": "p3",
            "d": "p4",
        }
        output_1 = dummy_a.test_me(**kwargs_1)
        self.assertEqual(kwargs_1, output_1)

        # test for equal signature
        sig_a_test_me = inspect.signature(dummy_a.test_me).parameters
        sig_b_test_me = inspect.signature(dummy_b.test_me).parameters
        self.assertEqual(sig_a_test_me, sig_b_test_me)

        # test non working case with missing c
        kwargs_2 = {
            "a": "p1",
            "b": "p2",
            "d": "p4",
        }
        self.assertRaises(TypeError, dummy_a.test_me, **kwargs_2)

        # test for equal signature
        sig_a_test_me_alt = inspect.signature(dummy_a.test_me_alt).parameters
        sig_b_test_me_alt = inspect.signature(dummy_b.test_me_alt).parameters
        self.assertEqual(sig_a_test_me_alt, sig_b_test_me_alt)

        kwargs_3 = {
            "a": "p1",
            "c": "p2",
        }
        dummy_a.test_me_alt(**kwargs_3)

        kwargs_3 = {
            "a": "p1",
            "b": "p2",
            "c": "p3",
        }
        self.assertRaises(TypeError, dummy_a.test_me_alt, **kwargs_3)

        # test for hiding methods that the wrapped object does not have
        self.assertFalse(hasattr(dummy_a, "test_me_hidden"))

        sig_a_test_me_hint = inspect.signature(dummy_a.test_me_hint)
        sig_b_test_me_hint = inspect.signature(dummy_b.test_me_hint)

        self.assertEqual(
            sig_a_test_me_hint.return_annotation,
            sig_b_test_me_hint.return_annotation,
        )

        param_iterator = zip(
            sig_a_test_me_hint.parameters.items(),
            sig_b_test_me_hint.parameters.items(),
        )
        for (name_a, param_a), (name_b, param_b) in param_iterator:
            self.assertEqual(name_a, name_b)
            self.assertEqual(param_a.kind, param_b.kind)
            self.assertEqual(param_a.annotation, param_a.annotation)
            self.assertEqual(param_a.default, param_a.default)

    if successful_skorch_torch_import:

        def test_make_criterion_tuple_aware(self):
            input = torch.randn(20, 10)
            target = torch.randint(10, (20,))
            TupleAwareCrossEntropy = make_criterion_tuple_aware(
                nn.CrossEntropyLoss
            )
            tuple_aware_cross_entropy = make_criterion_tuple_aware(
                nn.CrossEntropyLoss(ignore_index=10)
            )

            # Test failing configurations.
            loss = nn.CrossEntropyLoss()(input, target)
            self.assertRaises(
                TypeError, nn.CrossEntropyLoss(), (input,), target
            )
            self.assertRaises(
                TypeError, nn.CrossEntropyLoss(), (input, input), target
            )
            self.assertRaises(
                TypeError, nn.CrossEntropyLoss(), [input, input], target
            )

            # Test tuple-aware criterion for class as input.
            sig_1 = inspect.signature(TupleAwareCrossEntropy).parameters
            sig_2 = inspect.signature(nn.CrossEntropyLoss).parameters
            self.assertEqual(sig_1, sig_2)
            sig_1 = inspect.signature(
                TupleAwareCrossEntropy.forward
            ).parameters
            sig_2 = inspect.signature(nn.CrossEntropyLoss.forward).parameters
            self.assertEqual(sig_1, sig_2)
            self.assertTrue(
                issubclass(TupleAwareCrossEntropy, nn.CrossEntropyLoss)
            )
            self.assertTrue(loss, TupleAwareCrossEntropy()(input, target))
            self.assertTrue(
                loss, TupleAwareCrossEntropy()((input, input), target)
            )
            self.assertRaises(
                TypeError, TupleAwareCrossEntropy(), [input, input], target
            )

            # Test tuple-aware criterion for instance as input.
            sig_1 = inspect.signature(
                tuple_aware_cross_entropy.forward
            ).parameters
            sig_2 = inspect.signature(nn.CrossEntropyLoss().forward).parameters
            self.assertEqual(sig_1, sig_2)
            self.assertTrue(
                isinstance(tuple_aware_cross_entropy, nn.CrossEntropyLoss)
            )
            self.assertTrue(tuple_aware_cross_entropy.ignore_index, -10)
            self.assertTrue(loss, tuple_aware_cross_entropy(input, target))
            self.assertTrue(
                loss, tuple_aware_cross_entropy((input, input), target)
            )
            self.assertRaises(
                TypeError, tuple_aware_cross_entropy, [input, input], target
            )
