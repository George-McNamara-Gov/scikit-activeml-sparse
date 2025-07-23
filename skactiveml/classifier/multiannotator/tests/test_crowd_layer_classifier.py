import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.utils.validation import NotFittedError, check_is_fitted

successful_skorch_torch_import = False
try:
    import torch
    from torch import nn
    from skorch.helper import predefined_split
    from skorch.dataset import Dataset
    from skactiveml.classifier.multiannotator import CrowdLayerClassifier

    successful_skorch_torch_import = True
except ImportError:
    pass  # pragma: no cover

if successful_skorch_torch_import:

    class TestCrowdLayerClassifier(unittest.TestCase):
        def setUp(self):
            self.X, self.y_true = make_blobs(
                n_samples=300, n_features=2, centers=3, random_state=0
            )
            self.X = self.X.astype(np.float32)
            self.y = np.array([self.y_true, self.y_true], dtype=float).T
            self.y[:100, 0] = -1
            self.neural_net_param_dict = {
                "train_split": None,
                "verbose": False,
                "optimizer": torch.optim.RAdam,
                "device": "cpu",
                "max_epochs": 10,
                "batch_size": 1,
                "lr": 0.001,
            }
            self.clf_init_params = {
                "n_annotators": 2,
                "classes": [0, 1, 2],
                "missing_label": -1,
                "cost_matrix": None,
                "random_state": 1,
                "neural_net_param_dict": self.neural_net_param_dict,
            }

        def test_init_param_module_gt_net(self):
            clf = CrowdLayerClassifier(gt_net="Test", **self.clf_init_params)
            self.assertEqual(clf.gt_net, "Test")
            self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y)

            clf = CrowdLayerClassifier(gt_net=None, **self.clf_init_params)
            self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y)

            clf = CrowdLayerClassifier(
                gt_net=[("nn.Module", TestNeuralNet)], **self.clf_init_params
            )
            self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y)

            clf_init_params = self.clf_init_params.copy()
            clf_init_params["classes"] = [0, 1, 2]
            clf = CrowdLayerClassifier(
                gt_net=TestNeuralNet, **self.clf_init_params
            )
            self.assertRaises(TypeError, clf.fit, X=self.X, y=self.y)

        def test_fit(self):
            gt_net = TestNeuralNet()
            clf = CrowdLayerClassifier(
                gt_net=gt_net,
                **self.clf_init_params,
            )

            np.testing.assert_array_equal([0, 1, 2], clf.classes)
            self.assertRaises(NotFittedError, check_is_fitted, clf)
            clf.fit(self.X, self.y)
            check_is_fitted(clf)

        def test_predict(self):
            gt_net = TestNeuralNet()
            clf = CrowdLayerClassifier(
                gt_net=gt_net,
                **self.clf_init_params,
            )
            clf.predict(X=self.X)
            clf = CrowdLayerClassifier(
                gt_net=gt_net,
                **self.clf_init_params,
            )
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            self.assertEqual(len(y_pred), len(self.X))

        def test_predict_annotator_pref(self):
            gt_net = TestNeuralNet()
            clf = CrowdLayerClassifier(
                gt_net=gt_net,
                **self.clf_init_params,
            )
            clf.predict(X=self.X)
            clf.fit(self.X, self.y)
            annot_pref = clf.predict_annotator_perf(self.X[:2])
            self.assertEqual(annot_pref.shape[0], 2)
            self.assertEqual(annot_pref.shape[1], 2)
            confusion_matrix = clf.predict_annotator_perf(self.X[:2], True)
            self.assertEqual(confusion_matrix.shape[0], 2)
            self.assertEqual(confusion_matrix.shape[1], 2)
            self.assertEqual(confusion_matrix.shape[2], 3)
            self.assertEqual(confusion_matrix.shape[3], 3)

        def test_predict_proba(self):
            gt_net = TestNeuralNet()
            clf = CrowdLayerClassifier(
                gt_net=gt_net,
                **self.clf_init_params,
            )
            clf.predict_proba(X=self.X)
            clf.fit(self.X, self.y)
            proba = clf.predict_proba(self.X)
            self.assertEqual(len(self.X), proba.shape[0])
            self.assertEqual(3, proba.shape[1])

        def test_predict_proba_annot(self):
            gt_net = TestNeuralNet()
            clf = CrowdLayerClassifier(
                gt_net=gt_net,
                **self.clf_init_params,
            )
            proba = clf.predict_proba_annot(self.X)
            clf.fit(self.X, self.y)
            annot = clf.predict_proba_annot(self.X)
            print(annot.shape)
            n_classes = len(np.unique(self.y_true))
            n_annotators = self.clf_init_params["n_annotators"]
            self.assertEqual(annot.shape[0], len(self.X))
            self.assertEqual(annot.shape[1], n_classes)
            self.assertEqual(annot.shape[2], n_annotators)

    class TestNeuralNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_to_hidden = nn.Linear(
                in_features=2, out_features=2, bias=True
            )
            self.hidden_to_output = nn.Linear(
                in_features=2, out_features=3, bias=True
            )

        def forward(self, X):
            hidden = self.input_to_hidden(X)
            hidden = torch.relu(hidden)
            output_values = self.hidden_to_output(hidden)
            return output_values
