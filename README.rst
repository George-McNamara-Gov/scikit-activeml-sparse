.. intro_start

.. image:: https://raw.githubusercontent.com/scikit-activeml/scikit-activeml/master/docs/logos/scikit-activeml-logo.png
   :class: dark-light
   :align: center
   :width: 40%

|

==========================================================================
scikit-activeml: A Comprehensive and User-friendly Active Learning Library
==========================================================================
|Doc| |Codecov| |PythonVersion| |PyPi| |Black| |Downloads| |Paper|

.. |Doc| image:: https://img.shields.io/badge/docs-latest-green
   :target: https://scikit-activeml.github.io/latest/

.. |Codecov| image:: https://codecov.io/gh/scikit-activeml/scikit-activeml/branch/master/graph/badge.svg
   :target: https://app.codecov.io/gh/scikit-activeml/scikit-activeml

.. |PythonVersion| image:: https://img.shields.io/badge/python-03.10%20%7C%203.11%20%7C%203.12%20%7C3.13-blue.svg
   :target: https://pypi.org/project/scikit-activeml/

.. |PyPi| image:: https://badge.fury.io/py/scikit-activeml.svg
   :target: https://pypi.org/project/scikit-activeml/

.. |Paper| image:: https://img.shields.io/badge/paper-10.20944/preprints202507.0252.v1-blue.svg
   :target: https://www.preprints.org/manuscript/202507.0252/v1

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. |Downloads| image:: https://static.pepy.tech/badge/scikit-activeml
   :target: https://www.pepy.tech/projects/scikit-activeml

Machine learning models often require substantial amounts of training data to perform effectively.
While unlabeled data can be gathered with relative ease, labeling is typically difficult,
time-consuming, or expensive. Active learning addresses this challenge by querying labels
for the most informative samples, achieving high performance with fewer labeled examples.
With this goal in mind, **scikit-activeml** has been developed as a Python library for active
learning on top of `scikit-learn <https://scikit-learn.org/stable/>`_. As a
result, it natively supports **deep active learning** via
`skorch <https://skorch.readthedocs.io/en/stable/>`_. Illustrations for pool-based and stream-based active learning with code
snippets are given below:


.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - .. image:: https://raw.githubusercontent.com/scikit-activeml/scikit-activeml/refs/heads/skorch_wrapper/docs/logos/readme_pool.gif
          :width: 100%
     - .. image:: https://raw.githubusercontent.com/scikit-activeml/scikit-activeml/refs/heads/skorch_wrapper/docs/logos/readme_stream.gif
          :width: 100%

.. raw:: html

   <div style="clear: both;"></div>

.. raw:: html

   <details>
   <summary style="font-size: 100%; font-weight: bold;">
      <b>Pool-based Active Learning: Code Snippet 🏊</b>
   </summary>

The following snippet implements an active learning cycle with 15
iterations using a PyTorch-based classifier (wrapped via
:code:`SkorchClassifier`) and the BADGE query strategy on
sentence-transformer embeddings of the Reuters-21578 dataset obtained
from the pretrained SentenceTransformer model `all-MiniLM-L6-v2`.
Unlabeled data is represented by the value :code:`missing_label` in the
label vector :code:`y_train`. Note that the packages `torch <https://pytorch.org/>`_,
`sentence_transformers <https://www.sbert.net/>`_, and `datasets
<https://huggingface.co/docs/datasets/en/quickstart>`_ are not included in the
default :code:`skactiveml` installation and must be installed separately.

.. code-block:: python

   import numpy as np
   import torch

   from datasets import load_dataset
   from matplotlib import pyplot as plt, animation
   from sentence_transformers import SentenceTransformer
   from skactiveml.classifier import SkorchClassifier
   from skactiveml.pool import Badge
   from sklearn.manifold import TSNE
   from skorch.callbacks import LRScheduler
   from torch import nn
   from torch.optim.lr_scheduler import CosineAnnealingLR

   # Define the device depending on its availability.
   device = "cuda" if torch.cuda.is_available() else "cpu"

   # Load data from Huggingface and encode it via `sentence_transformers`.
   ds_train = load_dataset("yangwang825/reuters-21578", split="train")
   ds_test = load_dataset("yangwang825/reuters-21578", split="test")
   mdl = SentenceTransformer("all-MiniLM-L6-v2", device=device)
   X_pool = mdl.encode(ds_train["text"])
   y_pool = np.asarray(ds_train["label"], dtype=np.int64)
   X_test = mdl.encode(ds_test["text"])
   y_test = np.asarray(ds_test["label"], dtype=np.int64)
   n_features, classes = X_pool.shape[1], np.unique(y_pool)
   missing_label = -1

   # Build your `torch` module for classification, which outputs:
   # - classification logits,
   # - learned sample embeddings.
   class ClassificationModule(nn.Module):
        def __init__(self, n_features, n_classes, n_hidden_units):
            super().__init__()
            self.linear_1 = nn.Linear(n_features, n_hidden_units)
            self.linear_2 = nn.Linear(n_hidden_units, n_classes)
            self.activation = nn.ReLU()

        def forward(self, x):
            x_embed = self.linear_1(x)
            logits = self.linear_2(self.activation(x_embed))
            return logits, x_embed

   # Wrap your torch module via a `skactiveml` wrapper, which requires the
   # definition of training parameters.
   clf = SkorchClassifier(
       module=ClassificationModule,
       criterion=nn.CrossEntropyLoss,
       neural_net_param_dict={
           # Module-related parameters.
           "module__n_features": n_features,
           "module__n_hidden_units": 128,
           "module__n_classes": len(classes),
           "predict_nonlinearity": nn.Softmax(dim=-1),
           # Optimizer-related parameters.
           "max_epochs": 100,
           "batch_size": 16,
           "lr": 0.01,
           "optimizer": torch.optim.RAdam,
           "callbacks": [
               ("lr_scheduler", LRScheduler(policy=CosineAnnealingLR, T_max=100))
           ],
           # General parameters.
           "verbose": 0,
           "device": device,
           "train_split": False,
           "iterator_train__shuffle": True,
           "torch_load_kwargs": {"weights_only": True},
       },
       classes=classes,
       missing_label=missing_label,
   ).initialize()

   # Start the active learning cycle with zero initial labels.
   y_train = np.full_like(y_pool, missing_label)

   # Create a deep active learning query strategy.
   qs = Badge(
       missing_label=missing_label,
       clf_embedding_flag_name="return_embeddings",
   )

   # Define the active learning parameters.
   n_cycles = 15
   batch_size = 4

   # Execute active learning cycles.
   for c in range(n_cycles):
       query_idx = qs.query(
           X=X_pool,
           y=y_train,
           batch_size=batch_size,
           clf=clf,
           fit_clf=False,
       )
       y_train[query_idx] = y_pool[query_idx]
       clf.fit(X_pool, y_train)

   print(f"Final accuracy: {clf.score(X_test, y_test)}")

.. raw:: html

   </details>

.. raw:: html

   <details>
   <summary style="font-size: 100%; font-weight: bold;">
     <b>Stream-based Active Learning: Code Snippet 🌊</b>
   </summary>

The following snippet implements a stream-based active learning cycle over
300 time steps on CIFAR-10 embeddings computed with the pretrained DINOv2
vision transformer. A PyTorch-based classifier
(wrapped via :code:`SkorchClassifier`) is trained online, and the
Split query strategy is used with a labeling budget of 10% of the
stream. Unlabeled data is represented by the value :code:`missing_label` in
the label vector :code:`y_train`. Note that the packages `torch <https://pytorch.org/>`_,
`transformers <https://huggingface.co/docs/transformers/en/quicktour>`_, and `datasets
<https://huggingface.co/docs/datasets/en/quickstart>`_ are not included in the
default :code:`skactiveml` installation and must be installed separately.

.. code-block:: python

   import numpy as np
   import torch

   from matplotlib import pyplot as plt, animation
   from torch import nn
   from torch.optim.lr_scheduler import CosineAnnealingLR
   from datasets import load_dataset
   from skactiveml.classifier import SkorchClassifier
   from skactiveml.stream import Split
   from skactiveml.utils import is_labeled
   from sklearn.manifold import TSNE
   from skorch.callbacks import LRScheduler
   from transformers import AutoImageProcessor, Dinov2Model

   # Define the device depending on its availability.
   device = "cuda" if torch.cuda.is_available() else "cpu"

   # Load data.
   ds = load_dataset("cifar10")
   processor = AutoImageProcessor.from_pretrained(
       "facebook/dinov2-small", use_fast=True
   )
   model = Dinov2Model.from_pretrained("facebook/dinov2-small").to(device).eval()
   def embed(batch):
       inputs = processor(images=batch["img"], return_tensors="pt").to(device)
       with torch.no_grad():
           out = model(**inputs).last_hidden_state[:, 0]
       batch["emb"] = out.cpu().numpy()
       return batch
   ds = ds.map(embed, batched=True, batch_size=128)
   X_stream = np.stack(ds["train"]["emb"], dtype=np.float32)[:1000]
   y_stream = np.array(ds["train"]["label"], dtype=np.int64)[:1000]
   X_test = np.stack(ds["test"]["emb"], dtype=np.float32)
   y_test = np.array(ds["test"]["label"], dtype=np.int64)
   n_features, classes = X_stream.shape[1], np.unique(y_stream)
   missing_label = -1

   # Build `torch` module for classification, outputting classification logits.
   class ClassificationModule(nn.Module):
       def __init__(self, n_features, n_classes, n_hidden_units):
           super().__init__()
           self.linear_1 = nn.Linear(n_features, n_hidden_units)
           self.linear_2 = nn.Linear(n_hidden_units, n_classes)
           self.activation = nn.ReLU()

       def forward(self, x):
           x_embed = self.linear_1(x)
           logits = self.linear_2(self.activation(x_embed))
           return logits

   # Wrap your torch module via a `skactiveml` wrapper, which requires the
   # definition of training parameters.
   clf = SkorchClassifier(
       module=ClassificationModule,
       criterion=nn.CrossEntropyLoss,
       neural_net_param_dict={
           # Module-related parameters.
           "module__n_features": n_features,
           "module__n_hidden_units": 128,
           "module__n_classes": len(classes),
           "predict_nonlinearity": nn.Softmax(dim=-1),
           # Optimizer-related parameters.
           "max_epochs": 100,
           "batch_size": 16,
           "lr": 0.01,
           "optimizer": torch.optim.RAdam,
           "callbacks": [
               ("lr_scheduler", LRScheduler(policy=CosineAnnealingLR, T_max=100))
           ],
           # General parameters.
           "verbose": 0,
           "device": device,
           "train_split": False,
           "iterator_train__shuffle": True,
           "torch_load_kwargs": {"weights_only": True},
       },
       classes=classes,
       missing_label=missing_label,
   ).initialize()

   # Initialize training data as empty lists.
   y_train = np.full_like(y_stream, missing_label)

   # Execute active learning cycle.
   qs = Split(random_state=0, budget=0.1)
   n_cycles = 300
   query_idx = []
   for t in range(n_cycles):
       query_idx = qs.query(
           candidates=X_stream[[t]], y=y_stream[t], clf=clf, fit_clf=False
       )
       qs.update(candidates=X_stream[[t]], queried_indices=query_idx)
       if len(query_idx) > 0:
           y_train[t] = y_stream[t]
           clf.fit(X_stream, y_train)

   print(f"Final accuracy: {clf.score(X_test, y_test)}")

.. raw:: html

   </details>

.. intro_end

.. user_installation_start

💾 User Installation
--------------------------
**Minimal installation:** The easiest way to install scikit-activeml is using
``pip``:

.. code-block:: bash

   pip install -U scikit-activeml

This installs only the minimum requirements to avoid potential package
downgrades within your existing environment.

**Recommended installation for deep active learning:** In most cases, we
recommend installing scikit-activeml together with the optional dependencies:

.. code-block:: bash

   pip install -U scikit-activeml[opt]

The ``opt`` extra installs additional packages such as ``skorch`` to enable
more sophisticated deep learning support and other extended functionality.
Version constraints are chosen to be reasonably flexible so that scikit-activeml
can integrate well into an existing environment.

**Tested fallback configuration:** If you prefer a configuration where
dependency versions have been tested explicitly for this release, you can
install scikit-activeml with the maximum tested core and optional requirements:

.. code-block:: bash

   pip install -U scikit-activeml[max,opt_max]

This setup uses the versions listed in ``requirements_max.txt`` and
``requirements_opt_max.txt`` and corresponds to the configuration used in
our continuous integration tests. You can also install only the maximum
tested core dependencies via:

.. code-block:: bash

   pip install -U scikit-activeml[max]

**Note on PyTorch and skorch:** The optional deep learning functionality
(via ``skorch``) assumes that ``torch`` (PyTorch) is already installed in
your environment. Since the correct PyTorch build depends on your hardware
and CUDA setup, we do **not** install PyTorch automatically.
Please install PyTorch separately by following the installation
instructions of from `skorch <https://github.com/skorch-dev/skorch?tab=readme-ov-file#pytorch>`_.

.. user_installation_end

.. overview_start

🗂️ Query Strategy Overview
--------------------------
For better orientation, we provide a `overview <https://scikit-activeml
.github.io/latest/generated/strategy_overview.html>`_
(including paper references and `visual examples <https://scikit-activeml.github.io/latest/generated/sphinx_gallery_examples/index.html>`_)
of the query strategies implemented by ``skactiveml``. We indicate each
strategy’s target learning tasks (regression and/or classification),
flag multi-annotator scenarios, and mark strategies that consider diversity
between samples within a selected batch. Furthermore, we categorize query
strategies by their selection principles, i.e., informativeness (model
uncertainty), representativeness (data‐distribution coverage), and hybrid
(combining both). The following mind map illustrates these different
attributes of a query strategy.

.. image:: https://raw.githubusercontent.com/scikit-activeml/scikit-activeml/refs/heads/518-strategy-documentation/docs/logos/scikit-activeml-query-strategy-overview.svg
   :class: dark-light
   :align: center
   :width: 100%

.. overview_end

.. examples_start

📚 In-depth Tutorials
---------------------

The table below summarizes a selection of our in-depth tutorials.
Each entry lists the active learning scenario, prediction task, data
modality, and model family used in the tutorial.

.. list-table::
   :header-rows: 1
   :widths: 20 12 12 12 12

   * - Tutorial
     - Scenario
     - Task
     - Data
     - Model
   * - Pool-based Active Learning: Getting Started
     - Pool
     - Classification
     - Toy
     - Logistic Regression
   * - Stream-based Active Learning: Getting Started
     - Stream
     - Classification
     - Toy
     - Parzen Window Classifier
   * - Pool-based Active Learning for Regression: Getting Started
     - Pool
     - Regression
     - Toy
     - Kernel Regressor
   * - Deep Active Learning for Fine-tuning Vision Foundation Models
     - Pool
     - Classification
     - Image
     - Vision Transformer
   * - Stream-based Acitve Learning in Batches
     - Stream
     - Classification
     - Toy
     - Parzen Window Classifier
   * - Deep Active Learning for Regression Tasks
     - Pool
     - Regression
     - Tabular
     - Extreme Gradient Boosted Tree
   * - Paper Annotation Tool
     - Pool
     - Classification
     - Text
     - Logistic Regression

.. examples_end

.. citing_start

🧾 Citing
---------
If you use ``skactiveml`` in your research projects and find it helpful, please cite the following:

.. code-block:: tex

    @article{skactiveml2021,
        title={{scikit-activeml: A Comprehensive and User-friendly Active Learning Library}},
        author={Herde, Marek and Pham, Minh Tuan and Kottke, Daniel and Benz, Alexander and L{\"u}hrs, Lukas and Mergard, Pascal and Sandrock, Christoph and Cheng, Jiaying and Roghman, Atal and M{\"u}jde, Mehmet and Rauch, Lukas and Sick, Bernahrd},
        journal={Preprints},
        doi={10.20944/preprints202507.0252.v1},
        year={2025},
        url={https://github.com/scikit-activeml/scikit-activeml}
    }

.. citing_end