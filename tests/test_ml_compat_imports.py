from __future__ import annotations

from core.research.ml import datasets as compat_datasets
from core.research.ml import evaluation as compat_evaluation
from core.research.ml import labels as compat_labels
from core.research.ml.data import datasets as canonical_datasets
from core.research.ml.features import labels as canonical_labels
from core.research.ml.metrics import evaluation as canonical_evaluation


def test_dataset_compat_imports_reexport_canonical_implementations():
    assert compat_datasets.MLDataset is canonical_datasets.MLDataset
    assert compat_datasets.build_dataset is canonical_datasets.build_dataset
    assert (
        compat_datasets.dataset_leakage_audit
        is canonical_datasets.dataset_leakage_audit
    )
    assert compat_datasets.write_dataset is canonical_datasets.write_dataset


def test_evaluation_compat_import_reexports_canonical_implementation():
    assert (
        compat_evaluation.classification_metrics
        is canonical_evaluation.classification_metrics
    )
    assert compat_evaluation.classification_metrics([1, 0], [1, 1])["samples"] == 2


def test_labels_compat_import_reexports_canonical_implementation():
    assert compat_labels.MLLabelBuildResult is canonical_labels.MLLabelBuildResult
    assert (
        compat_labels.ShouldReduceExposureLabelBuilder
        is canonical_labels.ShouldReduceExposureLabelBuilder
    )
