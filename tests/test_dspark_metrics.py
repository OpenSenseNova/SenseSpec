"""Local DSpark loss and acceptance probability metrics."""

import json

import pytest
from PIL import Image


def test_local_acceptance_matches_weighted_distribution_overlap_and_is_replotable(tmp_path):
    pytest.importorskip(
        "matplotlib", reason="Plotting requires the optional full-data dependencies"
    )
    from sensespec.utils.dspark_metrics import LocalDSparkMetrics, plot_acceptance_curve

    target = [[0.6, 0.3, 0.1], [0.1, 0.1, 0.8]]
    draft = [[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]]
    weights = [1.0, 0.25]
    overlap = sum(
        w * sum(min(p, q) for p, q in zip(ps, qs, strict=True))
        for w, ps, qs in zip(weights, target, draft, strict=True)
    ) / sum(weights)
    l1 = sum(
        w * sum(abs(p - q) for p, q in zip(ps, qs, strict=True))
        for w, ps, qs in zip(weights, target, draft, strict=True)
    ) / sum(weights)
    logger = LocalDSparkMetrics(str(tmp_path), l1_alpha=0.9)
    logger.record(
        {
            "train/ce_loss": 3.0,
            "train/l1_loss": l1,
            "train/confidence_loss": 0.5,
            "train/avg_acc": 0.99,
        },
        1,
    )
    logger.finalize(1)
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert records[0]["estimated_training_acceptance"] == pytest.approx(overlap)
    assert records[0]["estimated_training_acceptance"] != pytest.approx(0.99)
    plot_acceptance_curve(records, tmp_path / "replot.png")
    for name in ("acceptance.png", "loss.png", "replot.png"):
        with Image.open(tmp_path / name) as image:
            image.verify()


@pytest.mark.parametrize("l1,weight", [(-0.1, 0.9), (2.1, 0.9), (float("nan"), 0.9), (0.0, 0.0)])
def test_acceptance_rejects_unmeasured_or_invalid_distribution_loss(l1, weight):
    from sensespec.utils.dspark_metrics import estimated_training_acceptance

    with pytest.raises(ValueError):
        estimated_training_acceptance({"l1_loss": l1, "loss_weights": {"l1_loss": weight}})
