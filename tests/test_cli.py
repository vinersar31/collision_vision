import pytest

from collision_vision.cli import _build_parser


def test_cli_preprocess_args():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "preprocess",
            "--format",
            "via",
            "--annotations",
            "ann.json",
            "--images",
            "img/",
        ]
    )
    assert args.command == "preprocess"
    assert args.format == "via"
    assert args.annotations == "ann.json"
    assert args.images == "img/"
    assert args.output == "data/processed"
    assert args.val_ratio == 0.2
    assert args.seed == 0
    assert args.classes == ["dent", "scratch", "structural"]
    assert args.attribute_key is None


def test_cli_train_args():
    parser = _build_parser()
    args = parser.parse_args(["train"])
    assert args.command == "train"
    assert args.config == "config.yaml"
    assert args.model is None
    assert args.epochs is None
    assert args.batch is None
    assert args.imgsz is None
    assert args.device is None


def test_cli_infer_args():
    parser = _build_parser()
    args = parser.parse_args(["infer", "--weights", "model.pt", "--source", "img.jpg"])
    assert args.command == "infer"
    assert args.weights == "model.pt"
    assert args.source == "img.jpg"
    assert args.output == "outputs"
    assert args.conf == 0.25


def test_cli_missing_command():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
