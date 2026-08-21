import pytest

from collision_vision.config import Config


def test_config_file_not_found(tmp_path):
    # Construct a path to a non-existent file
    non_existent_path = tmp_path / "does_not_exist.yaml"

    # Assert that calling from_yaml with the non-existent path raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        Config.from_yaml(non_existent_path)
