from pathlib import Path


def test_training_config_exists():
    assert Path("configs/training.yaml").exists()
