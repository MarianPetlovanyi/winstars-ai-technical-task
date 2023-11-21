import opendatasets
import pandas
import hydra
from omegaconf import DictConfig
"""
This script is created to download dataset for project. 
If you have kaggle.json file with credentials(Kaggle API token) 
you can copy it to root directory and data will be passed automatically.
"""

@hydra.main(config_path="../config", config_name="config.yaml")
def download_dataset(cfg: DictConfig) -> None:
    opendatasets.download(cfg.dataset.url, data_dir=cfg.dataset.download_path)

if __name__ == "__main__":
    download_dataset()