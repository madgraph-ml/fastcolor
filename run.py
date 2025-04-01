import hydra
from src.main import main

@hydra.main(config_path="config", config_name="gatr", version_base=None)
def run(cfg):
    main(cfg)

if __name__ == "__main__":
    run()