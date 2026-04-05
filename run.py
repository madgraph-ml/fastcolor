import hydra

@hydra.main(config_path="config", config_name="gatr", version_base=None)
def run(cfg):
    task = str(getattr(cfg, "task", "undefined"))
    if task == "regression":
        from regression.main import main
        main(cfg)
    elif task == "sampling":
        from sampling.main import main
        main(cfg)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    run()
