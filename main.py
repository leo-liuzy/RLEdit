import hydra
from omegaconf import DictConfig, OmegaConf

import importlib

from data.base import make_loader
from model import make_model

import wandb


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    
    wandb.init(
        project = f"{config.data.name}_{config.model.name}",
        name = f"{config.editor.name}_{str(config.data.n_edits)}",
        config = OmegaConf.to_container(config, resolve = True)
    )
    
    data_module = importlib.import_module(f"data.{config.data.name}")
    data_class = getattr(data_module, f"{config.data.name.upper()}Dataset")

    train_loader, valid_loader = make_loader(config, data_class)

    model = make_model(config.model).to(config.model_device)

    editor_module = importlib.import_module(f"editor.{config.editor.name}")
    editor_class = getattr(editor_module, config.editor.name.upper())
    editor = editor_class(config, model)

    if config.mode == "reedit":
        editor.run_reedit(train_loader, valid_loader)
    elif config.mode == "single":
        editor.run_single(train_loader, valid_loader)
    elif config.mode == "sequential":
        editor.run_sequential(train_loader, valid_loader)
    elif config.mode == "sequential_retrain":
        editor.run_sequential_retrain_full(train_loader, valid_loader)
    elif config.mode == "valid":
        editor.run_test(valid_loader)
    elif config.mode == "preedited":
        editor.run_preedited(train_loader, valid_loader)


if __name__ == "__main__":
    main()