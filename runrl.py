import hydra
from omegaconf import DictConfig, OmegaConf

import importlib

from data.base import make_loader
from model import make_model
from rleditor.editor import PPOEditor

import wandb

@hydra.main(version_base=None, config_path="config", config_name="config")
def run(config: DictConfig):
    
    wandb.init(
        project = f"{config.data.name}_{config.model.name_or_path}",
        name = f"{config.editor.name}_{str(config.data.n_edits)}",
        config = OmegaConf.to_container(config, resolve = True)
    )
    
    data_module = importlib.import_module(f"data.{config.data.name}")
    data_class = getattr(data_module, f"{config.data.name.upper()}Dataset")
    train_loader, valid_loader = make_loader(config, data_class)
    
    model = make_model(config.model).to(config.model_device)

    # editor_module = importlib.import_module(f"editor.{config.editor.name}")
    # editor_class = getattr(editor_module, config.editor.name.upper())
    # editor = editor_class(config, model)
    # editor.run(train_loader, valid_loader)
    # editor.run_sequential(train_loader, valid_loader)
    editor = PPOEditor(model, config)
    editor.train()
    results = editor.evaluate()
    print(f"Evaluation results: {results}")

    
if __name__ == "__main__":
    run()