import torch
from src.utils import load_config, set_seed
from src.tasks.synthetic_tasks import get_task_and_dataloader
from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel
from src.trainer import Trainer
import torch.optim as optim

def build_model(cfg):
    model_cfg = dict(cfg.model)
    model_type = model_cfg.pop("type")  # remove "type" before passing kwargs

    if model_type == "transformer":
        return TransformerModel(**model_cfg)
    elif model_type == "lstm":
        return LSTMModel(**model_cfg)
    else:
        raise ValueError(f"Unknown model type {model_type}")

def main(config_path="config/train_teacher.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    task, dataloader = get_task_and_dataloader(batch_size=cfg.data["batch_size"], seq_len=cfg.data["seq_len"], n_samples=cfg.train["epoch_size"],)
    model = build_model(cfg)

    optimizer = optim.Adam(model.parameters(), lr=cfg.train["lr"])

    teacher = None
    if hasattr(cfg.distillation, "teacher_checkpoint"):
        teacher = build_model(cfg)
        teacher.load_state_dict(torch.load(cfg.distillation.teacher_checkpoint))
        teacher.eval()
    print(f"device: {cfg.device}")
    trainer = Trainer(model, optimizer, dataloader, device=cfg.device, teacher=teacher, cfg=cfg, ignore_index=task.ignore_token)
    trainer.train(cfg.train["epochs"], save_path=f"{cfg.train['save_dir']}/{cfg.save_name}")

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/train_teacher.yaml"
    main(config_path)
