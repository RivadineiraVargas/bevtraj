import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed, find_latest_checkpoint
from utils.config import save_config_as_txt
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import os


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    save_config_as_txt(cfg)

    model = build_model(cfg.MODEL)
    train_set = build_dataset(cfg.TRAIN_DATASET, val=False)
    val_set = build_dataset(cfg.VAL_DATASET, val=True)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices) // train_set.data_chunk_size, 1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices) // val_set.data_chunk_size, 1)

    call_backs = []

    if cfg.save_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath='ckpt/' + cfg.exp_name,
            monitor='val/minADE5',
            filename='{epoch}-{val/minADE5:.2f}',
            save_top_k=3,
            mode='min',  # 'min' for loss/error, 'max' for accuracy
            every_n_epochs=1,
        )
        call_backs.append(checkpoint_callback)

        # TESIS: red de seguridad por PASOS. El checkpoint de arriba solo guarda al
        # cerrar una epoca; en esta maquina una corrida puede cortarse a mitad de
        # epoca (degradacion de la GPU, corte de energia) y se pierde entera.
        # `last.ckpt` se sobreescribe cada N pasos y es lo que busca el resume.
        call_backs.append(ModelCheckpoint(
            dirpath='ckpt/' + cfg.exp_name,
            filename='step-{step}',
            every_n_train_steps=cfg.method.get('ckpt_every_n_steps', 1000),
            save_top_k=1,
            save_last=True,
        ))

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=train_set.collate_fn)

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=WandbLogger(project="unitraj", name=cfg.exp_name, id=cfg.exp_name),
        devices=cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        accelerator="gpu",
        profiler="simple",
        strategy="ddp_find_unused_parameters_true",
        callbacks=call_backs,
        check_val_every_n_epoch=cfg.method.get('check_val_every_n_epoch', 1),
        accumulate_grad_batches=cfg.method.get('accumulate_grad_batches', 1),
        num_sanity_val_steps=0,
        enable_checkpointing=cfg.save_checkpoint,
    )

    # TESIS: la reanudacion automatica pasa a ser EXPLICITA (`auto_resume: true`).
    # Con `save_checkpoint` activado, relanzar un brazo con el mismo `exp_name`
    # reanudaba desde `last.ckpt` en vez de empezar de cero, sin ningun aviso: una
    # corrida con otra semilla heredaba el estado de la anterior y parecia normal.
    # En una comparacion controlada entre brazos eso invalida el resultado en silencio.
    if cfg.ckpt_path is None and cfg.get('auto_resume', False):
        cfg.ckpt_path = find_latest_checkpoint(os.path.join('ckpt', cfg.exp_name))
        if cfg.ckpt_path:
            print(f"[train] REANUDANDO desde {cfg.ckpt_path}")
    elif cfg.ckpt_path is None:
        previo = find_latest_checkpoint(os.path.join('ckpt', cfg.exp_name))
        if previo:
            print(f"[train] AVISO: existe {previo} de una corrida anterior con este "
                  f"exp_name. NO se reanuda (auto_resume=false). Use un exp_name "
                  f"distinto por brazo y semilla, o borre ese directorio.")

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    train()
