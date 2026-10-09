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


def _exige_frontera_de_epoca(ruta, lotes_por_epoca, acumulacion):
    """TESIS (auditoria 09/10/2026): solo se reanuda desde el final de una epoca.

    Un checkpoint a mitad de epoca hace que Lightning 2.1 repita parte de la epoca con el
    orden de la epoca 0 y descoloque la acumulacion (ver el callback `fin-epoca`). Se
    comprueba con el global_step guardado: debe ser multiplo de los pasos por epoca.
    """
    import math
    ck = torch.load(ruta, map_location='cpu')
    pasos_por_epoca = math.ceil(lotes_por_epoca / acumulacion)
    paso = int(ck.get('global_step', -1))
    del ck
    if paso < 0 or paso % pasos_por_epoca != 0:
        raise RuntimeError(
            f"[train] {ruta} esta en el paso {paso}, que no es frontera de epoca "
            f"({pasos_por_epoca} pasos por epoca). Reanudar a mitad de epoca no reproduce "
            f"el entrenamiento: use fin-epoca.ckpt o empiece de cero.")
    print(f"[train] checkpoint en frontera de epoca: paso {paso} = epoca {paso // pasos_por_epoca}")


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
            # TESIS: monitor configurable. El diseno prohibe reportar minADE_k solo, asi
            # que no debe ser tampoco el criterio de seleccion; el piloto usa
            # brier-minFDE1. Por defecto se conserva el del original.
            monitor=cfg.method.get('ckpt_monitor', 'val/minADE5'),
            # TESIS: sin `auto_insert_metric_name=False`, Lightning mete el NOMBRE de la
            # metrica en el fichero, y como se llama `val/minADE5` la barra crea un
            # directorio: queda `epoch=0-val/minADE5=3.90.ckpt`. Ademas los `=` rompen
            # la gramatica de overrides de Hydra, asi que ese checkpoint no se puede
            # pasar como `ckpt_path=...` para evaluarlo. Se deja un nombre plano.
            filename='ep{epoch:02d}-ade{val/minADE5:.2f}-brier{val/brier_minFDE10:.2f}',
            auto_insert_metric_name=False,
            save_top_k=1,   # TESIS: disco al 95 %; 311 MB por checkpoint
            mode='min',  # 'min' for loss/error, 'max' for accuracy
            every_n_epochs=1,
        )
        call_backs.append(checkpoint_callback)

        # TESIS: red de seguridad POR EPOCA (auditoria del 09/10/2026). Hasta entonces se
        # guardaba `last.ckpt` cada 1.000 pasos y se reanudaba desde ahi, a mitad de epoca.
        # Con Lightning 2.1 eso NO reproduce el entrenamiento (calibra_B, 03/10): reutiliza
        # el iterador creado antes de fijar la epoca del muestreador (orden de la epoca 0),
        # no salta los lotes ya vistos, descoloca la acumulacion de gradiente (un paso con
        # un solo lote) y avanza el scheduler de mas. Ahora el unico estado reanudable se
        # guarda al cerrar cada epoca (`fin-epoca.ckpt`, se sobreescribe); un corte pierde
        # como mucho la epoca en curso, y la reanudacion rechaza cualquier checkpoint que
        # no este en una frontera de epoca (_exige_frontera_de_epoca, abajo).
        call_backs.append(ModelCheckpoint(
            dirpath='ckpt/' + cfg.exp_name,
            filename='fin-epoca',
            every_n_epochs=1,
            save_top_k=1,
            save_last=False,
            save_on_train_epoch_end=True,
            enable_version_counter=False,
        ))

    # TESIS: EMA de los pesos (ver utils/ema.py). Solo si el config trae `ema_decay`;
    # sin el, todo queda exactamente como antes.
    from utils.ema import RegistraOrden, RegistraValidacion
    call_backs.append(RegistraOrden())
    call_backs.append(RegistraValidacion(os.path.join('ckpt', cfg.exp_name, 'validacion.jsonl')))
    ema = None
    if cfg.method.get('ema_decay', None):
        from utils.ema import EMAPesos
        ema = EMAPesos(float(cfg.method['ema_decay']))
        call_backs.append(ema)

    # TESIS: el original no barajaba: el mismo orden en TODAS las epocas. Barajar entre
    # epocas es lo estandar; se aplica igual a todos los brazos y se declara. Con la
    # estrategia DDP, Lightning lo convierte en un DistributedSampler con shuffle que se
    # rebaraja en cada epoca con la semilla.
    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        shuffle=cfg.method.get('shuffle_train', True), collate_fn=train_set.collate_fn)

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
        # TESIS: precision configurable; por defecto la de Lightning 2.1 ('32-true'), es
        # decir, la de BEVTraj publicado. '16-mixed' solo tras validarla (BEVFusion entrena
        # en fp16, pero BEVTraj en fp32).
        precision=cfg.method.get('precision', '32-true'),
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
    if cfg.ckpt_path:
        _exige_frontera_de_epoca(cfg.ckpt_path, len(train_loader),
                                 cfg.method.get('accumulate_grad_batches', 1))

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)

    # TESIS: el estado FINAL se guarda de forma explicita para evaluar "el ultimo epoch" sin
    # depender de que callback guardo ultimo (hasta el 09/10, `last.ckpt` iba hasta 999
    # pasos por detras; ahora `fin-epoca.ckpt` deberia coincidir, pero no se supone).
    if cfg.save_checkpoint:
        trainer.save_checkpoint(os.path.join('ckpt', cfg.exp_name, 'final.ckpt'))
        print(f"[train] modelo final guardado en ckpt/{cfg.exp_name}/final.ckpt")
        if ema is not None:
            ema.guardar_con_ema(trainer, model, os.path.join('ckpt', cfg.exp_name, 'final_ema.ckpt'))


if __name__ == '__main__':
    train()
