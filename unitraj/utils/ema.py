"""TESIS: promedio exponencial (EMA) de los pesos entrenables.

Por que: entre tres semillas el modelo final variaba hasta 0,50 en brier-minFDE10 (12 %),
mas que cualquier diferencia entre brazos (confirmacion del 01-02/10/2026). Evaluar un
unico estado final, a lote 1, recoge el ruido de los ultimos pasos. El EMA de los pesos
promedia ese ruido sin coste de computo (Morales-Brotons et al., TMLR 2024,
arXiv 2411.18704: mejor generalizacion y predicciones mas consistentes).

Uso: train.py lo activa con `ema_decay` en el config y, al terminar, guarda
`final_ema.ckpt` ademas de `final.ckpt`. `final.ckpt` no cambia en nada: el EMA es una
lectura ADICIONAL, y cual de las dos es la principal lo fija cada pre-registro.

Detalles que importan:
  - Se actualiza tras cada paso REAL del optimizador (cambio de global_step), no tras
    cada lote: con acumulacion de gradiente, un lote no es un paso.
  - Calentamiento: decay efectivo min(decay, (1+n)/(10+n)), para que los primeros pasos
    (pesos casi aleatorios) no pesen durante miles de pasos.
  - Solo parametros con requires_grad: el encoder congelado no cambia. Los buffers
    (BatchNorm) se quedan los del modelo vivo, que es la practica estandar.
  - Guarda y recupera su estado en los checkpoints de Lightning: una corrida reanudada
    sigue el mismo promedio.
"""
import torch
import pytorch_lightning as pl


class EMAPesos(pl.Callback):
    def __init__(self, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema_decay debe estar en (0, 1), es {decay}")
        self.decay = float(decay)
        self.sombra = {}          # nombre -> tensor EMA
        self.n = 0                # actualizaciones hechas
        self._ultimo_paso = None

    @property
    def state_key(self):
        return f"EMAPesos[decay={self.decay}]"

    def _entrenables(self, pl_module):
        return {n: p for n, p in pl_module.named_parameters() if p.requires_grad}

    def on_train_start(self, trainer, pl_module):
        params = self._entrenables(pl_module)
        if not self.sombra:
            self.sombra = {n: p.detach().clone() for n, p in params.items()}
        else:   # reanudacion: el estado vino del checkpoint, en CPU
            faltan = set(params) ^ set(self.sombra)
            if faltan:
                raise RuntimeError(f"[EMA] el estado reanudado no coincide con el modelo: {sorted(faltan)[:5]}")
            self.sombra = {n: t.to(params[n].device) for n, t in self.sombra.items()}
        self._ultimo_paso = trainer.global_step
        print(f"[EMA] decay={self.decay}, {len(self.sombra)} tensores, "
              f"{sum(t.numel() for t in self.sombra.values())/1e6:.2f} M parametros, n={self.n}")

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == self._ultimo_paso:
            return                  # acumulando gradiente: todavia no hubo paso
        self._ultimo_paso = trainer.global_step
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for n, p in self._entrenables(pl_module).items():
            self.sombra[n].mul_(d).add_(p.detach(), alpha=1.0 - d)
        self.n += 1

    @torch.no_grad()
    def guardar_con_ema(self, trainer, pl_module, ruta):
        """Guarda un checkpoint con los pesos EMA y deja el modelo como estaba."""
        params = self._entrenables(pl_module)
        respaldo = {n: p.detach().clone() for n, p in params.items()}
        for n, p in params.items():
            p.copy_(self.sombra[n])
        try:
            trainer.save_checkpoint(ruta)
        finally:
            for n, p in params.items():
                p.copy_(respaldo[n])
        print(f"[EMA] modelo EMA guardado en {ruta} (n={self.n} actualizaciones)")

    def state_dict(self):
        return {"decay": self.decay, "n": self.n,
                "sombra": {n: t.detach().cpu() for n, t in self.sombra.items()}}

    def load_state_dict(self, state_dict):
        if abs(state_dict["decay"] - self.decay) > 1e-12:
            raise RuntimeError(f"[EMA] decay del checkpoint {state_dict['decay']} != config {self.decay}")
        self.n = state_dict["n"]
        self.sombra = dict(state_dict["sombra"])


class RegistraOrden(pl.Callback):
    """TESIS: deja en el log el muestreador y los primeros indices de la epoca 0.

    Los brazos de una misma semilla deben ver los datos en el MISMO orden (numeros
    aleatorios comunes). Con la estrategia DDP, Lightning sustituye el muestreador por un
    DistributedSampler cuya semilla es PL_GLOBAL_SEED, independiente del estado global de
    numeros aleatorios. Esto lo deja comprobado en cada corrida en vez de supuesto. Solo
    itera el muestreador si es distribuido (generador propio, sin efectos laterales).
    """
    def on_train_start(self, trainer, pl_module):
        from itertools import islice
        from torch.utils.data.distributed import DistributedSampler
        try:    # un registro nunca debe tumbar un entrenamiento
            dl = trainer.train_dataloader
            if isinstance(dl, (list, tuple)):
                dl = dl[0]
            s = dl.sampler
            tipo = type(s).__name__
            if isinstance(s, DistributedSampler):
                primeros = list(islice(iter(s), 8))
                print(f"[orden] muestreador {tipo} (seed={s.seed}, epoca={s.epoch}): primeros indices {primeros}")
            else:
                print(f"[orden] muestreador {tipo}: no distribuido, el orden depende del estado global")
        except Exception as e:
            print(f"[orden] no se pudo registrar el orden: {type(e).__name__}: {e}")


class RegistraValidacion(pl.Callback):
    """TESIS: escribe las metricas de cada validacion en un JSONL (una linea por epoca).

    Para el criterio de CONVERGENCIA hace falta la curva de validacion legible y en disco;
    hasta el 03/10 solo quedaba el minigrafico de wandb. Se usa on_validation_end y no
    on_validation_epoch_end porque en este ultimo Lightning todavia no ha agregado las
    metricas de la epoca (ModelCheckpoint usa el mismo gancho). Abre en modo anadir: una
    corrida reanudada sigue el mismo fichero; cada linea lleva el paso para deduplicar.
    """
    def __init__(self, ruta):
        self.ruta = ruta

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        import json, os
        fila = {"epoca": int(trainer.current_epoch), "paso": int(trainer.global_step)}
        fila.update({k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith("val/")})
        os.makedirs(os.path.dirname(self.ruta), exist_ok=True)
        with open(self.ruta, "a") as f:
            f.write(json.dumps(fila) + "\n")
