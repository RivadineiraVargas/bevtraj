import math
from torch.optim.lr_scheduler import _LRScheduler


def fija_lr_de_epoca(sch, epoca):
    """TESIS (auditoria 09/10/2026): fija la LR de la epoca como funcion PURA de la epoca.

    Lightning 2.1 avanza los schedulers por epoca una vez de mas en la primera epoca tras
    reanudar (calibra_B: epocas 3-14 con la LR de la siguiente). Llamada al empezar cada
    epoca, deja last_epoch = epoca y la LR de todos los grupos = get_lr(), que es lo que
    el scheduler da en una corrida sin cortes. Devuelve (lrs, last_epoch que traia).
    Probada en tools/prueba_lr_por_epoca.py (incluido que sin ella la prueba falla).
    """
    traia = sch.last_epoch
    sch.last_epoch = int(epoca)
    lrs = sch.get_lr()
    for g, lr in zip(sch.optimizer.param_groups, lrs):
        g['lr'] = lr
    sch._last_lr = list(lrs)
    return [float(l) for l in lrs], traia


class WarmupCosLR(_LRScheduler):
    def __init__(
        self, optimizer, min_lr, lr, warmup_epochs, epochs, last_epoch=-1, verbose=False
    ) -> None:
        # TESIS: conversion defensiva. Los configs publicados traen `min_lr: 1e-5`: con
        # PyYAML a secas es una cadena, pero Hydra carga los configs con OmegaConf 2.3,
        # que ya lo lee como float (comprobado en la imagen bevtraj:cu118, 19/11/2026).
        # Este comentario decia (23/09) que el coseno fallaba con TypeError en la epoca 2:
        # en este entorno no ocurre (auditoria independiente del 19/11/2026).
        self.min_lr = float(min_lr)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.last_epoch = last_epoch
        
        if last_epoch != -1:
            init_lrs = [self.get_init_lr() for _ in optimizer.param_groups]
            for group, init_lr in zip(optimizer.param_groups, init_lrs):
                group.setdefault("initial_lr", init_lr)
        
        super(WarmupCosLR, self).__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_init_lr(self):
        lr = self.lr / self.warmup_epochs
        return lr

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.lr * (self.last_epoch + 1) / self.warmup_epochs
        else:
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.epochs - self.warmup_epochs)
                )
            )
        # TESIS: el original solo miraba si param_groups[0] tenia "lr_scale": si lo tenia,
        # lo exigia en todos los grupos (KeyError si faltaba en alguno); si no, lo
        # ignoraba en todos. Con `get`, cada grupo usa el suyo (1,0 por omision). Con los
        # grupos que crea configure_optimizers, todos con lr_scale, el original ya daba LR
        # distintas: es robustez. Este comentario decia (23/09) que pisaba la LR del
        # encoder con la del decoder: era falso (auditoria independiente del 19/11/2026).
        return [lr * group.get("lr_scale", 1.0) for group in self.optimizer.param_groups]