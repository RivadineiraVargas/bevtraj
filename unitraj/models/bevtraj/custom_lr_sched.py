import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosLR(_LRScheduler):
    def __init__(
        self, optimizer, min_lr, lr, warmup_epochs, epochs, last_epoch=-1, verbose=False
    ) -> None:
        # TESIS: los configs publicados traen `min_lr: 1e-5`, que YAML 1.1 parsea
        # como CADENA (exige 1.0e-5). La rama del coseno hace `self.lr - self.min_lr`
        # y revienta con TypeError en la primera epoca posterior al warmup: con
        # warmup_epochs=2 el entrenamiento muere en la epoca 2 de 10. Se fuerza a float.
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
        # TESIS: antes se miraba "lr_scale" SOLO en param_groups[0]; con grupos
        # separados (encoder pre-entrenado vs decoder aleatorio) el scheduler
        # pisaba la LR del encoder con la del decoder en cada paso. Es el mismo
        # error que invalido el exp. 18 del proyecto MOTF.
        return [lr * group.get("lr_scale", 1.0) for group in self.optimizer.param_groups]