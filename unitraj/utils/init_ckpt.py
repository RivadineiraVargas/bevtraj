"""TESIS (16/11/2026, tesis_lidar/docs/PREREGISTRO_C2_TRAS_CONGELADO_2026-11-16.md, E1): inicializar los PESOS del
modelo desde el checkpoint de otra corrida. Es la etapa de ajuste fino del LP-FT (el decoder ya entrenado con el
encoder congelado, `congela_C_s{s}`) y de su control de presupuesto (A ya entrenado, `calibra_A_s{s}`).

Solo pesos: ni optimizador, ni programa de LR, ni epoca, ni EMA. La EMA arranca de estos pesos porque EMAPesos copia la
sombra en on_train_start, despues de esta carga. Carga estricta: falta o sobra una clave y se aborta.

Cargar sin desajustes no prueba que el modelo funcione: el brazo B del piloto cargaba entero y estaba roto (CLAUDE.md,
02/10). Por eso, antes de entrenar, `comprueba` valida el modelo inicializado en el mismo subconjunto que su corrida de
origen (las 1.000 muestras fijas de train_val, val_subset_num) y exige reproducir la ultima validacion de esa corrida,
la de `validacion.jsonl` junto al checkpoint. Si no, aborta antes del primer paso.
"""
import hashlib
import json
import os

import torch

METRICA = "val/brier_minFDE10"


def _md5(ruta):
    h = hashlib.md5()
    with open(ruta, "rb") as f:   # por bloques: el fichero pesa cientos de MB
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def tolerancia(cfg_method):
    """init_tol del config de E1, o None si el config no es de E1. Vive en MODEL, junto a encoder_bn_fijo (los escribe
    genera_configs_tv.py --lpft con la sangria de MODEL).

    TESIS (16/11/2026): train.py la buscaba en la RAIZ del metodo (cfg.method.get('init_tol')), donde no esta: el aborto
    de un config de E1 lanzado sin init_ckpt no saltaba nunca, y la tolerancia salia del valor por defecto (5e-3 por
    casualidad). Lo vio la revision previa al humo; la prueba en CPU solo miraba que la clave existiera en MODEL.
    """
    m = cfg_method.get("MODEL", None)
    v = None if m is None else m.get("init_tol", None)
    return None if v is None else float(v)


def inicializa(model, ruta):
    """Carga estricta del state_dict de `ruta` (un checkpoint de Lightning) en `model`. Devuelve los tensores cargados."""
    ck = torch.load(ruta, map_location="cpu")
    sd = ck["state_dict"]
    model.load_state_dict(sd, strict=True)
    print(f"[init] pesos INICIALIZADOS desde {ruta} (md5 {_md5(ruta)}, {len(sd)} tensores, carga estricta; "
          f"origen en la epoca {ck.get('epoch')}, paso {ck.get('global_step')})", flush=True)
    return len(sd)


def referencia(ruta):
    """Ultima validacion de la corrida de origen: (valor de METRICA, epoca). Si una epoca se repitio, vale la ultima."""
    fichero = os.path.join(os.path.dirname(ruta), "validacion.jsonl")
    filas = [json.loads(l) for l in open(fichero) if l.strip()]
    ultima = max(filas, key=lambda r: (r["epoca"], r["paso"]))
    return float(ultima[METRICA]), int(ultima["epoca"])


def comprueba(trainer, model, val_loader, ruta, tol):
    """Valida el modelo recien inicializado y aborta si no reproduce la ultima validacion de su corrida de origen."""
    ref, epoca = referencia(ruta)
    r = trainer.validate(model=model, dataloaders=val_loader, verbose=False)[0]
    v = float(r[METRICA])
    dif = abs(v - ref) / abs(ref)
    print(f"[init] validacion previa: {METRICA} {v:.5f} frente a {ref:.5f} de la corrida de origen (epoca {epoca}); "
          f"diferencia relativa {dif:.1e}, tolerancia {tol:.0e}", flush=True)
    if not dif <= tol:
        raise SystemExit(f"[init] ABORTO: el modelo inicializado desde {ruta} no reproduce su corrida de origen")
    return v, ref
