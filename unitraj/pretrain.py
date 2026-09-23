"""Pre-entrenamiento auto-supervisado del encoder BEV. Contribución C2 de la tesis.

Produce un checkpoint intercambiable con `pretraining_ckpt/bevfusion-seg.pth`, de modo
que un brazo C se lanza igual que el brazo B: cambiando `weight_path` en el config.

Por qué no usa el dataset combinado: el pre-entrenamiento no necesita trayectorias ni
ScenarioNet, sólo nubes. Usar `SENSOR_DATASET` directamente evita construir la caché de
trayectorias y además aprovecha TODOS los keyframes, no sólo los que tienen un objetivo
de predicción válido.

Uso:
    python unitraj/pretrain.py --objetivo mae --config bevtraj_nusc_lidar_mini \
        --epocas 10 --batch 2 --salida pretraining_ckpt/c_mae.pth
"""
import argparse, os, sys, time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.bevtraj.bevfusion import BEVFusion
from models.bevtraj.pretrain_heads import CabezaMAE, Cabeza4D
from datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from utils.utils import set_seed

# Claves que el brazo C debe entregar. `heads.map.*` se deja fuera a propósito: son las
# 14 claves que en el brazo B vienen de supervisión de mapa, y la cabeza nunca se ejecuta.
PREFIJOS = ('encoders.lidar.', 'decoder.')


def junta(lote):
    return dict(puntos=[b['inputs']['points'] for b in lote],
                muestras=[b['data_samples'] for b in lote])


def construye_modelo(cfg):
    mc = OmegaConf.to_container(cfg.MODEL.SENSOR_ENCODER, resolve=True)
    mc.pop('weight_path', None)          # C2 parte de init aleatoria, nunca del mapa
    return BEVFusion(**mc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--objetivo', choices=['mae', '4d'], required=True)
    ap.add_argument('--config', default='bevtraj_nusc_lidar_mini')
    ap.add_argument('--epocas', type=int, default=10)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--acumula', type=int, default=4)
    ap.add_argument('--semilla', type=int, default=0)
    ap.add_argument('--ratio', type=float, default=0.7, help='fracción oculta (mae)')
    ap.add_argument('--corte', type=float, default=0.25, help='frontera pasado/presente en s (4d)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limite', type=int, default=0, help='usar sólo N muestras')
    ap.add_argument('--salida', required=True)
    a = ap.parse_args()

    set_seed(a.semilla)
    raiz = Path(__file__).resolve().parent
    cfg = OmegaConf.load(raiz / 'configs' / 'method' / f'{a.config}.yaml')
    rango = OmegaConf.to_container(cfg.point_cloud_range, resolve=True)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelo = construye_modelo(cfg).to(dev)
    n_par = sum(p.numel() for p in modelo.parameters())
    print(f"[C2] encoder construido: {n_par/1e6:.1f} M parámetros, init ALEATORIA", flush=True)

    sc = OmegaConf.to_container(cfg.TRAIN_DATASET.SENSOR_DATASET, resolve=True)
    ds = NuScenesDataset(**sc)
    idx = list(range(len(ds)))
    if a.limite:
        idx = idx[:a.limite]
    sub = torch.utils.data.Subset(ds, idx)
    dl = DataLoader(sub, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    collate_fn=junta, drop_last=True)

    # el número de canales del BEV sale del propio config, no se supone
    c_bev = sum(cfg.MODEL.SENSOR_ENCODER.decoder.neck.out_channels)
    if a.objetivo == 'mae':
        cabeza = CabezaMAE(c_bev, rango, ratio=a.ratio).to(dev)
    else:
        cabeza = Cabeza4D(c_bev, rango, corte=a.corte).to(dev)

    opt = torch.optim.AdamW(list(modelo.parameters()) + list(cabeza.parameters()),
                            lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=max(1, a.epocas * len(dl) // a.acumula), pct_start=0.1)
    escala = torch.cuda.amp.GradScaler(enabled=(dev == 'cuda'))

    print(f"[C2] objetivo={a.objetivo}  muestras={len(sub)}  batch={a.batch} "
          f"x{a.acumula} acumulados  épocas={a.epocas}", flush=True)

    paso = 0
    for ep in range(a.epocas):
        modelo.train(); cabeza.train()
        t0 = time.time(); acum = []; saltadas = 0
        for k, lote in enumerate(dl):
            nubes = [p.to(dev, non_blocking=True) for p in lote['puntos']]

            if a.objetivo == '4d':
                # descartar las muestras sin señal temporal (keyframe duplicado)
                pares = [Cabeza4D.separa(n, a.corte) for n in nubes
                         if Cabeza4D.tiene_senal(n)]
                pares = [(p, q) for p, q in pares if len(p) > 100 and len(q) > 100]
                saltadas += len(nubes) - len(pares)
                if not pares:
                    continue
                entrada = [p for p, _ in pares]
                objetivo = [q for _, q in pares]
                muestras = lote['muestras'][:len(pares)]
            else:
                m = cabeza.mascara((128, 128), dev)   # tamaño real se comprueba abajo
                entrada = [cabeza.oculta_puntos(n, m, (128, 128)) for n in nubes]
                entrada = [e for e in entrada if len(e) > 100]
                if len(entrada) != len(nubes):
                    saltadas += len(nubes) - len(entrada); continue
                objetivo = nubes
                muestras = lote['muestras']

            with torch.cuda.amp.autocast(enabled=(dev == 'cuda')):
                bev = modelo.get_bev_feature(
                    dict(points=entrada, imgs=None), muestras)
                if paso == 0:
                    print(f"[C2] mapa BEV: {tuple(bev.shape)}", flush=True)
                    assert bev.shape[1] == c_bev, f"canales {bev.shape[1]} != {c_bev}"
                if a.objetivo == 'mae':
                    assert bev.shape[-2:] == (128, 128), \
                        f"BEV {tuple(bev.shape[-2:])}: ajustar el tamaño de la máscara"
                    perdida, reg = cabeza(bev, objetivo, m)
                else:
                    perdida, reg = cabeza(bev, objetivo)

            escala.scale(perdida / a.acumula).backward()
            acum.append(float(perdida.detach()))
            if (k + 1) % a.acumula == 0:
                escala.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    list(modelo.parameters()) + list(cabeza.parameters()), 5.0)
                escala.step(opt); escala.update(); opt.zero_grad(set_to_none=True)
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
                paso += 1
            if k % 50 == 0 and acum:
                vram = torch.cuda.max_memory_allocated() / 2**30 if dev == 'cuda' else 0
                print(f"  ep{ep} {k}/{len(dl)}  pérdida {np.mean(acum[-50:]):.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  VRAM {vram:.2f} GB", flush=True)
        print(f"[C2] época {ep}: pérdida {np.mean(acum):.4f}  "
              f"({time.time()-t0:.0f} s, {len(acum)} lotes, {saltadas} muestras saltadas)",
              flush=True)

        guarda(modelo, a, ep, np.mean(acum) if acum else float('nan'))
    print("[C2] terminado")


def a_layout_spconv1(sd):
    """Pasa los pesos de convolución dispersa al layout con que se publicó bevfusion-seg.pth.

    MEDIDO (23/09/2026): `state_dict()` de spconv 2.x entrega (Cout,kD,kH,kW,Cin), su
    layout nativo, mientras que `bevfusion-seg.pth` guarda (kD,kH,kW,Cin,Cout), el de
    spconv 1.x. El gancho de carga de spconv convierte SIEMPRE asumiendo el layout
    viejo, así que un checkpoint guardado en el nativo se permuta una vez de más y
    revienta con desajuste de forma. Guardar en el layout viejo es lo que hace que el
    brazo C sea de verdad intercambiable con el B.

    La permutación (1,2,3,4,0) se verificó contra bevfusion-seg.pth en las 21 claves de
    convolución dispersa. Los pesos 2-D de SECOND/SECONDFPN son de 4 dimensiones y no
    se tocan.
    """
    fuera = {}
    for k, v in sd.items():
        fuera[k] = v.permute(1, 2, 3, 4, 0).contiguous() if v.dim() == 5 else v
    return fuera


def guarda(modelo, a, ep, perdida):
    sd = {k: v.cpu() for k, v in modelo.state_dict().items() if k.startswith(PREFIJOS)}
    sd = a_layout_spconv1(sd)
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=sd,
                    meta=dict(objetivo=a.objetivo, epoca=ep, perdida=float(perdida),
                              semilla=a.semilla, config=a.config,
                              ratio=a.ratio, corte=a.corte)),
               a.salida)
    print(f"[C2] guardadas {len(sd)} claves en {a.salida} "
          f"(encoders.lidar: {sum(1 for k in sd if k.startswith('encoders.lidar'))}, "
          f"decoder: {sum(1 for k in sd if k.startswith('decoder'))})", flush=True)


if __name__ == '__main__':
    main()
