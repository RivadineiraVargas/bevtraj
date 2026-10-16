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

Objetivo `futuro` (F, 16/10/2026; docs/PREREGISTRO_PILOTO_F_2026-10-16.md): ocupación futura a
0,5-3 s con etiquetas precalculadas por tesis_lidar/tools/precalcula_etiquetas_c2.py. El conjunto
de muestras es el del índice de esas etiquetas (escenas de train_val y val excluidas por las
listas OFICIALES, auditoría X27, y con 3 s de futuro); --muestras elige sobre él igual que el
script de las etiquetas, y se comprueba que cada etiqueta es de la nube que se carga.
    python unitraj/pretrain.py --objetivo futuro --etiquetas data/nuscenes/c2_etiquetas_v2 \
        --muestras 3000 --salida pretraining_ckpt/f_piloto.pth
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.bevtraj.bevfusion import BEVFusion
from models.bevtraj.pretrain_heads import CabezaMAE, Cabeza4D, CabezaFutura, rasteriza
from datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from utils.utils import set_seed

# Claves que el brazo C debe entregar. `heads.map.*` se deja fuera a propósito: son las
# 14 claves que en el brazo B vienen de supervisión de mapa, y la cabeza nunca se ejecuta.
PREFIJOS = ('encoders.lidar.', 'decoder.')


def junta(lote):
    return dict(puntos=[b['inputs']['points'] for b in lote],
                muestras=[b['data_samples'] for b in lote])


def md5_lista(idx):
    """El mismo md5 que tesis_lidar/tools/precalcula_etiquetas_c2.py."""
    return hashlib.md5(json.dumps([int(i) for i in idx]).encode()).hexdigest()


def comprueba_orientacion(nubes, etq, rango, z_suelo=-1.4):
    """TESIS (hábito 6: cada convención contra su medida). La etiqueta F a 0,5 s, ocupada, debe
    coincidir con la ocupación no-suelo de la nube de ENTRADA rasterizada como el BEV (filas = x)
    mucho más que traspuesta. Medido el 16/10 en 40 etiquetas de prueba (nube clave, no_suelo):
    IoU 0,54 directa frente a 0,04 traspuesta. Aquí la entrada son 10 barridos y el suelo se
    separa por altura (z < -1,4 m, como los rasgos crudos de la sonda)."""
    H, W = etq.shape[-2:]
    d, t = [], []
    for n, e in zip(nubes, etq):
        ocu = rasteriza(n[n[:, 2] > z_suelo], rango, (H, W))[0] > 0
        lab = e[0] == 1
        d.append(float((ocu & lab).sum()) / max(float((ocu | lab).sum()), 1.0))
        t.append(float((ocu & lab.T).sum()) / max(float((ocu | lab.T).sum()), 1.0))
    d, t = float(np.mean(d)), float(np.mean(t))
    print(f"[C2] orientacion de F: IoU con la ocupacion de la entrada {d:.3f} directa, {t:.3f} traspuesta",
          flush=True)
    assert d >= 0.15 and d > 2 * t, "las etiquetas F no estan en la disposicion del BEV (filas = x)"


def construye_modelo(cfg):
    mc = OmegaConf.to_container(cfg.MODEL.SENSOR_ENCODER, resolve=True)
    mc.pop('weight_path', None)          # C2 parte de init aleatoria, nunca del mapa
    return BEVFusion(**mc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--objetivo', choices=['mae', '4d', 'futuro'], required=True)
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
    ap.add_argument('--muestras', type=int, default=0,
                    help='N muestras al azar (con --semilla-datos) repartidas entre escenas')
    # TESIS: semilla PROPIA para elegir las nubes, separada de --semilla (pesos y orden).
    # Antes las dos eran la misma, el acoplamiento que base_dataset._submuestra corrige en
    # el entrenamiento: con dos semillas de pre-entrenamiento cambiaria A LA VEZ la
    # inicializacion y que nubes se vieron. Por defecto 0, lo que reproduce exactamente el
    # piloto del 30/09 (que corrio con --semilla 0).
    ap.add_argument('--semilla-datos', type=int, default=0)
    ap.add_argument('--excluir-escenas-de', nargs='*', default=[],
                    help='directorios ScenarioNet cuyas ESCENAS no deben verse al pre-entrenar')
    ap.add_argument('--etiquetas', default='',
                    help='(futuro) directorio de precalcula_etiquetas_c2.py; su indice.json da el conjunto')
    ap.add_argument('--salida', required=True)
    a = ap.parse_args()
    a.md5_seleccion = None

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
    if a.objetivo == 'futuro':
        # TESIS (16/10/2026): el conjunto es el del índice de las etiquetas, en su orden, para que
        # --muestras elija exactamente lo que eligió precalcula_etiquetas_c2.py (se comprueba por md5)
        assert a.etiquetas, '--objetivo futuro necesita --etiquetas'
        indice = json.load(open(f"{a.etiquetas}/indice.json"))
        idx = [int(i) for i in indice['muestras']]
        assert md5_lista(idx) == indice['md5_muestras'], 'indice.json alterado'
        assert max(idx) < len(ds), 'el indice no es de este dataset'
        print(f"[C2] conjunto de las etiquetas: {len(idx)} muestras de "
              f"{len({ds.get_data_info(i)['scene_token'] for i in idx})} escenas; "
              f"exclusion: {indice['exclusion']}; md5 {indice['md5_muestras']}", flush=True)
    # TESIS: el pre-entrenamiento no debe ver nubes de las escenas con que luego se
    # selecciona o se evalua el modelo. Aunque no use etiquetas, veria exactamente los
    # entornos de train_val/val y la seleccion dejaria de ser limpia. Se excluye por
    # ESCENA, no por muestra, porque las muestras vecinas de una escena son casi iguales.
    if a.excluir_escenas_de:
        # TESIS (16/10/2026): aqui habia un `import json` local; desde que json se importa arriba, ese
        # import hacia `json` LOCAL en todo main() y el codigo de F, que lo usa antes, moria con
        # UnboundLocalError (lo cazo el humo de tools/piloto_f.sh).
        nombre2tok = {s['name']: s['token'] for s in json.load(open(
            f"{sc['data_root']}/v1.0-trainval/scene.json"))}
        fuera = set()
        for d in a.excluir_escenas_de:
            import pickle as _pk
            summ = _pk.load(open(f"{d}/dataset_summary.pkl", 'rb'))
            fuera |= {nombre2tok[m['id'].split('_')[0]] for m in summ.values()}
        antes = len(idx)
        idx = [i for i in idx if ds.get_data_info(i)['scene_token'] not in fuera]
        print(f"[C2] excluidas {len(fuera)} escenas: {antes} -> {len(idx)} muestras", flush=True)
        # con 'futuro' el indice ya excluye las listas oficiales, que contienen a estas
        assert a.objetivo != 'futuro' or len(idx) == antes, "el indice de F contiene escenas excluidas"
    if a.muestras:
        rng = np.random.RandomState(a.semilla_datos)
        idx = sorted(rng.choice(idx, size=min(a.muestras, len(idx)), replace=False).tolist())
        n_esc = len({ds.get_data_info(i)['scene_token'] for i in idx})
        a.md5_seleccion = md5_lista(idx)
        print(f"[C2] {len(idx)} muestras al azar de {n_esc} escenas distintas; md5 {a.md5_seleccion}",
              flush=True)
    if a.limite:
        idx = idx[:a.limite]
    if a.objetivo == 'futuro':
        # cada etiqueta debe existir y ser de la nube que el dataset carga con ese índice
        for i in idx:
            nube = Path(ds.get_data_info(i)['lidar_points']['lidar_path']).name
            et = str(np.load(f"{a.etiquetas}/{i}.npz")['lidar'])
            assert et == nube, f"la etiqueta {i} es de {et}, el dataset carga {nube}"
        print(f"[C2] etiquetas F comprobadas: {len(idx)} de {len(idx)}, cada una de su nube", flush=True)
    sub = torch.utils.data.Subset(ds, idx)
    dl = DataLoader(sub, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    collate_fn=junta, drop_last=True)

    # el número de canales del BEV sale del propio config, no se supone
    c_bev = sum(cfg.MODEL.SENSOR_ENCODER.decoder.neck.out_channels)
    if a.objetivo == 'mae':
        cabeza = CabezaMAE(c_bev, rango, ratio=a.ratio).to(dev)
    elif a.objetivo == 'futuro':
        cabeza = CabezaFutura(c_bev).to(dev)
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
        t0 = time.time(); acum = []; saltadas = 0; por_h = []
        for k, lote in enumerate(dl):
            nubes = [p.to(dev, non_blocking=True) for p in lote['puntos']]

            if a.objetivo == 'futuro':
                etq = torch.from_numpy(np.stack([np.load(f"{a.etiquetas}/{int(m.sample_idx)}.npz")['F']
                                                 for m in lote['muestras']])).to(dev)
                if ep == 0 and k == 0:
                    comprueba_orientacion(nubes, etq, rango)
                entrada = nubes
                muestras = lote['muestras']
            elif a.objetivo == '4d':
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
                elif a.objetivo == 'futuro':
                    perdida, reg = cabeza(bev, etq)
                    if reg:
                        por_h.append(reg['por_horizonte'])
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
        if por_h:
            print(f"[C2] época {ep}: pérdida por horizonte (0,5 ... 3,0 s) "
                  + " ".join(f"{x:.4f}" for x in np.nanmean(np.array(por_h), 0)), flush=True)

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
                              semilla=a.semilla, semilla_datos=a.semilla_datos, config=a.config,
                              ratio=a.ratio, corte=a.corte, etiquetas=a.etiquetas,
                              md5_seleccion=a.md5_seleccion)),
               a.salida)
    print(f"[C2] guardadas {len(sd)} claves en {a.salida} "
          f"(encoders.lidar: {sum(1 for k in sd if k.startswith('encoders.lidar'))}, "
          f"decoder: {sum(1 for k in sd if k.startswith('decoder'))})", flush=True)


if __name__ == '__main__':
    main()
