"""Objetivos auto-supervisados para pre-entrenar el encoder BEV sin rótulos de mapa.

TESIS (C2): el brazo B inicializa el encoder con `bevfusion-seg.pth`, entrenado con
supervisión de mapa HD. Aquí se construyen las dos alternativas que NO usan ninguna
etiqueta, y que son la contribución de la tesis:

  C_mae  reconstrucción de columnas BEV enmascaradas  (objetivo estático, estilo BEV-MAE)
  C_4d   predicción del presente a partir del pasado  (objetivo dinámico)

Las dos producen pesos para `encoders.lidar.*` y `decoder.*`, que son exactamente las
claves que `BEVFusion.load_weights` espera. No producen `heads.map.*`, que en el brazo B
son 14 claves entrenadas con mapa: el brazo C queda libre de ellas por construcción, y
como la cabeza de mapa nunca se ejecuta (BEVTraj sólo llama a `get_bev_feature`) su
ausencia no cambia nada del camino hacia adelante.

MEDIDO sobre nuScenes mini (23/09/2026): las muestras traen 10 barridos con desfases de
0,000 a 0,500 s en el canal 4, unos 32,5 k puntos en el keyframe y 24,4 k por barrido.
Ocho de 323 muestras son la primera de su escena, no tienen barridos y el cargador las
rellena duplicando el keyframe: en esas el canal 4 es todo cero y el objetivo dinámico
no tiene señal. `Cabeza4D` las descarta explícitamente.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- utilidades comunes -------------------------------------------------------

def rasteriza(puntos, rango, hw, canal_z=2):
    """Rasteriza una nube a una rejilla BEV (H, W) de ocupación, conteo y altura media.

    `rango` es el point_cloud_range del config; `hw` el tamaño del mapa BEV que sale
    del encoder. Devuelve tres tensores (H, W).
    """
    H, W = hw
    x0, y0, _, x1, y1, _ = rango
    x, y, z = puntos[:, 0], puntos[:, 1], puntos[:, canal_z]
    # columna = x, fila = y, igual que la convención del BEV del encoder
    cx = ((x - x0) / (x1 - x0) * W).long()
    cy = ((y - y0) / (y1 - y0) * H).long()
    ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
    idx = (cy[ok] * W + cx[ok])
    n = torch.zeros(H * W, device=puntos.device, dtype=torch.float32)
    n.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    sz = torch.zeros_like(n)
    sz.scatter_add_(0, idx, z[ok].float())
    ocu = (n > 0).float()
    alt = torch.where(n > 0, sz / n.clamp(min=1), torch.zeros_like(n))
    return ocu.view(H, W), n.view(H, W), alt.view(H, W)


class _Cabeza(nn.Module):
    """Cabeza convolucional ligera sobre el mapa BEV del encoder."""

    def __init__(self, c_in, c_out, oculto=128):
        super().__init__()
        self.red = nn.Sequential(
            nn.Conv2d(c_in, oculto, 3, padding=1), nn.BatchNorm2d(oculto), nn.ReLU(inplace=True),
            nn.Conv2d(oculto, oculto, 3, padding=1), nn.BatchNorm2d(oculto), nn.ReLU(inplace=True),
            nn.Conv2d(oculto, c_out, 1))

    def forward(self, x):
        return self.red(x)


# --- C_mae: reconstrucción de columnas BEV enmascaradas -----------------------

class CabezaMAE(nn.Module):
    """Predice ocupación, densidad y altura de las columnas BEV que se ocultaron.

    El enmascarado es por BLOQUES en BEV, no por punto: ocultar puntos sueltos es
    trivial de resolver por interpolación local y no obliga a entender la escena.
    Con bloque=4 celdas y celda=0,8 m, cada bloque son 3,2 m: del orden de un vehículo.

    La pérdida se calcula SÓLO en las celdas ocultas. Si se calculara en todas, el
    objetivo se resolvería copiando la entrada visible.
    """

    def __init__(self, c_in, rango, ratio=0.7, bloque=4, peso_alt=1.0, peso_cnt=1.0):
        super().__init__()
        self.cabeza = _Cabeza(c_in, 3)
        self.rango = rango
        self.ratio = ratio
        self.bloque = bloque
        self.peso_alt = peso_alt
        self.peso_cnt = peso_cnt

    def mascara(self, hw, device, generador=None):
        """Máscara booleana (H, W): True = oculto."""
        H, W = hw
        b = self.bloque
        hb, wb = (H + b - 1) // b, (W + b - 1) // b
        r = torch.rand(hb, wb, device=device, generator=generador)
        m = r < self.ratio
        return m.repeat_interleave(b, 0).repeat_interleave(b, 1)[:H, :W]

    def oculta_puntos(self, puntos, m, hw):
        """Quita de la nube los puntos que caen en bloques ocultos."""
        H, W = hw
        x0, y0, _, x1, y1, _ = self.rango
        cx = ((puntos[:, 0] - x0) / (x1 - x0) * W).long().clamp(0, W - 1)
        cy = ((puntos[:, 1] - y0) / (y1 - y0) * H).long().clamp(0, H - 1)
        return puntos[~m[cy, cx]]

    def forward(self, bev, nubes_completas, m):
        """`bev` (B,C,H,W) del encoder alimentado con la nube enmascarada."""
        B, _, H, W = bev.shape
        p = self.cabeza(bev)
        ocu_l, cnt_l, alt_l = p[:, 0], p[:, 1], p[:, 2]
        perdidas = []
        for i, nube in enumerate(nubes_completas):
            ocu, cnt, alt = rasteriza(nube, self.rango, (H, W))
            oc = m                                   # celdas ocultas
            if oc.sum() == 0:
                continue
            l_ocu = F.binary_cross_entropy_with_logits(ocu_l[i][oc], ocu[oc])
            # densidad y altura sólo donde de verdad hay algo que reconstruir
            hay = oc & (ocu > 0)
            if hay.sum() > 0:
                l_cnt = F.l1_loss(cnt_l[i][hay], torch.log1p(cnt[hay]))
                l_alt = F.l1_loss(alt_l[i][hay], alt[hay])
            else:
                l_cnt = bev.new_zeros(()); l_alt = bev.new_zeros(())
            perdidas.append(l_ocu + self.peso_cnt * l_cnt + self.peso_alt * l_alt)
        if not perdidas:
            return bev.new_zeros(()), {}
        total = torch.stack(perdidas).mean()
        return total, dict(mae=float(total.detach()))


# --- C_4d: predecir el presente a partir del pasado ---------------------------

class Cabeza4D(nn.Module):
    """El encoder ve sólo los barridos ANTIGUOS y predice la ocupación del presente.

    Es la hipótesis principal de C2: un objetivo que obliga a modelar movimiento
    debería transferir mejor a predicción de trayectorias que uno estático. La
    estructura fija se predice por sí sola; lo que exige trabajo son los objetos que
    se han desplazado entre el pasado y el instante actual.

    Horizonte real: 0,5 s, que es lo que abarcan los 10 barridos acumulados de nuScenes.
    Es corto, y hay que decirlo: no es predicción a 6 s, es modelado de movimiento a
    corto plazo que sirve de inicialización.
    """

    def __init__(self, c_in, rango, corte=0.25, peso_pos=3.0):
        super().__init__()
        self.cabeza = _Cabeza(c_in, 1)
        self.rango = rango
        self.corte = corte
        # la ocupación BEV es escasa: sin reponderar, predecir "vacío" en todo ya acierta
        self.peso_pos = peso_pos

    @staticmethod
    def separa(nube, corte, canal_t=4):
        """(pasado, presente) según el desfase temporal del canal 4."""
        t = nube[:, canal_t]
        return nube[t >= corte], nube[t < corte]

    @staticmethod
    def tiene_senal(nube, canal_t=4, minimo=3):
        """Falso en las muestras donde el cargador duplicó el keyframe (canal 4 todo cero)."""
        return int(torch.unique(nube[:, canal_t]).numel()) >= minimo

    def forward(self, bev, presentes):
        B, _, H, W = bev.shape
        logit = self.cabeza(bev)[:, 0]
        perdidas = []
        for i, pres in enumerate(presentes):
            ocu, _, _ = rasteriza(pres, self.rango, (H, W))
            w = torch.where(ocu > 0, self.peso_pos, 1.0)
            perdidas.append(F.binary_cross_entropy_with_logits(logit[i], ocu, weight=w))
        if not perdidas:
            return bev.new_zeros(()), {}
        total = torch.stack(perdidas).mean()
        return total, dict(c4d=float(total.detach()))
