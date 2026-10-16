"""Objetivos auto-supervisados para pre-entrenar el encoder BEV sin rótulos de mapa.

TESIS (C2): el brazo B inicializa el encoder con `bevfusion-seg.pth`, entrenado con
supervisión de mapa HD. Aquí se construyen las dos alternativas que NO usan ninguna
etiqueta, y que son la contribución de la tesis:

  C_mae  reconstrucción de columnas BEV enmascaradas  (objetivo estático, estilo BEV-MAE)
  C_4d   predicción del presente a partir del pasado  (objetivo dinámico)
  F      ocupación FUTURA a 0,5-3 s, con etiquetas precalculadas de los barridos futuros
         (16/10/2026; docs/DISENO_C2_OBJETIVO_2026-10-03.md, estilo UnO)

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
    # TESIS (auditoria 09/10/2026): FILA = x, COLUMNA = y, la disposicion MEDIDA del BEV del
    # encoder (diagnostico_bev.py: IoU 0,69 frente a 0,27 traspuesta; el voxelizador emite
    # (x, y, z) y sparse_encoder deja (N, C*D, X, Y)), la misma de etiquetas_futuro.celdas.
    # Hasta el 09/10 aqui ponia "columna = x, fila = y, igual que la convencion del BEV", lo
    # que era falso: los objetivos de C_4d y C_mae quedaban TRASPUESTOS respecto del BEV (la
    # cabeza debia predecir en (x=a, y=b) lo que hay en (x=b, y=a)), sin ningun error, porque
    # la rejilla es cuadrada. Lo protege tools/prueba_rasteriza.py.
    fx = ((x - x0) / (x1 - x0) * H).long()
    cy = ((y - y0) / (y1 - y0) * W).long()
    ok = (fx >= 0) & (fx < H) & (cy >= 0) & (cy < W)
    idx = (fx[ok] * W + cy[ok])
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
        # TESIS (auditoria 09/10/2026): fila = x, columna = y, como rasteriza y el BEV.
        fx = ((puntos[:, 0] - x0) / (x1 - x0) * H).long().clamp(0, H - 1)
        cy = ((puntos[:, 1] - y0) / (y1 - y0) * W).long().clamp(0, W - 1)
        return puntos[~m[fx, cy]]

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


# --- F: ocupación FUTURA (estilo UnO) -----------------------------------------

class CabezaFutura(nn.Module):
    """Predice, para h = 0,5 ... 3,0 s, si cada celda del BEV ACTUAL estará ocupada o libre.

    TESIS (16/10/2026): objetivo F de docs/DISENO_C2_OBJETIVO_2026-10-03.md; piloto
    pre-registrado en docs/PREREGISTRO_PILOTO_F_2026-10-16.md. Las etiquetas se precalculan
    en la CPU (tesis_lidar/tools/precalcula_etiquetas_c2.py con etiquetas_futuro.ocupacion):
    -1 desconocida; 0 libre (atravesada por un rayo del keyframe futuro por debajo de la banda
    de obstáculos, o con retorno de suelo); 1 ocupada (retorno no-suelo del keyframe futuro),
    todo en el marco del LiDAR ACTUAL y en la rejilla del BEV (filas = x, columnas = y).

    Respaldo: UnO (Agro et al., CVPR 2024, arXiv 2406.08691) pronostica ocupación hasta 3 s
    con pseudo-etiquetas de rayos LiDAR y equilibra consultas ocupadas y libres. Aquí igual:
    en cada muestra y horizonte, la pérdida es la media de la BCE en las celdas ocupadas y la
    media en las libres, con el mismo peso; las desconocidas no cuentan. Una muestra-horizonte
    sin celdas ocupadas o sin libres no entra.

    La cabeza es la MISMA de C_4d (_Cabeza, oculto 128) con 6 salidas: la comparación con C_4d
    en la sonda es de objetivos, no de capacidad de la cabeza.
    """

    def __init__(self, c_in, n_horizontes=6):
        super().__init__()
        self.cabeza = _Cabeza(c_in, n_horizontes)
        self.n = n_horizontes

    def forward(self, bev, etiquetas):
        """`bev` (B,C,H,W) del encoder; `etiquetas` (B,n,H,W) int8 en el mismo dispositivo."""
        logit = self.cabeza(bev).float()
        assert logit.shape == etiquetas.shape, f"{tuple(logit.shape)} != {tuple(etiquetas.shape)}"
        bce = F.binary_cross_entropy_with_logits(logit, etiquetas.clamp(min=0).float(), reduction='none')
        ocu, lib = (etiquetas == 1).float(), (etiquetas == 0).float()
        n_o, n_l = ocu.sum((2, 3)), lib.sum((2, 3))
        l_o = (bce * ocu).sum((2, 3)) / n_o.clamp(min=1)
        l_l = (bce * lib).sum((2, 3)) / n_l.clamp(min=1)
        valido = (n_o > 0) & (n_l > 0)
        if not bool(valido.any()):
            return bev.new_zeros(()), {}
        por = 0.5 * (l_o + l_l)                     # (B, n)
        total = por[valido].mean()
        h = [float(por[:, k][valido[:, k]].mean().detach()) if bool(valido[:, k].any()) else float('nan')
             for k in range(self.n)]
        return total, dict(futuro=float(total.detach()), por_horizonte=h)
