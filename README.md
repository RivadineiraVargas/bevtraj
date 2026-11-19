# bevtraj (fork de tesis)

BEVTraj ([Kong y otros, 2025](https://www.arxiv.org/abs/2509.10080)) solo con LiDAR, para una tesis de predicción de trayectorias sin mapas HD, ni al predecir ni al pre-entrenar el encoder.

Este repositorio es un fork de [Kongminsang/bevtraj](https://github.com/Kongminsang/bevtraj). El original predice las trayectorias de los agentes de la vía en los próximos 6 s a partir de las cámaras, el LiDAR y la historia de movimiento, sin usar mapas al predecir. Su encoder de sensores, en cambio, parte de un checkpoint de segmentación de mapa en vista de pájaro (BEV), `bevfusion-seg.pth`, entrenado con etiquetas de mapa HD: el artículo explica que pre-entrena el encoder en segmentación de mapa BEV porque el modelo no recibe mapas HD como entrada. Aquí se usa el mismo modelo sin cámara y solo cambia el origen del encoder de LiDAR, para medir cuánto aporta ese pre-entrenamiento y sustituirlo por uno sin etiquetas: aleatorio; pre-entrenado con etiquetas de mapa (`lidar-only-seg.pth`, de [BEVFusion](https://github.com/mit-han-lab/bevfusion)); o pre-entrenado de forma auto-supervisada, prediciendo qué celdas del BEV estarán ocupadas dentro de 0,5 a 3 s, sin anotaciones.

Comparar encoders no es solo cambiar un checkpoint. Para atribuirle una diferencia al encoder, todo lo demás tiene que ser igual entre brazos y comprobable: el mismo decoder inicial, los mismos datos en el mismo orden, la misma LR en cada época, un encoder congelado que de verdad no cambie (BatchNorm incluida) y una evaluación determinista. La rama `tesis/c2-lpft` tiene el código de los autores más los commits que lo hacen posible: la variante solo LiDAR, el pre-entrenamiento auto-supervisado, los regímenes de entrenamiento (congelado y en dos etapas), la métrica brier-minFDE, la EMA, la reanudación segura y los controles. [DESIGN.md](DESIGN.md) explica cada cambio y por qué se hizo.

Es parte de una tesis de doctorado del PPGI/CT/UFES (Brasil). Aquí solo hay código y configuraciones; los resultados son de la tesis y de un artículo en preparación. Las herramientas de la tesis (evaluación final, pruebas, generador de configs) están en su repositorio, `tesis_lidar`, que es privado. El mérito de BEVTraj es de sus autores y la licencia es la original. Su README (instalación, preparación de datos, entrenamiento con cámara) sigue abajo, sin cambios.

## Cómo se usa

Todo corre en el contenedor de los autores (`Dockerfile`, `compose.yaml`, imagen `bevtraj:cu118`): PyTorch 1.12.1 con CUDA 11.6, mmdet3d y spconv. PyTorch viene compilado hasta sm_86: corre en una RTX 4060 y, según las reglas de CUDA, debería correr en el resto de las RTX 30 y 40; en una RTX 50 no correría. Fuera de la RTX 4060, nada de esto está probado.

Cada experimento es un config de Hydra en `unitraj/configs/method/` y se elige con `method=<nombre>`. Todos salen de `bevtraj_nusc_lidar.yaml`, y dentro de una familia los brazos solo difieren en el nombre y en el checkpoint del encoder, salvo en `e2`, donde A entrena el encoder y B y C lo congelan:

| configs | qué |
|---|---|
| `bevtraj_nusc_lidar` | la plantilla solo LiDAR (sin rama de cámara, `use_camera: false` y `fuser: null`) de la que salen los demás. No es un brazo: apunta al checkpoint de fusión de los autores, y cargarlo sin fuser aborta a propósito (`1bbe2e1`) |
| `bevtraj_calibra_{A,B}` | la receta del artículo (versión v2 de arXiv), con el encoder entrenable |
| `bevtraj_congela_{A,B,C}` | la misma receta con el encoder de sensores congelado; se entrena todo lo demás |
| `bevtraj_congela_Arec_s{3,4,5}` | A congelado, con la BatchNorm del encoder recalibrada (`a_rec_s*.pth`, que no están en el fork) |
| `bevtraj_lpft_{A,B,C}` | ajuste en dos etapas: carga con `+init_ckpt=ckpt/<corrida de origen>/final.ckpt` los pesos de una corrida anterior (B y C, su congelado; A, `calibra_A`, como control de presupuesto) y entrena 5 épocas más con el encoder suelto y una LR baja; sin `+init_ckpt` aborta |
| `bevtraj_lpftcong_{B,C}` | el control del anterior: las mismas épocas de más, con el encoder congelado |
| `bevtraj_e2_{A,B,C}` | pocas etiquetas: se entrena con una lista fija de escenas (`e2_claves_25.json`, que no está en el fork); A con el encoder entrenable, B y C congelados |

Los brazos son tres. **A**: encoder aleatorio (`weight_path: null`). **B**: `lidar-only-seg.pth`. **C**: el encoder que produce `unitraj/pretrain.py --objetivo futuro`. Los checkpoints van en `pretraining_ckpt/` y no se distribuyen aquí. Los configs `piloto_*`, `confirma_*`, `tv_*` y `*_mini*` son de etapas anteriores y se conservan para poder repetirlas; los que cargan `bevfusion-seg.pth` sin fuser necesitan `BEVTRAJ_PERMITIR_FUSION_SIN_FUSER=1`.

### Entrenamiento mínimo

```bash
# 1. código
git clone -b tesis/c2-lpft https://github.com/RivadineiraVargas/bevtraj.git ~/bevtraj
cd ~/bevtraj

# 2. entorno: una vez por máquina, y de nuevo tras cada cambio de código (el código va dentro de la imagen)
docker compose build
docker compose run --rm bevtraj python docker/verify_install.py --expect-gpus 1
#    (en una RTX 30, añadir --expect-capability 8.6)

# 3. datos: nuScenes v1.0-trainval en data/nuscenes/; infos solo LiDAR con los nombres que
#    esperan los configs (BEVTrajTV_infos_*, sample_token2idx_*_trainval)...
docker compose run --rm bevtraj python mmdet3d_tools/create_data.py nuscenes \
    --root-path data/nuscenes --version v1.0 --out-dir data/nuscenes \
    --extra-tag BEVTrajTV --token2idx-sufijo _trainval --sin-camara
#    ...y los tres splits de predicción en ScenarioNet (2 s de pasado, 6 s de futuro)
for s in train train_val val; do
  docker compose run --rm bevtraj python md_scenarionet/convert_nuscenes.py \
      -d data/nuscenes/prediction_converted_$s --split $s --dataroot data/nuscenes --num_workers 1
done

# 4. entrenar un brazo (en tmux, para que sobreviva a una conexión caída)
docker compose run --rm bevtraj python unitraj/train.py \
    method=bevtraj_congela_A 'devices=[0]' seed=3 method.exp_name=congela_A_s3
```

Se entrena con `train` y se valida durante el entrenamiento con 1.000 muestras fijas de `train_val`. La evaluación final en `val`, una sola vez por experimento, la hace una herramienta de la tesis que no está en el fork; con estos configs habría que sobrescribir `val_data_path`, `ann_file`, `token2idx_dict_path` y `val_subset_num`. Con el trainval, el conversor de los autores no cupo en los 15 GB de RAM de la máquina de la tesis; se usó una versión que comparte la memoria entre procesos y que todavía no está en el fork.

Cada corrida deja en `ckpt/<exp_name>/` el modelo de la última época, sin elegir (`final.ckpt`), su versión EMA (`final_ema.ckpt`), el estado para reanudar (`fin-epoca.ckpt`) y el registro por época (`validacion.jsonl` y `entrenamiento.jsonl`). Una corrida cortada se reanuda relanzando el mismo comando con `+auto_resume=true`, y solo desde el final de una época.

El pre-entrenamiento de C (`unitraj/pretrain.py --objetivo futuro`) necesita las etiquetas de ocupación futura ya calculadas. El script que las calcula todavía no está en el fork: ver «Puntos abiertos» en [DESIGN.md](DESIGN.md).

## Ramas

- `main`: el código de los autores tal como estaba al hacer el fork (`daf9795`). Los autores reescribieron su historia el 03/11/2026 y su `main` actual ya no comparte commits con este; por eso GitHub muestra el fork «por delante» y «por detrás» a la vez.
- `tesis/patches-base`: los cambios con los que se obtuvieron los resultados actuales de la tesis (hasta `62ca519`).
- `tesis/c2-lpft`: `tesis/patches-base` más los experimentos siguientes (esta rama).

## Licencia

MIT, como el original: ver [LICENSE](LICENSE).

---

# BEVTraj

Trajectory prediction from raw sensor data without relying on HD maps.

## 🚧 Project Status

⚠️ This project is under active development.

- The paper is currently **under revision**
- The codebase is **not fully complete**
- APIs and implementations may change without notice
- Additional features and fixes will be updated soon

Please treat the current version as a **preview/research prototype** rather than a final release.


## 📄 Paper
[BEVTraj: Map-Free End-to-End Trajectory Prediction in Bird's-Eye View with Deformable Attention and Sparse Goal Proposals](https://www.arxiv.org/abs/2509.10080)

![BEVTraj Overview](figs/bevtraj.jpg)

---

## 🛠️ Installation

🔥 BEVTraj is powered by [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [UniTraj](https://github.com/vita-epfl/UniTraj/tree/main), [Scenarionet](https://github.com/metadriverse/scenarionet) and [BEVFusion](https://github.com/mit-han-lab/bevfusion).

BEVTraj provides a Docker environment with all required dependencies. If
Docker is available, follow the Docker setup below and then proceed directly
to **Data Preparation**. To install BEVTraj directly on the host, follow the
manual installation steps instead.

### 🐳 Docker

#### Prerequisites

- Docker Engine with the Compose plugin
- An NVIDIA GPU and driver
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

The CUDA toolkit is included in the image and does not need to be installed on
the host.

#### Build

```bash
docker compose build
```

#### Verify

```bash
docker compose run --rm bevtraj \
    python docker/verify_install.py --expect-gpus 1
```

Change `--expect-gpus` when a different minimum GPU count is required.

#### Run

Start the service and open a shell:

```bash
docker compose up -d
docker compose exec bevtraj bash
```

Stop the service:

```bash
docker compose down
```

Use a temporary container for a one-off command:

```bash
docker compose run --rm bevtraj \
    python unitraj/train.py method=bevtraj_nusc
```

### Manual Installation

#### 1. Create Conda Environment & Install PyTorch

```bash
conda create -n bevtraj python=3.9
conda activate bevtraj

# Install PyTorch (adjust CUDA version as needed)
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

#### 2. mmdetection3d Setup

Follow the official instructions from [mmdetection3d documentation](https://mmdetection3d.readthedocs.io/en/latest/get_started.html):

```bash
pip install -U openmim
mim install mmengine
mim install 'mmcv==2.1.0'
mim install 'mmdet==3.2.0'
```
```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d 
pip install -v -e .
```

Follow the official instructions from [BEVFusion ](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion):

```bash
python projects/BEVFusion/setup.py develop

# Replace cu118 with your actual CUDA version if needed
pip install spconv-cu118 
```

If there are errors related to `bev_pool` or `voxel_layer` extension modules, manually copy the prebuilt `.so` files to the following directories:

```bash
cp mmdetection3d/projects/BEVFusion/bevfusion/ops/bev_pool/bev_pool_ext.*.so
bevtraj/unitraj/models/bevtraj/bevfusion/ops/bev_pool/

cp mmdetection3d/projects/BEVFusion/bevfusion/ops/voxel/voxel_layer.*.so
bevtraj/unitraj/models/bevtraj/bevfusion/ops/voxel/
```

#### 3. Scenarionet Setup

Follow the official instructions from [Scenarionet](https://scenarionet.readthedocs.io/en/latest/install.html)

```bash
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e .

git clone https://github.com/metadriverse/scenarionet.git
cd scenarionet
pip install -e .
```

#### 4. BEVTraj Setup

```bash
cd ~/bevtraj
pip install -r requirements.txt
python setup.py develop
```

If you encounter any issues during setup, feel free to open an issue.

---

## 📦 Data Preparation

Please download the required datasets from the official links below:

- [nuScenes Dataset](https://www.nuscenes.org/nuscenes#download)
- [Argoverse 2 Dataset](https://argoverse.github.io/user-guide/getting_started.html)
 
```
BEVTraj/
├── data/
│   ├── nuscenes/
│   │   ├── v1.0-trainval/
│   │   └── ...
│   └── av2_sensor/
│       ├── train/
│       ├── val/
│       └── ...
```

### 📝 Generate Info Files for MMDetection3D

Run the following commands to generate the annotation info files required by MMDetection3D:

```bash
# Convert nuScenes
python mmdet3d_tools/create_data.py nuscenes \
    --root-path data/nuscenes \
    --version v1.0 \
    --out-dir data/nuscenes \
    --extra-tag BEVTraj

# Convert Argoverse 2
python mmdet3d_tools/create_data.py argo2 \
    --root-path data/av2_sensor \
    --version trainval \
    --out-dir data/av2_sensor \
    --extra-tag BEVTraj \
    --workers 8
```

### 🔄 Convert datasets for ScenarioNet

After MMDetection3D formatting, convert the data into ScenarioNet format using the following commands:

```bash
# nuScenes - Train split
python md_scenarionet/convert_nuscenes.py \
    -d data/nuscenes/train_converted \
    --split train \
    --dataroot data/nuscenes \
    --num_workers 1

# nuScenes - Val split
python md_scenarionet/convert_nuscenes.py \
    -d data/nuscenes/val_converted \
    --split val \
    --dataroot data/nuscenes \
    --num_workers 1

# Argoverse 2 - Train split
python md_scenarionet/convert_argoverse2.py \
    -d data/av2_sensor/train_converted \
    --raw_data_path data/av2_sensor/train \
    --num_workers 4

# Argoverse 2 - Val split
python md_scenarionet/convert_argoverse2.py \
    -d data/av2_sensor/val_converted \
    --raw_data_path data/av2_sensor/val \
    --num_workers 4
```

Download the pretrained BEVFusion segmentation checkpoint from the link below and place it in the `pretraining_ckpt/` directory:
- [BEVFusion github](https://github.com/mit-han-lab/bevfusion?tab=readme-ov-file#bev-map-segmentation-on-nuscenes-validation)
- [📎 Download pretrained weight](https://www.dropbox.com/scl/fi/8lgd1hkod2a15mwry0fvd/bevfusion-seg.pth?rlkey=2tmgw7mcrlwy9qoqeui63tay9&dl=1)

---

## Train

```bash
# nuscenes
python unitraj/train.py method=bevtraj_nusc

# argoverse 2 sensor
python unitraj/train.py method=bevtraj_av2sensor

```

To enable multi-GPU training using Distributed Data Parallel (DDP), configure the `devices` option in your `config.yaml` file.

For example:

```yaml
devices: [0, 1, 2, 3]
```

## License

This project is licensed under the MIT License.  

## Acknowledgments

This project builds upon the work of several open-source libraries and prior research.  
Special thanks to the authors of the following projects:

- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [UniTraj](https://github.com/vita-epfl/UniTraj)
- [ScenarioNet](https://github.com/metadriverse/scenarionet)

We thank the open-source community for their invaluable tools and contributions.
