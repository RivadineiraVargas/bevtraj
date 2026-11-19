# bevtraj (fork de tesis): registro de diseño

Qué cambia este fork respecto a [Kongminsang/bevtraj](https://github.com/Kongminsang/bevtraj), y por qué. Recoge los cambios principales, no todos. La base es `daf9795`, el `main` de los autores cuando se hizo el fork. Los autores reescribieron su historia el 03/11/2026 y su `main` actual no comparte commits con este.

Cada fila apunta al commit que hizo el cambio; su mensaje (en portugués) trae el detalle y las comprobaciones que se hicieron. Algunos commits agrupan varios cambios. En el código, cada parche lleva un comentario `TESIS:` con lo que fallaba antes y, casi siempre, la fecha. Cuando una afirmación resultó falsa, la corrigió un commit posterior, que lo dice. Algunos mensajes antiguos afirman cosas que después resultaron falsas (las marca la §2): donde no coincidan, vale esta tabla.

## 1. El objetivo

La tesis busca predecir trayectorias sin mapas HD, ni al predecir ni al pre-entrenar. BEVTraj predice para cada agente 10 trayectorias posibles en los próximos 6 s, a partir de los sensores y de la historia de movimiento, sin mapa al predecir; su encoder de sensores, en cambio, se inicializa con `bevfusion-seg.pth`, un checkpoint de segmentación de mapa BEV entrenado con etiquetas de mapa HD. De ahí las dos preguntas: cuánto aporta esa inicialización, que es lo que habría que sustituir, y si un pre-entrenamiento sin etiquetas puede sustituirla.

De ahí salen los tres requisitos que explican los cambios de abajo:

- **Solo LiDAR.** La tesis trabaja con el LiDAR y la historia de movimiento, sin cámara.
- **Brazos comparables.** Para atribuirle una diferencia al encoder, todo lo demás tiene que ser igual entre brazos, y tiene que poder comprobarse.
- **Un encoder que se pueda pre-entrenar sin etiquetas**, con las mismas piezas del predictor.

## 2. Entrenar solo con LiDAR en la máquina de la tesis

Para entrenar la variante solo LiDAR, con el mini y con el trainval, en una GPU de 8 GB y 15 GB de RAM, hizo falta lo siguiente. Ninguna de estas filas impide entrenar el config con cámara de los autores, y la última es un error del propio fork.

| commit | cambio | por qué |
|---|---|---|
| `283682c` | `configure_optimizers` crea grupos encoder/decoder con su `lr_scale`, y `get_lr` usa `group.get("lr_scale", 1.0)` | el original usaba un solo grupo; su `get_lr` exigía `lr_scale` en todos los grupos si lo tenía el primero, y lo ignoraba en todos si el primero no lo tenía. El mensaje del commit dice que solo leía el del primero: es falso (auditoría del 19/11/2026) |
| `283682c` | `min_lr` y `lr` se convierten a número | conversión defensiva. Con PyYAML a secas `1e-5` es texto, pero Hydra carga los configs con OmegaConf 2.3, que ya lo lee como número (comprobado en la imagen). El mensaje del commit dice que el coseno fallaba en la época 2: en este entorno no ocurre (auditoría del 19/11/2026) |
| `283682c` | `create_data.py` acepta `v1.0-mini` | probar con el mini sin bajar antes el trainval completo |
| `283682c` | checkpoints activados, acumulación de gradiente y registro de las claves cargadas y ausentes en `load_weights` | con `save_checkpoint` apagado el entrenamiento terminaba, pero sin checkpoints ni reanudación; la carga no decía qué faltaba |
| `283682c` | el `collate_fn` de `combined_nuscenes` admite muestras sin imágenes; config `bevtraj_nusc_lidar`, sin rama de cámara y con `fuser: null` | daba la cámara por supuesta (el config de los autores siempre la lleva), lo que impedía la variante solo LiDAR |
| `3498d97` | `create_data.py --token2idx-sufijo`; con `con_camara=False` no se carga NuScenes entero | los infos del trainval pisaban en silencio los del mini (los dos caminos existen desde que este fork añadió el mini); cargar NuScenes solo para las cajas 2D de las cámaras agotaba los 15 GB de RAM |
| `e9c9151` | `create_data.py --sin-camara` | regenerar los infos del trainval por la vía normal volvía a agotar los 15 GB de RAM |
| `1c0aa87` | la caché se construye con `cache_num_workers` procesos, y su `tmp/` va dentro de la propia caché | con un proceso por núcleo se agotaban los 15 GB de RAM; el `tmp/` relativo se compartía entre construcciones |
| `9a580cc` | `pretrain.py` sin el `import json` local | error del propio fork: volvía local `json` en todo `main()`, y el objetivo F moría con `UnboundLocalError` |

## 3. Brazos comparables

| commit | cambio | por qué |
|---|---|---|
| `e2833e5` | `freeze_sensor_encoder` congela también la BatchNorm del encoder | las estadísticas de BatchNorm son buffers, no parámetros, y Lightning devuelve las capas a modo de entrenamiento al empezar cada época: el encoder congelado seguía cambiando. La congelación es del propio fork (`283682c`); el original no congelaba |
| `3ebf801` | la reanudación automática pasa a ser explícita (`auto_resume`, apagada por omisión) | relanzar un brazo con el mismo `exp_name` reanudaba desde `last.ckpt` sin avisar: una corrida con otra semilla heredaba el estado de la anterior |
| `17e2ad2` | `final.ckpt` al terminar; el `DataLoader` de entrenamiento baraja en cada época; monitor configurable; validación por época sobre un subconjunto fijo (`val_subset_num`) | `last.ckpt` solo se escribía cada 1.000 pasos (lo había añadido `283682c`) y no era el modelo final; sin barajar, todas las épocas veían el mismo orden; validar entero en cada época costaba demasiado |
| `1c0aa87` | el subconjunto de entrenamiento sale de claves ordenadas y de un generador propio (`data_subset_seed`) | con el generador global, dos brazos o dos semillas podían entrenar con muestras distintas, y esa diferencia se le atribuiría al encoder |
| `1c0aa87` | la validación durante el entrenamiento usa `train_val`, con los infos de train | se selecciona en `train_val`; `val` queda para la evaluación final, que hace una herramienta de la tesis |
| `1bbe2e1` | `aislar_rng_encoder`: el decoder inicial es el mismo en todos los brazos | se midió que ya lo era; queda como garantía, con su prueba |
| `1bbe2e1` | cargar un checkpoint de fusión en un modelo sin fuser es un error (`permitir_ckpt_fusion_sin_fuser` lo permite, para repetir corridas antiguas) | un checkpoint de fusión cargaba sin un solo desajuste en el modelo solo LiDAR, y el encoder resultante estaba roto |
| `1bbe2e1` | las transformaciones agente↔BEV toman como centro el origen del LiDAR (`lidar_origin_offset`) | el BEV está centrado en el LiDAR, unos 0,94 m por delante del vehículo, y el original usaba la posición del vehículo como centro |
| `7b0c8c9` | la evaluación usa siempre los 9 barridos más cercanos (`test_mode`) | el pipeline de test también descartaba al azar 1 de cada 10 barridos al evaluar: dos pasadas del mismo modelo no daban lo mismo |
| `c199532` | la LR de cada época se fija al empezarla, como función de la época (`fija_lr_de_epoca`) | al reanudar a mitad de época, Lightning 2.1 avanzaba dos veces el programa de LR |
| `c199532` | el estado reanudable es `fin-epoca.ckpt`, escrito al cerrar cada época; reanudar desde otro punto es un error | reanudar a mitad de época (desde los checkpoints por pasos de `283682c`) repetía datos y desplazaba la LR |
| `e8d06d7` | brazo de control Z: el decoder recibe el BEV de otra escena (`shuffle_bev`) | dice si la escena aporta algo o si el modelo predice solo con la historia; no se usan ceros, porque una entrada nula fabrica ventajas espurias |

## 4. Métrica y registro

| commit | cambio | por qué |
|---|---|---|
| `a7aacbc` | brier-minFDE: al minFDE del mejor de los K modos se le suma (1 − p)², con p la probabilidad que el modelo le dio a ese modo (definición de Argoverse) | el código publicado no la calculaba; penaliza acertar una trayectoria sin creer en ella |
| `e9c9151` | el monitor y el nombre del checkpoint usan `brier_minFDE10` | los configs del piloto seleccionaban por la variante de un solo modo |
| `1bbe2e1` | EMA de los pesos entrenables (`ema_decay`), con `final_ema.ckpt` aparte; su estado va en los checkpoints y sobrevive a una reanudación | una lectura menos ruidosa del modelo final, sin tocar `final.ckpt` |
| `1bbe2e1`, `c199532` | `RegistraOrden` deja en el log el muestreador y los dos primeros lotes de cada época; `validacion.jsonl` guarda la LR de cada época | poder comprobar que dos brazos con la misma semilla ven los datos en el mismo orden y con la misma LR |
| `e20b0ea` | la loss del predictor se registra por época (`entrenamiento.jsonl`, `validacion.jsonl` y wandb) | el original la calculaba y no la registraba; solo se registra `loss.detach()`, así que el entrenamiento no cambia |

## 5. Pre-entrenamiento sin etiquetas

| commit | cambio | por qué |
|---|---|---|
| `fd49d3c` | `unitraj/pretrain.py` y `unitraj/models/bevtraj/pretrain_heads.py`, con dos objetivos: `mae` (oculta bloques del BEV y reconstruye ocupación, densidad y altura de las celdas ocultas) y `4d` (con los barridos antiguos predice la ocupación del instante actual) | pre-entrenar el encoder con las mismas piezas del predictor y sin anotaciones; la pérdida solo cuenta las celdas ocultas, porque con todas bastaría copiar la entrada |
| `fd49d3c` | el encoder pre-entrenado se guarda con la disposición de pesos de spconv que espera el gancho de carga | spconv 2.x entrega los pesos de las convoluciones dispersas en otro orden que `bevfusion-seg.pth`, y guardarlos tal cual rompía esas capas al cargar |
| `aa7371b` | objetivo `futuro` (F): `CabezaFutura` predice la ocupación del BEV de 0,5 a 3 s, con BCE de igual peso para celdas ocupadas y libres; las desconocidas no cuentan | objetivo inspirado en UnO, que cita el commit; la idea, no comprobada, es que predecir el futuro obliga al encoder a representar el movimiento, que es lo que necesita el predictor |
| `aa7371b` | el conjunto de F sale del índice de las etiquetas ya calculadas, comprobado por md5; cada etiqueta tiene que ser de la nube que se carga, y la orientación (filas = x) se mide en el primer lote | que el pre-entrenamiento use exactamente las nubes elegidas y que las etiquetas estén en la disposición del BEV |
| `c199532` | `rasteriza` y `oculta_puntos` usan filas = x, como el BEV del encoder | los objetivos estaban traspuestos respecto al BEV |
| `17e2ad2`, `e9c9151` | `--excluir-escenas-de` deja fuera escenas enteras de `train_val` y `val`; `--semilla-datos` separa la semilla de los datos de la de los pesos | que el pre-entrenamiento no vea las escenas de evaluación, y que cambiar la semilla de los pesos no cambie las nubes |

## 6. Regímenes de entrenamiento

| commit | cambio | por qué |
|---|---|---|
| `283682c`, `fd49d3c` | el encoder se puede congelar o tener su propia LR (`freeze_sensor_encoder`, `sensor_encoder_lr_mult`) | comparar el encoder tal como llega con el encoder ajustado |
| `d803266` | configs `calibra_{A,B}`: la receta del artículo en su versión v2 de arXiv, del 14/02/2026 (15 épocas, AdamW, calentamiento lineal y coseno hasta 5e-6; la v3 termina en 1e-5), con lote 8 como lote 1 por acumulación 8 | la GPU de la tesis tiene 8 GB |
| `62ca519` | configs `congela_{A,B,C}`: la misma receta con `freeze_sensor_encoder: true` | régimen congelado: aprende todo salvo el encoder de sensores |
| `dcca8c4`, `d3a665f` | `unitraj/utils/init_ckpt.py`: carga estricta de los pesos de otra corrida; antes de entrenar valida el modelo y aborta si no reproduce la última validación de la corrida de origen (`init_tol`) | cargar sin desajustes no prueba que el modelo funcione |
| `dcca8c4` | `encoder_bn_fijo`: la BatchNorm del encoder queda en modo eval aunque el encoder se entrene | ajustar el encoder con lote 1 sin mover sus estadísticas |
| `be1b177` | ajuste en dos etapas: el encoder con una LR 10 veces menor que la del decoder (enmienda de la tesis del 17/11, pendiente de confirmar), y calentamiento de 2 épocas | con la LR fijada por época, `warmup_epochs: 1` no calentaba: la época 0 ya iba al pico |
| `1231668` | configs `lpftcong_{B,C}` | el control del ajuste en dos etapas: las mismas épocas de más, con el encoder congelado |
| `b9aba26` | configs `congela_Arec_s{3,4,5}` | A congelado con la BatchNorm recalibrada; según su pre-registro en la tesis, para separar «encoder aleatorio» de «BatchNorm sin calibrar» |
| `165b8ee`, `b63bfa6` | `claves_permitidas`: filtro de escenas de entrenamiento por lista de claves, antes del barajado; configs `e2_{A,B,C}`: A con el encoder entrenable, como `calibra_A`; B y C congelados, como `congela_{B,C}` | régimen de pocas etiquetas; A se entrena entero porque es la referencia sin pre-entrenamiento (la asimetría está declarada en el pre-registro de la tesis); una lista que no cuadra con la caché (faltan claves u otro md5) es un error |

## 7. Comprobaciones

El fork no tiene una batería de tests propia. Las comprobaciones de cada cambio (pruebas en CPU, pruebas de humo con GPU, verificación funcional de los pesos cargados) están en el repositorio de la tesis (`tesis_lidar`), que es privado; cuando el mensaje de un commit dice cuáles se hicieron, es la referencia. Los configs de cada familia se generan desde una plantilla con un script de ese repositorio. Aquí cada config es un YAML completo, que no hereda de otros, aunque algunos piden ficheros externos: checkpoints, `e2_claves_25.json` y, en `lpft_*`, `+init_ckpt`.

## 8. Puntos abiertos

- **Las etiquetas de F** se calculan con un script del repositorio de la tesis (`precalcula_etiquetas_c2.py`) que todavía no está en el fork. Con el fork solo no se puede repetir el pre-entrenamiento de C.
- **A_rec y pocas etiquetas necesitan ficheros que no están en el fork:** el script que recalibra la BatchNorm y produce `a_rec_s{3,4,5}.pth`, y la lista de escenas `e2_claves_25.json`. Los dos están en el repositorio de la tesis.
- **La conversión del trainval a ScenarioNet** se hizo con una versión del conversor de los autores que comparte la
  memoria entre procesos (el original no cupo en los 15 GB de RAM de la máquina de la tesis); tampoco está en el fork todavía.
- **El ajuste en dos etapas, A_rec y pocas etiquetas** tienen sus pruebas en CPU en el repositorio de la tesis (de los commits, solo `dcca8c4` lo dice); falta la prueba de humo con GPU. La receta del
  ajuste en dos etapas de `be1b177` está pendiente de confirmar.
- **`tesis/c2-lpft` aún no está fusionada** en `tesis/patches-base`. Se fusionará cuando termine la confirmación de C1 de la tesis (en curso a 19/11/2026).
- **Solo nuScenes.** Los configs de Argoverse 2 de los autores no se tocaron ni se probaron con estos cambios.
- **GPU.** La imagen de los autores está compilada hasta sm_86; una RTX 50 o una H100 necesitarían reconstruirla con otra versión de PyTorch (sin probar), lo que cambiaría el entorno de todos los resultados.
