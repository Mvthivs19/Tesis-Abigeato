# Sistema de Detección de Intrusos vs. Ganado

Vigilancia perimetral nocturna con YOLOv8, alertas con evidencia, escalating por
inacción, roles de acceso y evaluación de falsos positivos y retorno de
inversión.

Desarrollado y probado en **Windows** (Asus TUF Gaming, GTX 1650 4 GB, 8 GB RAM).
Todos los comandos son de PowerShell y están listos para copiar y pegar.

---

## Estado del modelo entregado

| Dato | Valor |
|---|---|
| Pesos | `runs_detect/intrusos_custom_v15/weights/best.pt` |
| Precision | 92.95 % |
| Recall | 86.30 % |
| mAP50 | 92.26 % |
| mAP50-95 | 77.96 % |
| Clases | `humano`, `bovino`, `equino`, `ovino`, `porcino` |
| Dataset | 24.309 train / 2.446 val / 1.780 test · 134.667 anotaciones |

**El modelo ya está entrenado.** No hace falta correr el paso de entrenamiento
para usar el sistema. El paso 4 de la guía más abajo solo sirve si quieres
generar otro modelo desde cero.

---

## Dos formas de usar el sistema

### A. Interfaz gráfica (recomendado)

```powershell
cd ruta\al\proyecto
venv\Scripts\activate
python main.py
```

Abre la aplicación completa: login por rol, video en vivo con HUD, historial,
salud del sistema, evaluación de FPR/ROI y configuración.

Credenciales iniciales:

| Usuario | Contraseña | Rol |
|---|---|---|
| `admin` | `admin123` | Administrador |
| `operador` | `operador123` | Visualizador |

> **Antes de instalar esto en una finca real, cambia ambas contraseñas** desde
> Configuración → Gestión de usuarios. Son credenciales de ejemplo y no deben
> quedar en un sistema desplegado.

### B. Línea de comandos

```powershell
python src\06_realtime_pipeline.py                  # detección con ventana
python src\06_realtime_pipeline.py --headless       # sin ventana
python src\06_realtime_pipeline.py --seconds 120    # corre 2 min y para
python src\06_realtime_pipeline.py --evaluate       # solo FPR + ROI y salir
```

Teclas: `q` detener · `ESPACIO` pausar · `a` confirmar · `f` falso positivo ·
`s` sincronizar cola.

Ambas interfaces consumen el **mismo** `AlertCoordinator`. No son dos
implementaciones: el horario, el filtro de calidad, la cola offline y la cadena
de alertas se comportan igual en las dos.

---

## Instalación desde cero

### 1. Verificar Python

```powershell
python --version
```

Debe mostrar `Python 3.11.x`. Si no está instalado, descárgalo de
[python.org/downloads](https://www.python.org/downloads/) y marca
**"Add python.exe to PATH"**.

### 2. Crear el entorno virtual

```powershell
cd ruta\al\proyecto
python -m venv venv
venv\Scripts\activate
```

Debes ver `(venv)` al principio de la línea. Si PowerShell bloquea el script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 3. Instalar las dependencias

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**El orden importa.** `requirements.txt` incluye `ultralytics`, que arrastra
`torch`. Si instalas `torch` sin soporte CUDA primero, el entrenamiento correrá
en CPU aunque tengas GPU. Verifica:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Debe imprimir una versión 2.7.x, `True` y el nombre de tu GPU. Si sale `False`,
actualiza los drivers de NVIDIA y repite el primer `pip install`.

### 4. Verificar el sistema

```powershell
python tests\test_sistema.py
python tests\test_gui_navigation.py
python tests\test_re18_dialog.py
python tests\test_roi_sensitivity.py
```

Cada uno debe terminar en `OK`. Si `pytest` no está instalado, estos scripts
funcionan igual: usan `unittest` de la biblioteca estándar.

| Archivo | Qué cubre |
|---|---|
| `test_sistema.py` | 130 pruebas: módulos, base de datos, RE y CLI |
| `test_gui_navigation.py` | Recorre la aplicación real: login por rol, todas las vistas renderizan y la navegación no deja hilos vivos |
| `test_re18_dialog.py` | El diálogo de falso positivo rama por rama, hasta la etiqueta YOLO escrita en disco |
| `test_roi_sensitivity.py` | Que el ROI responda a cada supuesto económico y no sea un número fijo |

---

## Configurar la zona de vigilancia

La zona delimita qué parte del encuadre se vigila. Sin ella el sistema
monitoriza el frame completo, que es un funcionamiento válido pero poco útil.

```powershell
python src\05_define_zone.py
```

Marca los puntos con **click izquierdo** (mínimo 3) y guarda con `s`.

También puedes editarla a mano en `configs/config.yaml` → `zone.polygon`.

> **Rehaz la zona antes de desplegar.** El polígono que viene en el repositorio
> está dibujado para un encuadre concreto: sobre el video de ejemplo
> (1080x1920, vertical) cubre solo el **37%** del frame y su borde inferior
> llega hasta y=776, mientras que las personas aparecen alrededor de
> y=1270-1440. Esas detecciones son correctas pero caen fuera de la zona, así
> que el sistema no dispara. Una zona mal trazada no produce una falsa alarma:
> produce **vigilancia muda**, que es peor porque el sistema parece correcto.
> Traza la zona sobre el encuadre real de la cámara y comprueba la cobertura;
> `tools\verificar_fpr_maquina.py` imprime ese diagnóstico.

---

## Estructura del proyecto

```
configs\
  config.yaml            <- toda la configuración (zona, horarios, umbrales)
  custom_data.yaml       <- dataset YOLO (lo genera el paso de preparación)
main.py                  <- punto de entrada de la interfaz gráfica
src\
  alert_coordinator.py   <- flujo compartido de alertas (RE02/04/11-15/17/19-21)
  alert_system.py        <- RE12, RE13, RE14, RE19: alertas y cola offline
  database.py            <- SQLite: eventos, contactos, auditoría, clips
  access_control.py      <- RE05: usuarios, roles y matriz de permisos
  schedule.py            <- RE02: ventanas horarias
  frame_filter.py        <- RE11: calidad de imagen
  integrity_monitor.py   <- RE21: compromiso físico de la cámara y FPS
  health_monitor.py      <- RE21: CPU, RAM, disco y logs
  video_recorder.py      <- RE17: clip probatorio pre + post evento
  contacts.py            <- RE04: directorio de emergencias
  retraining.py          <- RE18: exporta falsos positivos a formato YOLO
  fpr_evaluation.py      <- Objetivo 4: tasa de falsos positivos
  roi_analysis.py        <- Objetivo 4: retorno de la inversión
  model_loader.py        <- carga segura de checkpoints (PyTorch >= 2.6)
  gui\                   <- vistas de la interfaz (una por módulo)
  06_realtime_pipeline.py<- orquestador del CLI
tests\
  test_sistema.py        <- 130 pruebas: modulos, BD, RE y CLI
  test_gui_navigation.py  <- recorrido real de la GUI por rol y vista
  test_re18_dialog.py     <- dialogo de falso positivo, rama por rama
  test_roi_sensitivity.py <- que el ROI responde a cada supuesto
  smoke_deteccion.py     <- prueba de humo con el modelo real
tools\
  verificar_fpr_maquina.py <- recorre un clip con el modelo real y muestra
                              el veredicto, el informe y el diagnostico de zona
data\                    <- NO se versiona en Git (eventos, clips, logs)
runs_detect\             <- pesos y métricas del entrenamiento
```

---

## Los 21 requerimientos y su estado

### Verificados en software

| RE | Descripción | Dónde |
|---|---|---|
| RE01 | Zona de vigilancia perimetral | `zone.polygon`, `point_in_zone` |
| RE02 | Horario de vigilancia | `schedule.py`, `AlertCoordinator.can_detect` |
| RE03 | Umbrales configurables | `configs/config.yaml` |
| RE04 | Contactos de emergencia | `contacts.py` |
| RE05 | Roles y permisos | `access_control.py` |
| RE06 | Fuente de video configurable | `camera.source` |
| RE11 | Filtro de calidad de imagen | `frame_filter.py` |
| RE12 | Alerta local | `alert_system.py` |
| RE15 | Confirmación y escalamiento | `AlertCoordinator.acknowledge_last` |
| RE16 | Registro de eventos | `database.insert_event` |
| RE17 | Evidencia: snapshot + clip | `video_recorder.py` |
| RE18 | Exportación para reentrenamiento | `retraining.py` |
| RE19 | Operación sin conexión | cola en SQLite |
| RE20 | Sincronización manual | `AlertCoordinator.sync_pending` |
| RE21 | Monitoreo de integridad | `integrity_monitor.py` |

### Implementados pero **no verificables** en este entorno

Estos requisitos tienen el código escrito y las rutas de ejecución probadas, pero
no se pueden cerrar sin el hardware o los datos correspondientes. En la tesis
deben presentarse como «implementado, pendiente de validación en campo»:

| RE | Falta para verificarlo |
|---|---|
| RE13 | Gateway de SMS/email/webhook real |
| RE14 | Sirena, relé o GPIO físico |
| NIR | Sensor NIR y ground truth NIR (ver abajo) |
| Objetivo 4 (FPR < 5 %) | Clips grabados en la finca **sin intrusiones** |
| Objetivo 4 (ROI) | Costos, cobertura, energía y vida útil reales |

---

## Sobre la cámara NIR

El pipeline de imágenes usa el canal visible, **no NIR**. El paso
`src/02_simulate_nir.py` genera aproximaciones en escala de grises a partir de
imágenes de color, lo que sirve para probar el flujo pero **no reproduce la
respuesta espectral de un sensor infrarrojo real**.

Las conclusiones sobre el filtrado infrarrojo dependen de la banda concreta
(850 nm frente a 940 nm), la iluminación y el sensor. Ninguna afirmación sobre
el comportamiento NIR está respaldada por mediciones reales en este repositorio:
hay que decirlo así en la tesis o conseguir el sensor.

---

## Falsos positivos y reentrenamiento (RE18)

Cuando una alerta es un falso positivo, el Administrador puede marcarla desde el
panel de alertas o con la tecla `f`. El sistema pregunta **qué había realmente**
en la región:

- **Clase real declarada** → se escribe una etiqueta YOLO positiva con la caja
  detectada. El modelo aprende a corregir la clase que se equivocó.
- **Sin objeto real** (sombra, vehículo, maquinaria) → se escribe un archivo de
  etiqueta vacío. El modelo aprende a descartar ese fondo.

Nunca se etiqueta con la clase que *predijo* el modelo: eso validaría el error.

```powershell
# La exportación se dispara desde la GUI, pestaña de entrenamiento.
# Los archivos quedan en data/retraining/<sello>/:
#   images/   instantáneas
#   labels/   etiquetas YOLO
#   clips/    clips probatorios (contexto: una sombra se ve mejor en movimiento)
#   manifest.json  qué clase se declaró y si la muestra es positiva o negativa
```

El `COMO_REENTRENAR.txt` de cada exportación explica el siguiente paso.

> El modelo entregado **no se reentrena automáticamente**. El ciclo es
> deliberadamente manual: el dataset exportado se revisa antes de entrar al
> entrenamiento.

---

## Evaluación: FPR y ROI

Pestaña **Evaluación**, o por CLI con `--evaluate`. Los informes se escriben en
`data/evaluation/`.

### FPR (tasa de falsos positivos)

```yaml
evaluation:
  negative_clips:
    - data/sample_video.mp4
  negative_clips_verified: false   # ponlo en true solo si los clips no tienen intrusiones
  min_negative_minutes: 10
```

Mientras `negative_clips_verified` sea `false`, el informe declara el FPR como
**NO VERIFICABLE**, aunque muestre un porcentaje. Esto es a propósito: el set de
validación del proyecto contiene personas, así que un disparo allí puede ser una
detección correcta, y un «0 %» obtenido de esa forma no significa nada.

Para una cifra defendible:

1. Graba clips de la finca **sin intrusiones** (noches, lluvia, viento, sombras).
2. Anótalos en `evaluation.negative_clips`.
3. Pon `negative_clips_verified: true`.
4. Corre la evaluación y revisa que los minutos superen `min_negative_minutes`.

### ROI (retorno de la inversión)

El ROI sale de **supuestos económicos editables** en `configs/config.yaml`, no de
mediciones del sistema. Si el resultado supera el 400 %, la propia herramienta
emite una advertencia: ese número casi siempre indica un supuesto equivocado
(horas de vigilancia ahorradas, valor por animal) más que un sistema excelente.

**Reemplaza los montos por los tuyos antes de presentar el ROI.**

---

## Problemas comunes en Windows

| Problema | Solución |
|---|---|
| `python no se reconoce como comando` | Reinstala Python marcando "Add to PATH", o usa `py` en vez de `python`. |
| Error al activar `venv\Scripts\activate` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| GPU no detectada (`False`) | Actualiza drivers NVIDIA y repite el `pip install torch...` del paso 3. |
| `CUDA out of memory` al entrenar | Baja `batch` en `src\04_train.py` (8 → 4). |
| La ventana de video no responde a las teclas | Haz click sobre la ventana para darle foco, no sobre la terminal. |
| La GUI abre con la lista de módulos vacía | Falta iniciar sesión. `admin/admin123` es la cuenta inicial. |
| `No existe el archivo de pesos` | El modelo está en `runs_detect/intrusos_custom_v15/weights/best.pt`. Ajusta `model.weights_path` en `config.yaml`. |
| La cola offline sube y no baja | Es correcto si `alerts.remote.enabled` es `false`: nada sale. El HUD lo muestra como `REMOTO OFF` en vez de `COLA OFFLINE`. |

---

## Seguridad

- **Cambia las contraseñas iniciales** antes de cualquier despliegue.
- Los checkpoints se cargan con `torch.load(weights_only=True)` y una allowlist
  cerrada de clases (`src/model_loader.py`). El único punto que necesita el
  unpickler completo es el entrenamiento, y está acotado a esa llamada.
- `data/` está en `.gitignore`: la base de eventos, los clips y los logs no se
  versionan.
- Los rechazos de permisos quedan registrados en la tabla `audit_log`.

---

## Guía de pipeline completo (opcional)

Si quieres reconstruir el dataset y entrenar desde cero:

```powershell
python src\01_download_dataset.py    # descarga imágenes de Open Images
python src\02_simulate_nir.py        # simulación NIR (ver la advertencia arriba)
python src\03_prepare_yolo_dataset.py
python src\04_train.py               # el paso más largo
```

También existe `ejecutar_todo.ps1`, que automatiza los pasos completos. Ojo: **no
dejes que vuelva a entrenar si ya tienes el modelo v15**.

---

## Prueba de humo

```powershell
python tests\smoke_deteccion.py
```

Carga el modelo real, procesa `data/sample_video.mp4` e imprime las detecciones
por clase. Sirve para confirmar que pesos, configuración y entorno siguen
coherentes después de cualquier cambio.
