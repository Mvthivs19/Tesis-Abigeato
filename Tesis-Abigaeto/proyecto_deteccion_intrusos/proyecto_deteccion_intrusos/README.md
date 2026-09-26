# Sistema completo — Detección de intrusos vs. ganado (cámara NIR simulada)

Guía pensada para **Windows** (Asus TUF Gaming, GTX 1650 4GB, 8GB RAM), usando
PowerShell o Símbolo del sistema (CMD). Todos los comandos de abajo están
listos para copiar y pegar tal cual.

---

## Opción rápida: un solo script que hace todo

Si quieres correr todo el proceso sin ir copiando comando por comando, usa
`ejecutar_todo.ps1`. Automatiza los pasos 0 al 7 (instala dependencias,
descarga el dataset, simula NIR, prepara el dataset, entrena, genera un
video de prueba y te guía en los dos pasos interactivos: definir la zona y
correr el sistema final).

```powershell
cd Desktop\proyecto_deteccion_intrusos
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned    # solo la primera vez
.\ejecutar_todo.ps1
```

Por defecto corre en **modo prueba** (descarga solo 100 imágenes por clase,
para validar que todo funciona rápido). Cuando quieras el entrenamiento
real con más datos:

```powershell
.\ejecutar_todo.ps1 -Modo completo
```

El script se detiene automáticamente y te avisa en rojo si algún paso
falla, para que no sigas entrenando sobre un error. En los pasos 6 y 7
(interactivos, requieren que hagas clicks/tecles en una ventana) el script
te avisa antes de abrir la ventana correspondiente.

Si prefieres ir paso a paso manualmente (para entender o depurar cada
etapa), sigue la guía detallada más abajo.

---

## 0. Preparar el entorno (se hace UNA sola vez)

### 0.1 Verifica que tienes Python instalado

Abre **PowerShell** (búscalo en el menú de inicio) y escribe:

```powershell
python --version
```

Debe mostrarte algo como `Python 3.11.x`. Si te sale un error tipo
*"python no se reconoce como comando"*, instala Python desde
[python.org/downloads](https://www.python.org/downloads/) y **marca la
casilla "Add python.exe to PATH"** durante la instalación. Luego cierra y
vuelve a abrir PowerShell.

### 0.2 Ubícate en la carpeta del proyecto

Descomprime la carpeta `proyecto_deteccion_intrusos` (por ejemplo, en el
Escritorio) y navega ahí:

```powershell
cd Desktop\proyecto_deteccion_intrusos
```

### 0.3 Crea y activa el entorno virtual

```powershell
python -m venv venv
venv\Scripts\activate
```

Si PowerShell te muestra un error de **"la ejecución de scripts está
deshabilitada en este sistema"**, ejecuta esto una sola vez (permite scripts
solo para tu usuario, es seguro) y vuelve a intentar `venv\Scripts\activate`:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Cuando el entorno esté activo, tu línea de comandos empezará con `(venv)`.
**Debes ver ese `(venv)` antes de CADA comando de aquí en adelante** — si
cierras la terminal y la vuelves a abrir, tienes que activar de nuevo con
`venv\Scripts\activate`.

### 0.4 Instala las librerías (torch con GPU primero)

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Tarda varios minutos, es normal (son librerías pesadas). Al terminar,
verifica que tu GPU quedó detectada:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Debe imprimir: `True NVIDIA GeForce GTX 1650`. Si dice `False`, actualiza
los drivers de NVIDIA (GeForce Experience o nvidia.com/drivers) y repite el
`pip install torch...` de arriba.

---

## Los 7 pasos (en orden, cada uno depende del anterior)

> A partir de aquí, todos los comandos asumen que ya escribiste
> `venv\Scripts\activate` y ves `(venv)` en tu terminal.

### Paso 1 — Descargar el dataset

Antes de correrlo, abre `src\01_download_dataset.py` con el Bloc de notas
(o clic derecho → Editar) y busca la línea:

```python
SAMPLES_PER_CLASS = 400
```

Cámbiala a `100` para tu primera prueba (descarga mucho más rápido). Guarda
el archivo y ejecuta:

```powershell
python src\01_download_dataset.py
```

Vas a ver una barra de progreso. Al terminar debe decir
`Listo. Dataset crudo exportado en: data/raw_openimages`. Verifica que la
carpeta `data\raw_openimages\data` tenga archivos `.jpg`.

> **Nota Windows:** `fiftyone` usa una base de datos MongoDB interna que se
> descarga sola la primera vez. Si Windows Defender/Firewall te pregunta si
> permites la conexión, dale **"Permitir acceso"**.

### Paso 2 — Simular la cámara NIR

```powershell
python src\02_simulate_nir.py
```

Verás la barra "Simulando NIR". Al terminar, revisa que `data\nir_simulated`
tenga imágenes en blanco y negro.

### Paso 3 — Preparar el dataset para YOLO

```powershell
python src\03_prepare_yolo_dataset.py
```

Te mostrará cuántas imágenes quedaron en train/val/test. Genera
automáticamente `configs\data.yaml` (no lo edites a mano).

### Paso 4 — Entrenar el modelo

```powershell
python src\04_train.py
```

Es el paso más largo (de 20 minutos a varias horas, según cuántas imágenes
tengas). Verás una tabla que se actualiza en cada "época". **No cierres la
ventana** hasta que termine.

Si aparece un error rojo `CUDA out of memory`:
1. `Ctrl + C` para detenerlo.
2. Abre `src\04_train.py`, busca `batch=16` y cámbialo a `batch=8` (o `4`).
3. Vuelve a correr `python src\04_train.py`.

Al terminar, confirma que existe este archivo (es tu modelo entrenado):
```
runs_detect\intrusos_nir_v1\weights\best.pt
```

### Paso 5 (opcional) — Crear un video de prueba

Si no tienes un video propio para probar el sistema completo:

```powershell
python src\00_build_sample_video.py
```

Genera `data\sample_video.mp4`. Si ya tienes tu propio video, sáltate este
paso: abre `configs\config.yaml` con el Bloc de notas, busca `camera:` →
`source:` y reemplaza la ruta por la de tu archivo, por ejemplo:
```yaml
camera:
  source: "C:/Users/TuUsuario/Videos/mi_video.mp4"
```
(usa `/` en vez de `\` dentro del `.yaml`, o `\\` si prefieres mantener
backslash — ambos funcionan en Windows con Python).

### Paso 6 — Marcar la zona de vigilancia

```powershell
python src\05_define_zone.py
```

Se abre una ventana con el primer cuadro del video. Con **click izquierdo**
marca los puntos del área a vigilar (mínimo 3). Con la ventana de video
seleccionada (haz click sobre ella primero):
- `s` → guarda el polígono y cierra
- `r` → borra los puntos y empieza de nuevo
- `q` → sale sin guardar

### Paso 7 — Correr el sistema completo

```powershell
python src\06_realtime_pipeline.py
```

Se abre una ventana con el video, las detecciones (cajas de colores) y tu
zona dibujada en verde. Si detecta un humano dentro de la zona, verás en la
terminal mensajes como `[ALERTA LOCAL SIMULADA]` y `[DISUASIÓN SIMULADA]`,
y se guarda un clip en `data\event_clips`.

Para detener: haz click sobre la ventana de video y presiona `q`.

---

## Checklist rápido (para pegar y revisar mientras avanzas)

```
[ ] python --version funciona
[ ] cd Desktop\proyecto_deteccion_intrusos
[ ] python -m venv venv
[ ] venv\Scripts\activate   -> veo (venv) en la terminal
[ ] pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
[ ] pip install -r requirements.txt
[ ] GPU detectada = True
[ ] python src\01_download_dataset.py
[ ] python src\02_simulate_nir.py
[ ] python src\03_prepare_yolo_dataset.py
[ ] python src\04_train.py   -> existe runs_detect\intrusos_nir_v1\weights\best.pt
[ ] python src\00_build_sample_video.py   (si no tienes video propio)
[ ] python src\05_define_zone.py   -> guardé la zona con 's'
[ ] python src\06_realtime_pipeline.py   -> veo detecciones y alertas
```

---

## Problemas comunes en Windows

| Problema | Solución |
|---|---|
| `python no se reconoce como comando` | Reinstala Python marcando "Add to PATH", o usa `py` en vez de `python`. |
| Error al activar `venv\Scripts\activate` | Corre `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` en PowerShell. |
| `pip install` muy lento o se cae | Revisa tu conexión; reintenta el mismo comando (pip retoma donde quedó). |
| `CUDA out of memory` en el entrenamiento | Baja `batch` en `src\04_train.py` (16 → 8 → 4). |
| GPU no detectada (`False`) | Actualiza drivers NVIDIA y repite `pip install torch...` del paso 0.4. |
| La ventana de video no responde a `q`/`s`/`r` | Haz click sobre la ventana de video primero (debe estar en foco, no la terminal). |
| Windows Defender bloquea algo al descargar el dataset | Dale "Permitir acceso" cuando pregunte por conexión de red. |

---

## Estructura del proyecto

```
configs\
  config.yaml          <- toda la configuración del sistema (editar aquí)
  data.yaml             <- se genera solo (paso 3)
src\
  00_build_sample_video.py
  01_download_dataset.py
  02_simulate_nir.py
  03_prepare_yolo_dataset.py
  04_train.py
  05_define_zone.py
  06_realtime_pipeline.py   <- orquestador principal
  alert_system.py            <- RE12, RE13, RE14, RE19
  database.py                 <- eventos y salud (SQLite)
  health_monitor.py           <- RE21
  video_recorder.py           <- RE17
  utils.py                     <- config + logging compartidos
data\
  events.db              <- se crea sola al correr el paso 7
  event_clips\             <- clips + snapshots de evidencia
  logs\                     <- logs del sistema y de salud
runs_detect\               <- pesos y métricas del entrenamiento (paso 4)
```

## Próximos pasos sugeridos para la tesis

1. Corre todo el flujo con pocas imágenes (`SAMPLES_PER_CLASS=100`) para
   validar que no hay errores de punta a punta.
2. Sube el volumen de datos y vuelve a entrenar — es tu Objetivo específico
   4 (validar con precisión > 85% y falsos positivos < 5%). Las métricas
   quedan en `runs_detect\intrusos_nir_v1\` (curvas, matriz de confusión).
3. Con el modelo ya validado, corre el paso 7 y usa esos resultados
   (capturas, logs, base de datos de eventos) como evidencia de
   cumplimiento de los requerimientos (RE) en tu informe.
