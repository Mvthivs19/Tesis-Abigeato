# Sistema completo — Detección de intrusos vs. ganado (cámara NIR simulada)

Implementa de punta a punta lo descrito en tu tesis: generación del dataset
simulado, entrenamiento del modelo, y el sistema en tiempo real con zona de
vigilancia (RE01), procesamiento Edge simulado (RE06), alertas locales y
disuasión (RE12/RE14), notificación remota con reintentos offline (RE13/RE19),
grabación de evidencia (RE17) y monitoreo de salud del sistema (RE21).

## 0. Instalación

Tu PC (Asus TUF Gaming, GTX 1650 4GB, 8GB RAM) sí sirve para entrenar con GPU,
solo hay que instalar torch con soporte CUDA **antes** que el resto:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Verifica que la GPU quedó detectada:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Debería imprimir `True NVIDIA GeForce GTX 1650`. Si imprime `False`, revisa que
tus drivers de NVIDIA estén actualizados y reinstala torch con el comando de
arriba.

## Flujo completo (en orden)

### 1. Descargar dataset público (Open Images)
```bash
python src/01_download_dataset.py
```
Empieza con `SAMPLES_PER_CLASS = 100` (edítalo en el script) solo para probar
que todo funciona, luego súbelo a 400-1000 para el entrenamiento real.

### 2. Simular cámara NIR nocturna
```bash
python src/02_simulate_nir.py
```
Convierte las fotos RGB descargadas en imágenes que emulan un sensor NIR
bajo iluminación IR activa (escala de grises, contraste, viñeteado, ruido).

### 3. Preparar dataset en formato YOLO
```bash
python src/03_prepare_yolo_dataset.py
```
Genera `configs/data.yaml` y divide en train/val/test (80/10/10).

### 4. Entrenar el modelo (YOLOv8n, transfer learning)
```bash
python src/04_train.py
```
Ajustado para tu GPU: `imgsz=416`, `batch=16`, `device=0`, `amp=True`.
Si te sale **"CUDA out of memory"**, edita `src/04_train.py` y baja `batch`
a 8 o 4. Al terminar, los pesos quedan en:
`runs_detect/intrusos_nir_v1/weights/best.pt`

### 5. (Opcional) Generar un video de prueba
Si no tienes un video propio para probar el sistema en tiempo real:
```bash
python src/00_build_sample_video.py
```
Esto arma `data/sample_video.mp4` a partir de tus imágenes NIR simuladas.
Si ya tienes un video propio, solo cambia `camera.source` en
`configs/config.yaml` para que apunte a ese archivo.

### 6. Definir la zona de vigilancia (RE01)
```bash
python src/05_define_zone.py
```
Se abre el primer frame del video: haz click para marcar los puntos del
polígono, presiona `s` para guardar (mínimo 3 puntos) o `q` para salir sin
guardar. Queda registrado en `configs/config.yaml` → `zone.polygon`.

### 7. Correr el sistema completo en tiempo real
```bash
python src/06_realtime_pipeline.py
```
Esto:
- Lee el "flujo de video" simulado, frame por frame, limitado a los FPS
  configurados (emulando un dispositivo Edge de bajo consumo — RE06).
- Corre el modelo YOLOv8n entrenado sobre cada frame.
- Descarta detecciones fuera del polígono de vigilancia (RE01).
- Si detecta un humano dentro de la zona:
  - Dispara la alerta local simulada (RE12) y la disuasión simulada (RE14).
  - Guarda una foto (snapshot) y un clip de video con contexto antes/después
    del evento (RE17).
  - Registra el evento en `data/events.db` (SQLite) y encola una
    notificación remota (RE13).
- Un hilo aparte (`health_monitor.py`) vigila CPU/RAM cada minuto y registra
  anomalías (RE21).
- Aunque no haya Internet, el sistema sigue grabando y alertando localmente;
  las notificaciones remotas pendientes se reintentan solas cada 30
  segundos (RE19).

Presiona `q` en la ventana de video para detener el sistema ordenadamente.

## Estructura del proyecto

```
configs/
  config.yaml          # toda la configuración del sistema (editar aquí)
  data.yaml             # generado automáticamente (paso 3)
src/
  00_build_sample_video.py
  01_download_dataset.py
  02_simulate_nir.py
  03_prepare_yolo_dataset.py
  04_train.py
  05_define_zone.py
  06_realtime_pipeline.py   # orquestador principal
  alert_system.py           # RE12, RE13, RE14, RE19
  database.py                # persistencia de eventos y salud (SQLite)
  health_monitor.py          # RE21
  video_recorder.py          # RE17
  utils.py                    # config + logging compartidos
data/
  events.db              # se crea solo al correr el pipeline
  event_clips/            # clips + snapshots de evidencia
  logs/                    # logs del sistema y de salud
runs_detect/               # pesos y métricas del entrenamiento (paso 4)
```

## Si tu PC se atasca en algún paso

- **Descarga (paso 1)**: baja `SAMPLES_PER_CLASS`.
- **Entrenamiento (paso 4)**: baja `batch` a 8 o 4; baja `imgsz` a 320.
- **Tiempo real (paso 6)**: baja `camera.fps_limit` en `config.yaml` (por
  ejemplo a 5), o usa `device: "cpu"` en `model.device` si por algún motivo
  la GPU no está disponible en ese momento (será más lento pero funcional).

## Notificación remota (RE13) — opcional

Por defecto `alerts.remote.enabled: false` en `config.yaml` (para que el
sistema funcione sin configuración extra). Si quieres activarla, necesitas
un endpoint HTTP que reciba `POST` con los campos `event_id`, `class_name`,
`confidence` y el archivo `snapshot` (por ejemplo, un webhook de Telegram,
Discord, o un servidor propio). Pon esa URL en `alerts.remote.webhook_url`
y cambia `enabled: true`.

## Próximos pasos sugeridos para la tesis

1. Corre el flujo completo con pocas imágenes (paso 1 con
   `SAMPLES_PER_CLASS=100`) para validar que no hay errores.
2. Sube el volumen de datos y vuelve a entrenar — esto es tu
   Objetivo específico 4 (validar con precisión > 85% y falsos
   positivos < 5%). Las métricas quedan en
   `runs_detect/intrusos_nir_v1/` (curvas, matriz de confusión, etc.).
3. Con el modelo ya validado, corre el pipeline completo (paso 6) y
   usa esos resultados (capturas de pantalla, logs, base de datos de
   eventos) como evidencia de cumplimiento de los RE en tu informe.
