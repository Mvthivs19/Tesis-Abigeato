# ejecutar_todo.ps1
# ------------------------------------------------------------------
# Automatiza TODO el proceso: entorno virtual, dependencias, dataset,
# simulación NIR, entrenamiento y sistema en tiempo real.
#
# USO:
#   .\ejecutar_todo.ps1                 -> modo PRUEBA (rápido, 100 img/clase)
#   .\ejecutar_todo.ps1 -Modo completo  -> modo COMPLETO (400 img/clase)
#
# Si PowerShell no te deja ejecutar el script, corre esto una sola vez:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# ------------------------------------------------------------------

param(
    [ValidateSet("prueba", "completo")]
    [string]$Modo = "prueba"
)

$ErrorActionPreference = "Continue"

function Check-Paso($nombre) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR en: $nombre (codigo $LASTEXITCODE)." -ForegroundColor Red
        Write-Host "Revisa el mensaje de arriba, corrigelo, y vuelve a correr este script." -ForegroundColor Red
        exit 1
    }
}

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host " SISTEMA DE DETECCION DE INTRUSOS - EJECUCION COMPLETA" -ForegroundColor Cyan
Write-Host " Modo: $Modo" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

# ---------- Paso 0: entorno virtual ----------
Write-Host "`n[0/7] Preparando entorno virtual..." -ForegroundColor Green
if (-not (Test-Path "venv")) {
    python -m venv venv
    Check-Paso "creacion de entorno virtual"
} else {
    Write-Host "El entorno 'venv' ya existe, se reutiliza." -ForegroundColor Yellow
}

. .\venv\Scripts\Activate.ps1

if (-not $env:VIRTUAL_ENV) {
    Write-Host ""
    Write-Host "ERROR: no se pudo activar el entorno virtual 'venv'." -ForegroundColor Red
    Write-Host "Prueba borrando la carpeta 'venv' y volviendo a correr este script." -ForegroundColor Red
    exit 1
}
Write-Host "Entorno virtual activado correctamente: $env:VIRTUAL_ENV" -ForegroundColor Green

# ---------- Paso 0.4: dependencias ----------
Write-Host "`n[0.4/7] Instalando PyTorch con soporte CUDA (puede tardar varios minutos)..." -ForegroundColor Green
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
Check-Paso "instalacion de torch"

Write-Host "Instalando el resto de dependencias..." -ForegroundColor Green
pip install -r requirements.txt
Check-Paso "instalacion de requirements.txt"

Write-Host "`nVerificando GPU disponible..." -ForegroundColor Green
python -c "import torch; print('GPU disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'ninguna (se usara CPU)')"

# ---------- Configura modo prueba/completo ----------
if ($Modo -eq "prueba") {
    $env:SAMPLES_PER_CLASS = "100"
    Write-Host "`nModo PRUEBA: se descargaran ~100 imagenes por clase (rapido, para validar el pipeline)." -ForegroundColor Yellow
} else {
    $env:SAMPLES_PER_CLASS = "400"
    Write-Host "`nModo COMPLETO: se descargaran ~400 imagenes por clase (mas lento, mejores resultados)." -ForegroundColor Yellow
}

# ---------- Paso 1: descargar dataset ----------
Write-Host "`n[1/7] Descargando dataset (Open Images)..." -ForegroundColor Green
python src\01_download_dataset.py
Check-Paso "descarga del dataset"

# ---------- Paso 2: simular NIR ----------
Write-Host "`n[2/7] Simulando camara NIR nocturna..." -ForegroundColor Green
python src\02_simulate_nir.py
Check-Paso "simulacion NIR"

# ---------- Paso 3: preparar dataset YOLO ----------
Write-Host "`n[3/7] Preparando dataset en formato YOLO..." -ForegroundColor Green
python src\03_prepare_yolo_dataset.py
Check-Paso "preparacion del dataset YOLO"

# ---------- Paso 4: entrenar ----------
Write-Host "`n[4/7] Entrenando el modelo (esto puede tardar bastante, no cierres la ventana)..." -ForegroundColor Green
python src\04_train.py
Check-Paso "entrenamiento del modelo"

if (-not (Test-Path "runs_detect\intrusos_nir_v1\weights\best.pt")) {
    Write-Host "`nERROR: no se encontro runs_detect\intrusos_nir_v1\weights\best.pt tras el entrenamiento." -ForegroundColor Red
    exit 1
}
Write-Host "Modelo entrenado correctamente: runs_detect\intrusos_nir_v1\weights\best.pt" -ForegroundColor Green

# ---------- Paso 5: video de prueba (opcional, solo si no hay uno propio) ----------
Write-Host "`n[5/7] Generando video de prueba a partir de las imagenes NIR..." -ForegroundColor Green
Write-Host "(Si ya editaste 'camera.source' en configs\config.yaml con tu propio video, puedes ignorar este paso.)" -ForegroundColor Yellow
python src\00_build_sample_video.py
Check-Paso "generacion de video de prueba"

# ---------- Paso 6: definir zona (interactivo) ----------
Write-Host "`n[6/7] Definir zona de vigilancia (INTERACTIVO)." -ForegroundColor Magenta
Write-Host "Se abrira una ventana con el video. Instrucciones:" -ForegroundColor Magenta
Write-Host "  - Click izquierdo: agregar punto (minimo 3)" -ForegroundColor Magenta
Write-Host "  - Tecla 's': guardar y continuar" -ForegroundColor Magenta
Write-Host "  - Tecla 'r': reiniciar puntos" -ForegroundColor Magenta
Write-Host "  - Tecla 'q': salir sin guardar (se vigilara el frame completo)" -ForegroundColor Magenta
Read-Host "`nPresiona ENTER para abrir la ventana"
python src\05_define_zone.py

# ---------- Paso 7: pipeline en tiempo real (interactivo) ----------
Write-Host "`n[7/7] Ejecutando el sistema completo (INTERACTIVO)." -ForegroundColor Magenta
Write-Host "Se abrira una ventana con las detecciones en vivo." -ForegroundColor Magenta
Write-Host "Presiona 'q' sobre la ventana de video para detener el sistema." -ForegroundColor Magenta
Read-Host "`nPresiona ENTER para iniciar"
python src\06_realtime_pipeline.py

Write-Host "`n===================================================" -ForegroundColor Cyan
Write-Host " PROCESO COMPLETO FINALIZADO" -ForegroundColor Cyan
Write-Host " Revisa: data\events.db, data\event_clips, data\logs" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
