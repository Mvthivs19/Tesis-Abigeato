"""
CARGA SEGURA DEL MODELO (PyTorch >= 2.6)

PyTorch 2.6 cambio el valor por defecto de ``torch.load(weights_only=...)`` a
``True``. Los checkpoints de Ultralytics 8.2.0 se serializaron antes de ese
cambio, asi que al deserializarlos el nucleo rechaza las clases de la red y
lanza ``UnpicklingError``.

Hay dos salidas posibles:

1. Poner ``weights_only=False``. Funciona, pero desactiva por completo la
   proteccion contra ejecucion de codigo arbitrario al deserializar, y obliga a
   confiar en cualquier `.pt` que se abra. Es el camino que no se elige aqui.
2. Declarar explicitamente que clases del checkpoint son legitimas. Se mantiene
   ``weights_only=True`` y solo se admite una lista cerrada y auditable de
   tipos de ``torch.nn`` y ``ultralytics.nn``, que es exactamente lo que el
   checkpoint de un YOLO propio contiene.

Este modulo aplica la opcion 2 en un unico punto del sistema, para que todos
los llamados (GUI, CLI, entrenamiento y evaluacion) se comporten igual y no
dependan de que el modelo se haya cargado antes o despues.

Nota de seguridad: esto no convierte un checkpoint ajeno en seguro. Un `.pt`
de origen desconocido sigue siendo arbitrario aunque se allowsliste el
contenedor; la proteccion solo cubre los tipos propios del modelo. La
confianza en el origen del archivo sigue siendo responsabilidad de quien lo
descarga.
"""

import logging

import torch

logger = logging.getLogger("model_loader")

# Tipos que aparecen dentro de una arquitectura YOLOv8. Son todos modulos de
# contenedor o capas elementales: no hay ninguna clase que abra un socket,
# ejecute un comando o lea una ruta al deserializarse.
SAFE_GLOBALS = [
    "ultralytics.nn.tasks.DetectionModel",
    "ultralytics.nn.modules.conv.Conv",
    "ultralytics.nn.modules.conv.Concat",
    "ultralytics.nn.modules.block.C2f",
    "ultralytics.nn.modules.block.Bottleneck",
    "ultralytics.nn.modules.block.SPPF",
    "ultralytics.nn.modules.block.DFL",
    "ultralytics.nn.modules.head.Detect",
    "torch.nn.modules.container.Sequential",
    "torch.nn.modules.container.ModuleList",
    "torch.nn.modules.conv.Conv2d",
    "torch.nn.modules.batchnorm.BatchNorm2d",
    "torch.nn.modules.activation.SiLU",
    "torch.nn.modules.linear.Identity",
    "torch.nn.modules.pooling.MaxPool2d",
    "torch.nn.modules.upsampling.Upsample",
]


def _resolve(path):
    """Importa cada ruta de `SAFE_GLOBALS` y devuelve los objetos."""
    objects = []
    for dotted in path:
        module, _, name = dotted.rpartition(".")
        try:
            obj = getattr(__import__(module, fromlist=[name]), name)
            objects.append(obj)
        except (ImportError, AttributeError) as exc:
            # Una version distinta de ultralytics puede mover los modulos. No
            # es motivo para abortar: se omite y el resto sigue funcionando.
            logger.debug("Clase segura no disponible en esta version: %s (%s)",
                         dotted, exc)
    return objects


# Memo en un diccionario y no en un booleano suelto: evita por construccion
# el UnboundLocalError de un `global` mal declarado, que dejaba el modelo sin
# cargar y hacia fallar toda la deteccion.
_MEMO = {"ready": False, "count": 0}


def allow_checkpoint_globals():
    """Registra las clases propias de YOLOv8 como seguras de deserializar.

    Idempotente. Devuelve True si la lista quedo aplicada.
    """
    if _MEMO["ready"]:
        return True
    try:
        if hasattr(torch.serialization, "add_safe_globals"):
            objects = _resolve(SAFE_GLOBALS)
            if objects:
                torch.serialization.add_safe_globals(objects)
                _MEMO["count"] = len(objects)
        # Si torch no tiene allowlist (versiones < 2.4) el valor por defecto ya
        # era False y no hay nada que hacer.
        _MEMO["ready"] = True
        logger.debug("Allowlist de checkpoint activa (%d clases)", _MEMO["count"])
        return True
    except Exception as exc:  # pragma: no cover - defensivo
        logger.error("No se pudo registrar la allowlist de checkpoint: %s", exc)
        return False


def load_checkpoint(path, map_location="cpu"):
    """Carga un checkpoint propio conservando `weights_only=True`."""
    allow_checkpoint_globals()
    return torch.load(path, map_location=map_location, weights_only=True)


def load_yolo(weights_path, device=None):
    """Carga un modelo YOLO desde disco y devuelve la instancia de `YOLO`.

    Lanza `RuntimeError` con un mensaje accionable si falla. Antes esta
    operacion se envolvia en un `except Exception: self.model = None`, que
    dejaba la aplicacion en vivo sin modelo y sin una sola pista de por que:
    el operador veia un video con cero detecciones y lo interpretaba como
    "no hay intrusos".
    """
    from ultralytics import YOLO

    allow_checkpoint_globals()
    try:
        model = YOLO(weights_path)
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo cargar el modelo '{weights_path}': {exc}. "
            "Si el error menciona 'weights only load failed', la version de "
            "PyTorch es incompatible con el checkpoint; revisa "
            "src/model_loader.py y requirements.txt."
        ) from exc
    if device:
        try:
            model.to(device)
        except Exception as exc:
            logger.warning("No se pudo mover el modelo a '%s': %s", device, exc)
    return model
