"""
RE18: Exportacion de falsos positivos para reentrenamiento del modelo.

El ciclo de mejora continua del sistema funciona asi:

  1. El operador (Administrador) revisa los eventos y marca los que no eran
     intrusiones reales, con el atajo `F` o el boton "Falso positivo".
  2. Periodicamente se ejecuta la exportacion: este modulo copia los snapshots
     de esos eventos a una carpeta de dataset y escribe los archivos de
     etiqueta YOLO correspondientes.
  3. Esas muestras se agregan al dataset de entrenamiento y se reentrena.

La clave es generar etiquetas YOLO correctas. Un falso positivo NO se etiqueta
con la clase que el modelo detecto (eso seria validar el error), sino con la
clase real que el operador declara, o como negativo cuando no se declara nada.
Entrenar con la clase equivocada ensena el error al modelo.
"""

import json
import os
import shutil
from datetime import datetime

try:
    from src.utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger("retraining")

# Clase para muestras sin objeto legitimo: obliga al modelo a no disparar.
BACKGROUND = "background"


class FalsePositiveExporter:
    def __init__(self, config, db):
        self.config = config
        self.db = db
        self.cfg = config.get("retraining", {})
        self.output_dir = self.cfg.get("output_dir", "data/false_positive_dataset")
        self.copy_snapshots = self.cfg.get("copy_snapshots", True)
        self.copy_clips = self.cfg.get("copy_clips", False)
        self.class_names = config.get("classes", {}).get(
            "names", ["humano", "bovino", "equino", "ovino", "porcino"]
        )

    def _class_index(self, name):
        if not name:
            return None
        try:
            return self.class_names.index(name)
        except ValueError:
            return None

    @staticmethod
    def _parse_bbox(bbox):
        """`bbox` viene como "x1,y1,x2,y2" en pixeles. Devuelve la tupla, o
        None si no hay caja utilizable."""
        if not bbox:
            return None
        try:
            parts = [float(v) for v in str(bbox).replace(" ", "").split(",")]
        except (TypeError, ValueError):
            return None
        if len(parts) != 4:
            return None
        x1, y1, x2, y2 = parts
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _write_yolo_label(self, path, box, index, w, h):
        """Escribe una linea YOLO normalizada. Sin caja escribe un archivo vacio,
        que es la forma canonica de declarar "en esta imagen no hay objeto"."""
        if box is None or index is None or w <= 0 or h <= 0:
            open(path, "w", encoding="utf-8").close()
            return False
        x1, y1, x2, y2 = box
        x1 = max(0.0, min(float(x1), w))
        y1 = max(0.0, min(float(y1), h))
        x2 = max(0.0, min(float(x2), w))
        y2 = max(0.0, min(float(y2), h))
        if x2 <= x1 or y2 <= y1:
            open(path, "w", encoding="utf-8").close()
            return False
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        return True

    def export(self, min_confidence=0.0):
        """Exporta todos los falsos positivos etiquetados.

        Devuelve {"samples": n, "output_dir": ruta, "skipped": m,
                  "with_box": k}.

        Dos tipos de muestra, y la distincion importa para no ensenar el error:

        - Con `real_class` declarada por el operador y caja conocida: se escribe
          una etiqueta YOLO positiva con la clase REAL. Ejemplo: el modelo dijo
          "humano" sobre una vaca; el operador declara `bovino` y el modelo
          aprende a corregir la clase.
        - Sin clase declarada: la region se exporta como negativo. Para YOLO, un
          archivo de etiqueta vacio significa "no hay objeto aqui", que es
          justamente la supervision que necesita un falso positivo. Antes se
          escribia un comentario con la clase PREDICHA, lo que validaba el error
          en lugar de corregirlo.
        """
        events = self.db.get_false_positive_events(limit=2000)
        events = [e for e in events if (e[3] or 0) >= min_confidence]

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = os.path.join(self.output_dir, stamp)
        images_dir = os.path.join(target, "images")
        labels_dir = os.path.join(target, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        clips_dir = os.path.join(target, "clips")
        if self.copy_clips:
            os.makedirs(clips_dir, exist_ok=True)

        exported = 0
        skipped = 0
        with_box = 0
        clips = 0
        manifest = []

        for row in events:
            (event_id, timestamp, class_name, confidence,
             snapshot_path, bbox, real_class, clip_path) = row

            if not snapshot_path or not os.path.exists(snapshot_path):
                skipped += 1
                continue

            base = f"fp_{event_id}_{int(float(confidence) * 100)}"
            img_ext = os.path.splitext(snapshot_path)[1] or ".jpg"
            img_name = base + img_ext
            dest_image = os.path.join(images_dir, img_name)

            if self.copy_snapshots:
                try:
                    shutil.copy2(snapshot_path, dest_image)
                except OSError as exc:
                    logger.error(f"No se pudo copiar {snapshot_path}: {exc}")
                    skipped += 1
                    continue

            # Dimensiones reales: sin ellas la normalizacion de la caja seria
            # incorrecta y la muestra entrenaria con un recuadro desplazado.
            width = height = 0
            try:
                import cv2

                probe = cv2.imread(snapshot_path)
                if probe is not None:
                    height, width = probe.shape[:2]
            except Exception:  # noqa: BLE001
                pass

            index = self._class_index(real_class)
            box = self._parse_bbox(bbox)
            positive = index is not None and box is not None and width > 0 and height > 0
            if positive:
                self._write_yolo_label(
                    os.path.join(labels_dir, base + ".txt"),
                    box, index, width, height,
                )
                with_box += 1
            else:
                # Sin clase declarada o sin caja: negativo explicito. YOLO no admite
                # "region negativa" ni indice de clase -1: la unica forma de
                # ensenar "aqui no hay nada" es un archivo de etiqueta
                # vacio para la imagen completa. La region queda registrada en
                # el manifiesto para que un humano la dibuje si hace falta.
                self._write_yolo_label(os.path.join(labels_dir, base + ".txt"),
                                       None, None, width, height)

            # El clip probatorio da el contexto que un recorte no tiene: si la
            # "persona" era una sombra de una rama moviendose, se ve en el clip y
            # no en un fotograma. Sin el, el reentrenamiento se queda con una
            # imagen ambigua que nadie reviso.
            clip_name = None
            if self.copy_clips and clip_path and os.path.exists(clip_path):
                clip_ext = os.path.splitext(clip_path)[1] or ".mp4"
                clip_name = base + clip_ext
                try:
                    shutil.copy2(clip_path, os.path.join(clips_dir, clip_name))
                    clips += 1
                except OSError as exc:
                    # Un clip que no se copia NO invalida la muestra: la imagen
                    # y su etiqueta siguen siendo validas para entrenar.
                    logger.warning(
                        f"[RE18] No se pudo copiar el clip {clip_path}: {exc}"
                    )
                    clip_name = None

            manifest.append({
                "event_id": event_id,
                "timestamp": timestamp,
                "image": img_name,
                "clip": clip_name,
                "confidence": confidence,
                "clase_predicha": class_name,
                "clase_real": real_class or BACKGROUND,
                "bbox": bbox,
                "etiqueta": "positiva" if positive else "negativa",
            })
            exported += 1

        # Dataset vacio no sirve de nada: se limpia para no dejar carpetas
        # huerfanas que confundan al operador.
        if exported == 0:
            shutil.rmtree(target, ignore_errors=True)
        else:
            with open(os.path.join(target, "manifest.json"), "w",
                      encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

        self.db.record_false_positive_export(
            target if exported else "sin muestras",
            exported,
            note=f"umbral={min_confidence}",
        )

        logger.info(
            f"[RE18] Exportacion {stamp}: {exported} muestras "
            f"({with_box} con caja y clase real, {clips} con clip), "
            f"{skipped} omitidas"
        )

        # Instrucciones para completar el ciclo de reentrenamiento.
        readme = os.path.join(target, "COMO_REENTRENAR.txt")
        if exported > 0:
            with open(readme, "w", encoding="utf-8") as f:
                f.write(
                    "RE18 - Dataset de falsos positivos\n"
                    "==================================\n\n"
                    f"{exported} imagen(es) exportadas, {with_box} con etiqueta\n"
                    "positiva (caja + clase real declarada).\n\n"
                    "Que significa cada tipo de muestra:\n\n"
                    "  - Etiqueta positiva (una linea 'clase cx cy w h'): el\n"
                    "    operador declaro que clase real habia en esa region. El\n"
                    "    modelo aprende a corregir la clase que erro.\n"
                    "  - Etiqueta vacia: la region se declara NEGATIVA. Es la\n"
                    "    supervision correcta para un falso positivo: ahi no hay\n"
                    "    nada que detectar.\n\n"
                    "manifest.json lista, por evento, la clase predicha, la clase\n"
                    "real declarada y la caja, para revisar o completar a mano.\n\n"
                    "Para reentrenar el modelo:\n\n"
                    "  1. Revisa manifest.json y corrige a mano las muestras\n"
                    "     negativas que en realidad tengan un objeto.\n"
                    "  2. Copia images/ a data/custom_yolo/images/train/ y\n"
                    "     labels/ a data/custom_yolo/labels/train/, junto con el\n"
                    "     dataset original.\n"
                    "  3. Ejecuta python src/03_prepare_yolo_dataset.py para\n"
                    "     reconsolidar los splits.\n"
                    "  4. Reentrena:  python src/04_train.py\n\n"
                    "Nunca etiquetes con la clase que PREDJO el modelo: eso\n"
                    "validaria el error en vez de corregirlo.\n"
                )

        return {"samples": exported, "skipped": skipped, "with_box": with_box,
                "clips": clips, "output_dir": target}

    def stats(self):
        """Resumen para la GUI.

        Se reportan las DOS lecturas, no solo una. Agrupar por la clase PREDICHA
        es lo que el operador ya conoce (son los errores que el propio modelo
        cometio) y no le dice nada de que corregir. La clase REAL declarada es
        la que decide como se entrenara, y las muestras sin clase real declarada
        son las que van a enseñar al modelo a descartar el fondo.
        """
        rows = self.db.get_false_positive_exports(10)
        by_predicted = {}
        by_real = {}
        pending_real = 0
        for event in self.db.get_false_positive_events(limit=2000):
            predicted = event[2] or "?"
            real = event[6]
            by_predicted[predicted] = by_predicted.get(predicted, 0) + 1
            if real:
                by_real[real] = by_real.get(real, 0) + 1
            else:
                # Sin clase real declarada la muestra es un fondo: eso fue
                # justo el error, aunque el modelo acertara con una clase real.
                by_real[BACKGROUND] = by_real.get(BACKGROUND, 0) + 1
                pending_real += 1
        return {
            "total": self.db.count_false_positives(),
            "by_class": by_real,
            "by_predicted": by_predicted,
            "background": by_real.get(BACKGROUND, 0),
            "pending_real_class": pending_real,
            "exports": [{"path": r[1], "samples": r[2], "at": r[3]} for r in rows],
        }
