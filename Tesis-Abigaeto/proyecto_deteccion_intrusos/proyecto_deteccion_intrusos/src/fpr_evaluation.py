"""
Objetivo 4 del documento: medir la tasa de falsos positivos (FPR) y el
retorno de la inversion (ROI).

El documento fija como criterio de aceptacion un FPR menor al 5%. Ese numero
no se puede estimar a ojo: hay que correr el modelo sobre escenas que NO
contienen intrusiones reales y contar cuantas veces el sistema dispara.

Metodo (pesado, similar al de los detectores de objetos pero orientado a
alarmas):

  - Se recorren N clips de video sin intrusiones (escenarios negativos).
  - Para cada deteccion humana dentro de la zona perimetral se cuenta un
    falso positivo.
  - Un clip con al menos un disparo se considera un clip con FPs.

  FPR por evento   = FP / (personas ground-truth)  -- requiere anotacion
  FPR por escena   = escenas_con_FP / escenas_totales

Este modulo ofrece ambas. Cuando solo hay clips sin anotacion (el caso normal
en terreno), reporta FPR por escena, que es la cifra defendible sin etiquetar
a mano, y lo dice explicitamente para no presentar un numero engañoso.
"""

import json
import os
from datetime import datetime

import cv2

try:
    from src.utils import get_logger, load_config
except ImportError:
    from utils import get_logger, load_config

logger = get_logger("fpr_evaluation")


class FPREvaluation:
    def __init__(self, config, db=None, model=None):
        self.config = config
        self.db = db
        self.model = model
        self.output_path = os.path.join(
            config.get("evaluation", {}).get("output_dir", "data/evaluation"),
            "fpr_report.json",
        )

    # ---------- Nucleo de evaluacion ----------

    def _load_model(self):
        if self.model is not None:
            return self.model
        weights = self.config["model"]["weights_path"]
        logger.info(f"[FPR] Cargando modelo {weights}")
        # `load_yolo` centraliza la allowlist de PyTorch >= 2.6; llamar a
        # `YOLO(...)` directo aqui fallaba con UnpicklingError.
        from src.model_loader import load_yolo
        self.model = load_yolo(weights, device=self.config["model"].get("device"))
        return self.model

    def _zone_polygon(self):
        """Poligono de RE01 como array numpy int32, o None si no hay zona.

        `cv2.pointPolygonTest` exige un numpy array: entregarle la lista de
        tuplas del config lanzaba Bad argument en cuanto aparecia una persona.
        Sin poligono definido se vigila el frame completo (mismo criterio que
        el pipeline y la GUI).
        """
        import numpy as np

        points = self.config.get("zone", {}).get("polygon") or []
        if len(points) < 3:
            return None
        return np.array([tuple(p) for p in points], dtype="int32")

    @staticmethod
    def _in_zone(point, polygon):
        if polygon is None:
            return True
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def evaluate_video(self, video_path, max_frames=None, annotate_to=None):
        """Recorre un clip y cuenta las alarmas sobre una escena sin intrusiones.

        Devuelve un diccionario con los conteos y las marcas de tiempo de cada
        disparo, para que el operador pueda revisar esos instantes en el video.
        """
        model = self._load_model()
        polygon = self._zone_polygon()
        conf = self.config["model"]["conf_threshold"]
        names = self.config["classes"]["names"]
        human = self.config["classes"]["human_class"]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        limit = max_frames or total_frames or 10 ** 9

        false_positives = []
        frames_with_detection = 0
        processed = 0
        writer = None
        if annotate_to:
            os.makedirs(os.path.dirname(annotate_to) or ".", exist_ok=True)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                annotate_to, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )

        while processed < limit:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(frame, conf=conf, verbose=False)
            hit = False
            annotated = frame.copy()

            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = names[cls_id] if cls_id < len(names) else str(cls_id)
                    if class_name != human:
                        # RE07/RE10: ganado NO cuenta como falso positivo.
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    if not self._in_zone((cx, cy), polygon):
                        continue
                    hit = True
                    score = float(box.conf[0])
                    if annotate_to:
                        cv2.rectangle(annotated, (int(x1), int(y1)),
                                      (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(annotated, f"FALSO POSITIVO {score:.2f}",
                                    (int(x1), max(int(y1) - 6, 14)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if hit:
                frames_with_detection += 1
                false_positives.append({
                    "frame": processed,
                    "timestamp_s": round(processed / fps, 2),
                    "confidence": score,
                })

            if writer is not None:
                writer.write(annotated)

            processed += 1

        cap.release()
        if writer is not None:
            writer.release()

        return {
            "video": os.path.basename(video_path),
            "path": video_path,
            "fps": round(fps, 3),
            "frames_processed": processed,
            "duration_s": round(processed / fps, 2),
            "human_detections_in_zone": len(false_positives),
            "scenes_with_fp": 1 if false_positives else 0,
            "false_positives": false_positives,
        }

    def evaluate_dataset(self, video_paths, progress=None):
        """Agrega los resultados de varios clips en un veredicto.

        Criterio de aceptacion del documento: FPR < 5%.

        `meets_target` solo puede ser True si los clips son ground truth
        negativo VERIFICADO. Sin esa declaracion el sistema no puede afirmar
        que un disparo fue un falso positivo (puede ser una intrusion real no
        anotada), asi que el veredicto queda como NO VERIFICABLE.
        """
        video_paths = list(video_paths)
        evaluation_cfg = self.config.get("evaluation", {})
        verified = bool(evaluation_cfg.get("negative_clips_verified", False))
        min_minutes = float(evaluation_cfg.get("min_negative_minutes", 10.0))

        results = []
        for i, path in enumerate(video_paths):
            if progress:
                progress(i + 1, len(video_paths), os.path.basename(path))
            try:
                results.append(self.evaluate_video(path))
            except FileNotFoundError as exc:
                logger.error(f"[FPR] {exc}")
                results.append({"video": os.path.basename(path), "error": str(exc),
                                "frames_processed": 0, "duration_s": 0.0,
                                "human_detections_in_zone": 0,
                                "scenes_with_fp": 0, "false_positives": []})

        total_scenes = len(video_paths)
        scenes_with_fp = sum(r.get("scenes_with_fp", 0) for r in results)
        total_fp = sum(r.get("human_detections_in_zone", 0) for r in results)
        total_frames = sum(r.get("frames_processed", 0) for r in results)
        # La duracion la reporta cada clip con SU fps. Antes se dividia por un
        # 10 fps fijo, lo que inflaba o achataba la tasa segun la camara.
        total_seconds = sum(float(r.get("duration_s", 0.0)) for r in results)

        fpr_scene = (scenes_with_fp / total_scenes) if total_scenes else 0.0
        # Tasa por frame: detecciones por minuto de video, comparable entre
        # clips de distinta duracion.
        minutes = total_seconds / 60.0
        per_minute = (total_fp / minutes) if minutes > 0 else 0.0

        data_sufficient = verified and minutes >= min_minutes
        if total_scenes == 0:
            meets_target = False
            verdict_note = "No hay clips negativos configurados: no se puede medir el FPR."
        elif not verified:
            meets_target = False
            verdict_note = (
                "Los clips NO estan declarados como ground truth negativo "
                "(evaluation.negative_clips_verified = false). Un disparo aqui "
                "puede ser una intrusion real, no un falso positivo, asi que el "
                "FPR no es certificable. Graba clips en la finca sin intrusiones "
                "y ponlos en evaluation.negative_clips con la marca en true."
            )
        elif not data_sufficient:
            meets_target = False
            verdict_note = (
                f"Solo {minutes:.1f} min de video negativo; el analisis pide al "
                f"menos {min_minutes:.0f} min para que la cifra sea estable."
            )
        else:
            meets_target = fpr_scene < 0.05
            verdict_note = "Cifra calculada sobre ground truth negativo verificado."

        report = {
            "generated_at": datetime.now().isoformat(),
            "model": self.config["model"]["weights_path"],
            "confidence_threshold": self.config["model"]["conf_threshold"],
            "scenes_total": total_scenes,
            "scenes_with_false_positives": scenes_with_fp,
            "total_false_positives": total_fp,
            "frames_processed": total_frames,
            "video_seconds": round(total_seconds, 2),
            "fpr_per_scene": round(fpr_scene, 4),
            "fpr_percentage": round(fpr_scene * 100, 2),
            "false_positives_per_minute": round(per_minute, 3),
            "target": "< 5%",
            "meets_target": meets_target,
            "ground_truth_verified": verified,
            "data_sufficient": data_sufficient,
            "verdict_note": verdict_note,
            "metric_note": (
                "FPR por escena: clips sin intrusion real en los que el sistema "
                "disparo al menos una vez. Es la cifra defendible sin anotacion "
                "manual; el FPR por objeto exigiria ground truth etiquetado."
            ),
            "videos": results,
        }

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # El veredicto se calcula en un solo sitio (`verdict_of`) y lo usan
        # tanto este log como la vista de Evaluation.
        #
        # Ojo con el caso "verificado y suficiente pero por encima del
        # objetivo": es NO CUMPLE, no NO VERIFICABLE. Aqui si se midio, y la
        # medicion sale mal. Confundir "no se pudo medir" con "se midio y no
        # cumplio" permitiria presentar como pendiente un sistema que ya se
        # sabe que no cumple.
        verdict = self.verdict_of(report)

        logger.info(
            f"[FPR] {verdict}: {report['fpr_percentage']}% "
            f"({scenes_with_fp}/{total_scenes} escenas con disparo, "
            f"{minutes:.1f} min, objetivo {report['target']})"
        )
        if not report["meets_target"]:
            logger.warning(f"[FPR] {verdict}: {verdict_note}")
        return report

    @staticmethod
    def verdict_of(report):
        """Veredicto unico del reporte.

        Vive en un solo metodo a proposito: el log del evaluador y la vista de
        Evaluation calculaban el veredicto por separado y ya se habian
        desincronizado. Si divergen, la GUI puede mostrar una cosa y el informe
        guardado otra, que es justo lo que no puede pasar en una tesis.
        """
        if report.get("meets_target"):
            return "CUMPLE"
        if not report.get("ground_truth_verified"):
            return "NO VERIFICABLE"
        if not report.get("data_sufficient"):
            return "DATOS INSUFICIENTES"
        return "NO CUMPLE"

    @staticmethod
    def format_report(report):
        """Resumen legible para la GUI."""
        return {
            "fpr": f"{report.get('fpr_percentage', 0)}%",
            "verdict": FPREvaluation.verdict_of(report),
            "scenes": f"{report.get('scenes_with_false_positives', 0)}/"
                       f"{report.get('scenes_total', 0)}",
            "total_fp": report.get("total_false_positives", 0),
            "per_minute": report.get("false_positives_per_minute", 0),
            "note": report.get("verdict_note", ""),
        }


def default_negative_clips(config):
    """Clips sugeridos para la evaluacion: escenas sin intrusiones.

    El documento pide medir sobre el conjunto de validacion. Se priorizan los
    clips que ya existen en el repositorio y se documenta la LIMITACION: el
    set de validacion del proyecto contiene personas, asi que un disparo alli
    puede ser una deteccion correcta, no un falso positivo. Para una medicion
    honesta hay que usar clips grabados en la finca SIN intrusiones.
    """
    candidates = []
    for folder in ("data/sample_video.mp4", "data/videos"):
        if os.path.isfile(folder):
            candidates.append(folder)
        elif os.path.isdir(folder):
            for name in sorted(os.listdir(folder)):
                if name.lower().endswith((".mp4", ".avi", ".mkv")):
                    candidates.append(os.path.join(folder, name))
    return candidates
