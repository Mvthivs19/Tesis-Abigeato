"""
Verificacion de la maquinaria FPR con el modelo real sobre clip NEGATIVO
SINTETICO.

QUE ESTO ES Y QUE NO ES
-----------------------
Es una prueba de que la cadena funciona de extremo a extremo: decodifica el
video, infiere con los pesos reales, aplica la zona, cuenta escenas con
disparo y escribe el informe. El clip se DECLARA ground truth negativo a
proposito, para poder ver que la maquina llega a un veredicto en vez de
devolver "NO VERIFICABLE".

NO es una medicion del FPR del sistema. El clip es sintetico, no footage de la
finca, asi que la tasa que salga aqui no es el FPR del sistema y no debe
citarse como tal. El FPR real sigue sin certificar: exige clips grabados en la
finca sin intrusiones.

El umbral `min_negative_minutes` se baja a 0.01 solo para que la verificacion
no tarde 10 minutos de video. El resto de la logica de veredicto es la real.
"""

import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.fpr_evaluation import FPREvaluation
from src.model_loader import load_yolo
from src.utils import load_config

SIDE = 320
FPS = 10
SECONDS = 12


def synthetic_negative(path):
    """Escena sin personas: suelo con textura, una sombra que cruza y un
    poste. Es el tipo de escena que produce falsos positivos (contraste y
    sombras), que es justo lo que el FPR pretende medir."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (SIDE, SIDE))
    for i in range(FPS * SECONDS):
        img = np.full((SIDE, SIDE, 3), (58, 92, 64), np.uint8)
        # textura de suelo
        for y in range(0, SIDE, 17):
            cv2.line(img, (0, y), (SIDE, y), (52, 84, 58), 2)
        # poste fijo
        cv2.rectangle(img, (232, 96), (250, 250), (110, 108, 104), -1)
        # sombra que cruza (posible confusor)
        offset = int(i * 6) % (SIDE + 120) - 60
        cv2.ellipse(img, (offset, 210), (46, 20), 0, 0, 360, (40, 40, 42), -1)
        # manchas brighter
        cv2.circle(img, (90, 120), 16, (128, 150, 120), -1)
        writer.write(img)
    writer.release()
    return path


def escenario_sin_certificar(config, model, video):
    """Segundo escenario: un clip con personas reales, SIN declararlo ground
    truth negativo.

    Sirve para lo contrario de lo que parece: comprueba que el contador de
    detecciones funciona con inferencia real y que, aun assim, el veredicto se
    niega a certificar. Declarar como "negativo" un clip con gente seria
    precisamente el error que este sistema pretende evitar.
    """
    config = dict(config)
    evaluation = dict(config["evaluation"])
    evaluation["negative_clips"] = [video]
    evaluation["negative_clips_verified"] = False
    config["evaluation"] = evaluation

    evaluator = FPREvaluation(config, model=model)
    report = evaluator.evaluate_dataset([video])
    resumen = FPREvaluation.format_report(report)
    return report, resumen


def diagnostico_zona(video, config):
    """Cobertura de la zona configurada sobre el encuadre real del clip.

    Importa porque una zona pequena no es solo "menos detecciones": es
    vigilancia muda. Si la zona no cubre la parte del encuadre por donde pasa
    la gente, el sistema no dispara y parece que todo va bien.
    """
    poly = config.get("zone", {}).get("polygon") or []
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    if not ok or not poly:
        return None
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 1)
    coverage = float(mask.sum()) / float(h * w)
    pts = np.array(poly, dtype=np.int32)
    return {
        "resolucion": f"{w}x{h}",
        "cobertura_pct": round(coverage * 100, 1),
        "x": (int(pts[:, 0].min()), int(pts[:, 0].max())),
        "y": (int(pts[:, 1].min()), int(pts[:, 1].max())),
    }


def main():
    tmp = tempfile.mkdtemp(prefix="fpr_maquina_")
    clip = synthetic_negative(os.path.join(tmp, "negativo_sintetico.mp4"))

    config = load_config()
    config["evaluation"]["negative_clips"] = [clip]
    config["evaluation"]["negative_clips_verified"] = True
    config["evaluation"]["min_negative_minutes"] = 0.01
    config["evaluation"]["output_dir"] = os.path.join(tmp, "eval")

    print("=" * 70)
    print("VERIFICACION DE MAQUINARIA FPR (clip sintetico, NO es el FPR real)")
    print("=" * 70)
    print(f"clip        : {os.path.basename(clip)}")
    print(f"duracion    : {SECONDS} s a {FPS} fps ({SECONDS / 60:.2f} min)")
    print(f"pesos       : {config['model']['weights_path']}")
    print(f"conf        : {config['model']['conf_threshold']}")
    print(f"verificado  : {config['evaluation']['negative_clips_verified']}")
    print(f"min minutos : {config['evaluation']['min_negative_minutes']}")
    print("-" * 70)

    model = load_yolo(config["model"]["weights_path"],
                      device=config["model"].get("device"))
    evaluator = FPREvaluation(config, model=model)
    report = evaluator.evaluate_dataset([clip])

    print(f"frames procesados      : {report['frames_processed']}")
    print(f"segundos de video      : {report['video_seconds']}")
    print(f"escenas con disparo    : "
          f"{report['scenes_with_false_positives']}/{report['scenes_total']}")
    print(f"disparos totales       : {report['total_false_positives']}")
    print(f"fpr por escena         : {report['fpr_percentage']}% "
          f"(objetivo {report['target']})")
    print(f"por minuto             : {report['false_positives_per_minute']}")
    print(f"ground truth verificado: {report['ground_truth_verified']}")
    print(f"datos suficientes       : {report['data_sufficient']}")
    print(f"cumple el objetivo     : {report['meets_target']}")

    resumen = FPREvaluation.format_report(report)
    print(f"VEREDICTO               : {resumen['verdict']}")
    print(f"nota                    : {resumen['note']}")
    print("-" * 70)
    print(f"informe escrito en      : {evaluator.output_path}")
    print(f"existe en disco         : "
          f"{os.path.exists(evaluator.output_path)}")

    print()
    print("COMO LEER ESTO: la maquina recorre el clip con el modelo real, "
          "agrega\nla tasa y decide veredicto. La cifra obtenida depende de un "
          "clip sintetico\ny NO es el FPR del sistema; el objetivo <5% sigue "
          "sin certificar hasta\ndisponer de clips de la finca declarados "
          "como ground truth negativo.")

    # ---- Escenario 2: clip con personas, sin declarar ground truth ----
    muestra = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "sample_video.mp4")
    if os.path.exists(muestra):
        print()
        print("=" * 70)
        print("ESCENARIO 2: clip con personas, NO declarado ground truth")
        print("=" * 70)
        report2, resumen2 = escenario_sin_certificar(config, model, muestra)
        print(f"clip                  : {os.path.basename(muestra)}")
        print(f"frames procesados     : {report2['frames_processed']}")
        print(f"escenas con disparo   : "
              f"{report2['scenes_with_false_positives']}/"
              f"{report2['scenes_total']}")
        print(f"disparos totales      : {report2['total_false_positives']}")
        print(f"ground truth verificado: {report2['ground_truth_verified']}")
        print(f"cumple el objetivo    : {report2['meets_target']}")
        print(f"VEREDICTO              : {resumen2['verdict']}")
        print("-" * 70)
        print("Aunque el contador registre disparos, sin declaracion de ground\n"
              "truth el veredicto NO puede ser CUMPLE: un disparo aqui puede\n"
              "ser una deteccion correcta de una persona real.")
        if resumen2["verdict"] == "CUMPLE":
            print("ATENCION: el veredicto es CUMPLE sin ground truth verificado. "
                  "Es un fallo.")

        zona = diagnostico_zona(muestra, config)
        if zona:
            print("-" * 70)
            print("DIAGNOSTICO DE ZONA (RE01)")
            print(f"  encuadre del clip   : {zona['resolucion']}")
            print(f"  zona configurada    : x{zona['x']} y{zona['y']}")
            print(f"  cobertura           : {zona['cobertura_pct']}% del encuadre")
            if zona["cobertura_pct"] < 60:
                print()
                print("  HALLAZGO: la zona por defecto cubre menos del 60% del "
                      "encuadre.\n  En este clip hay personas detectadas por "
                      "debajo del borde inferior\n  de la zona, asi que NO "
                  "disparan. Una zona mal trazada no es solo\n  'menos "
                  "detecciones': es vigilancia muda, y el sistema parece\n  "
                  "correcto mientras no vigila la zona por donde pasa la "
                  "gente.\n  ACCION REQUERIDA antes de desplegar: redibujar la "
                  "zona con la\n  camara real (RE01) y verificar la cobertura.")
    else:
        print()
        print(f"(escenario 2 omitido: no existe {muestra})")

    return report


if __name__ == "__main__":
    main()
