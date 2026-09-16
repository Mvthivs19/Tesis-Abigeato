"""
RE01: Permite al usuario (rol Administrador) trazar un polígono virtual
directamente sobre la interfaz visual capturada por la cámara simulada,
para delimitar la región exacta sometida a inferencia continua.

Uso:
  python src/05_define_zone.py

Instrucciones en pantalla:
  - Click izquierdo: agregar un punto al polígono
  - Tecla 's': guardar el polígono en configs/config.yaml y salir
  - Tecla 'r': reiniciar el polígono (borrar puntos)
  - Tecla 'q': salir sin guardar
"""

import cv2
from utils import load_config, save_config

points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def main():
    config = load_config()
    source = config["camera"]["source"]

    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print(f"No se pudo leer un frame desde: {source}")
        print("Verifica que 'camera.source' en configs/config.yaml apunte a un video válido.")
        return

    window_name = "Definir zona de vigilancia (RE01) - click para agregar puntos"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Click izquierdo: agregar punto | 's': guardar | 'r': reiniciar | 'q': salir")

    while True:
        display = frame.copy()

        for p in points:
            cv2.circle(display, tuple(p), 4, (0, 255, 0), -1)
        if len(points) > 1:
            cv2.polylines(display, [cv2_points(points)], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("s"):
            if len(points) < 3:
                print("Necesitas al menos 3 puntos para formar un polígono.")
                continue
            config["zone"]["polygon"] = points
            save_config(config)
            print(f"Zona guardada con {len(points)} puntos en configs/config.yaml")
            break
        elif key == ord("r"):
            points.clear()
        elif key == ord("q"):
            print("Salida sin guardar.")
            break

    cv2.destroyAllWindows()


def cv2_points(pts):
    import numpy as np
    return np.array(pts, dtype="int32")


if __name__ == "__main__":
    main()
