"""
RE17: Al confirmarse una intrusión, el sistema debe almacenar un clip de
video probatorio, incluyendo unos segundos ANTES del evento (gracias a un
buffer circular en memoria) y unos segundos DESPUÉS.

Uso típico dentro del loop principal:
  recorder.push_frame(frame)          # se llama en CADA frame, siempre
  ...
  if evento_detectado:
      clip_path = recorder.trigger(event_id)   # empieza a guardar el clip
  ...
  recorder.update()                    # se llama en CADA frame, siempre
"""

import os
import time
from collections import deque

import cv2


class ClipRecorder:
    def __init__(self, output_dir="data/event_clips", pre_seconds=5, post_seconds=10, fps=10):
        self.output_dir = output_dir
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.fps = fps
        os.makedirs(output_dir, exist_ok=True)

        self.buffer = deque(maxlen=pre_seconds * fps)
        self.recording = False
        self.writer = None
        self.frames_left = 0
        self.current_path = None

    def push_frame(self, frame):
        self.buffer.append(frame.copy())
        if self.recording and self.writer is not None:
            self.writer.write(frame)
            self.frames_left -= 1
            if self.frames_left <= 0:
                self._close_writer()

    def trigger(self, event_id):
        """Inicia la grabación de un nuevo clip (si no hay una en curso)."""
        if self.recording:
            return self.current_path  # ya hay una grabación de evento en curso

        h, w = self.buffer[-1].shape[:2] if self.buffer else (480, 640)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"evento_{event_id}_{timestamp}.mp4"
        path = os.path.join(self.output_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))

        # Vuelca el buffer pre-evento (contexto de antes del incidente)
        for buffered_frame in self.buffer:
            self.writer.write(buffered_frame)

        self.recording = True
        self.frames_left = self.post_seconds * self.fps
        self.current_path = path
        return path

    def _close_writer(self):
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.recording = False
        self.current_path = None
