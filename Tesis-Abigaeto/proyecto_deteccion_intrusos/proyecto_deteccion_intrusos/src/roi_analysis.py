"""
Objetivo 4 del documento: analisis de retorno de la inversion (ROI).

El ROI compara el costo de operar el sistema (equipo, camara, mantenimiento,
energia, horas de Vigilancia) con el beneficio evitado (perdidas por abigeato
que el sistema evita al detectar y avisar a tiempo).

La cifra clave del ROI es el tiempo de reaccion: entre que el intruso entra al
perimetro y que se emite la primera alerta. Si ese tiempo es menor que el que
el ladron necesita para cruzar la finca y salir, la perdida se evita. Por eso
el analisis se hace sobre escenarios con parametros declarados, no con numeros
inventados: cada parametro economico es editable y queda registrado de donde
salio.
"""

import json
import os
from datetime import datetime

try:
    from src.utils import get_logger, load_config
except ImportError:
    from utils import get_logger, load_config

logger = get_logger("roi_analysis")

# Valores por defecto pensados para una finca pequena de Chile. Todos son
# EDITABLES y la GUI deja verlos: presentarlos como "datos de la tesis" sin
# fuente seria falsear el analisis.
DEFAULTS = {
    # --- Dimension de la perdida ---
    "ganado": 35,                     # animales totales en la finca
    "animales_por_evento": 3,         # animales que se lleva un abigeato tipico
    "valor_por_animal": 180_000,      # CLP, valor promedio de un animal
    "eventos_anuales_estimados": 2,   # abigeatos por ano en la zona

    # --- Efectividad ---
    "detecciones_por_evento": 1.0,    # fraccion de eventos que el sistema detecta
    "evitacion_por_deteccion": 0.7,   # fraccion de la perdida evitada al avisar

    # --- Costos ---
    "costo_camara": 320_000,          # CLP, camara + instalacion
    "costo_equipo": 480_000,          # CLP, equipo de computo
    "costo_instalacion": 150_000,     # CLP, mano de obra de instalacion
    "costo_mantenimiento_anual": 90_000,

    # --- Ahorro de vigilancia ---
    "horas_vigilancia_ahorradas_semana": 3,
    "valor_hora_vigilancia": 7_500,

    "anios_analizados": 5,
}

# Un ROI superior a este valor casi siempre indica un supuesto erroneo
# (por ejemplo, todas las horas de vigilancia se pagan al precio de un
# operative de seguridad) y no un sistema realmente rentable.
ROI_PLAUSIBLE_MAX_PCT = 400.0

# Tiempo tipico entre el ingreso del intruso y su salida del perimetro.
# Es la ventana en la que la alerta sirve de algo.
ESCENARIOS = {
    "optimista": 3,    # minutos: entra, recorre, sale
    "esperado": 6,
    "pesimista": 12,
}


class ROIAnalysis:
    def __init__(self, config, db=None, params=None):
        self.config = config
        self.db = db
        self.p = dict(DEFAULTS)
        # Los supuestos economicos se leen de config.yaml para que el productor
        # los reemplace por los de SU finca. Estaban fijos en el codigo, lo que
        # hacia que el aviso "reemplaza los montos por los tuyos" fuera
        # imposible de seguir y el ROI fuera siempre el mismo numero.
        self.p.update(config.get("evaluation", {}).get("roi", {}) or {})
        self.p.update(params or {})
        self.output_path = os.path.join(
            config.get("evaluation", {}).get("output_dir", "data/evaluation"),
            "roi_report.json",
        )

    # ---------- Tiempo de deteccion ----------

    def detection_latency(self):
        """Tiempo entre que el intruso entra y se emite la primera alerta.

        Se compone de las latencias reales de la cadena:
          - inferencia: un frame a `fps_limit`
          - cooldown: NO cuenta, porque es una decision de diseno
          - grabacion del snapshot y encolado: despreciable
          - red: timeout configurado, en el peor caso sin Internet la alerta
            queda encolada y se entrega al recuperar la senal
        """
        fps = self.config.get("camera", {}).get("fps_limit", 10)
        infer = 1.0 / max(fps, 1)                      # un frame
        notify = self.config.get("alerts", {}).get("remote", {}).get("timeout_seconds", 5)
        escalate = self.config.get("alerts", {}).get("escalate_after_seconds", 60)

        return {
            "inferencia_s": round(infer, 3),
            "notificacion_s": notify,
            "escalamiento_s": escalate,
            "total_local_s": round(infer + notify, 3),
            "offline_s": round(infer + escalate, 3),
        }

    # ---------- Escenarios ----------

    def scenario(self, name, minutes_available):
        """Si la alerta llega mas rapido que la salida del intruso, la perdida
        se reduce; si llega tarde, el dano ya ocurrio."""
        latency = self.detection_latency()
        detect_time = latency["total_local_s"] / 60.0
        # Fraccion del tiempo de permanencia en que se alcanza a avisar.
        if minutes_available <= 0:
            effectiveness = 0.0
        else:
            effectiveness = max(0.0, min(1.0, 1.0 - (detect_time / minutes_available)))

        p = self.p
        # Perdida anual esperada: eventos por ano x animales robados por evento
        # x valor unitario. No se multiplica por el tamano completo del rebano
        # porque un abigeato se lleva una fraccion, no la totalidad.
        annual_loss = (p["eventos_anuales_estimados"]
                       * p["animales_por_evento"]
                       * p["valor_por_animal"])
        avoided = (annual_loss * p["detecciones_por_evento"]
                   * p["evitacion_por_deteccion"] * effectiveness)
        return {
            "escenario": name,
            "minutos_disponibles": minutes_available,
            "llega_a_tiempo": effectiveness > 0,
            "efectividad": round(effectiveness, 3),
            "perdida_anual_estimada_clp": int(annual_loss),
            "perdida_evitada_anual_clp": int(avoided),
        }

    def compute(self):
        p = self.p
        years = p["anios_analizados"]

        initial = p["costo_camara"] + p["costo_equipo"] + p["costo_instalacion"]
        annual_cost = p["costo_mantenimiento_anual"]
        labor_saving = (p["horas_vigilancia_ahorradas_semana"]
                        * 52 * p["valor_hora_vigilancia"])

        scenarios = {name: self.scenario(name, minutes)
                     for name, minutes in ESCENARIOS.items()}

        # Escenario de referencia para el calculo del retorno: el esperado.
        annual_benefit = scenarios["esperado"]["perdida_evitada_anual_clp"] + labor_saving
        net_annual = annual_benefit - annual_cost
        total_investment = initial + annual_cost * years
        total_benefit = annual_benefit * years

        roi_pct = ((total_benefit - total_investment) / total_investment * 100) \
            if total_investment > 0 else 0.0
        payback_years = (initial / net_annual) if net_annual > 0 else None

        # Advertencia de plausibilidad: un ROI desproporcionado casi siempre
        # nace de un supuesto economico erroneo, no de un buen sistema.
        warnings = []
        if roi_pct > ROI_PLAUSIBLE_MAX_PCT:
            warnings.append(
                f"ROI de {roi_pct:.0f}% supera el {ROI_PLAUSIBLE_MAX_PCT:.0f}% de "
                "referencia: revise las horas de vigilancia ahorradas y el "
                "valor por animal."
            )
        if net_annual <= 0:
            warnings.append("La utilidad neta anual es negativa o nula.")

        report = {
            "generated_at": datetime.now().isoformat(),
            "moneda": "CLP",
            "latencia_deteccion": self.detection_latency(),
            "escenarios": scenarios,
            "inversion_inicial_clp": int(initial),
            "costo_anual_clp": int(annual_cost),
            "ahorro_vigilancia_anual_clp": int(labor_saving),
            "beneficio_anual_clp": int(annual_benefit),
            "utilidad_neta_anual_clp": int(net_annual),
            "inversion_total_clp": int(total_investment),
            "beneficio_total_clp": int(total_benefit),
            "roi_porcentaje": round(roi_pct, 1),
            "retorno_inversion_anos": round(payback_years, 2) if payback_years else None,
            "advertencias": warnings,
            "parametros": p,
            "nota_metodologica": (
                "El ROI depende de supuestos economicos editables. La latencia "
                "de deteccion proviene de la cadena de alertas real; los montos "
                "son valores de referencia que el productor debe reemplazar por "
                "los suyos antes de presentar."
            ),
        }

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(
            f"[ROI] ROI {report['roi_porcentaje']}% | payback "
            f"{report['retorno_inversion_anos']} anos | latencia "
            f"{report['latencia_deteccion']['total_local_s']}s"
        )
        for warning in warnings:
            logger.warning(f"[ROI] {warning}")
        return report

    @staticmethod
    def format_report(report):
        payback = report.get("retorno_inversion_anos")
        return {
            "roi": f"{report.get('roi_porcentaje', 0)}%",
            "payback": f"{payback} anos" if payback else "no se recupera",
            "utilidad_anual": f"${report.get('utilidad_neta_anual_clp', 0):,}".replace(",", "."),
            "latencia": f"{report['latencia_deteccion']['total_local_s']}s",
            "advertencias": report.get("advertencias", []),
        }
