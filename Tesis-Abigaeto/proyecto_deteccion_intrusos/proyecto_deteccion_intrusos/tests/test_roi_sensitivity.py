"""
Sensibilidad del ROI: el numero debe ser CONSECUENCIA de los supuestos, no un
valor fijo con decimales inalterables.

Que se verifica
---------------
1. Que cada parametro econmico mueva el ROI en el sentido que corresponde. Un
   ROI que no responde a `eventos_anuales_estimados` o a `costo_camara` esta
   cableado, y su presentacion como "analisis" seria falsa.
2. Que las advertencias aparezcan cuando corresponde: ROI implausible,
   utilidad negativa, y latencia que llega tarde.
3. Que los escenarios de permanencia (3/6/12 min) den efectividades
   coherentes con la latencia de la cadena de alertas real.

Que NO se verifica
------------------
Que el ROI sea "correcto". Depende de montos que el productor debe reemplazar
por los de su finca. Aqui solo se comprueba que la aritmetica responde a lo que
se le dice que calcule.
"""

import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.roi_analysis import ROIAnalysis, ROI_PLAUSIBLE_MAX_PCT
from src.utils import load_config


class RoiSensitivityTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.base = load_config()
        self.base["evaluation"]["output_dir"] = os.path.join(self.tmp, "eval")

    def _roi(self, **params):
        """ROI con los parametros indicados encima de la configuracion real."""
        analysis = ROIAnalysis(self.base, params=params or None)
        return analysis.compute()["roi_porcentaje"]

    def test_every_economic_parameter_moves_the_roi(self):
        """Cada supuesto debe cambiar el resultado en la direccion prevista.

        Si alguno no cambia nada, el ROI no es un analisis sino una constante
        disfrazada de calculo."""
        base = self._roi()

        # Supuestos que AUMENTAN el beneficio -> ROI sube.
        for nombre, valor in (("eventos_anuales_estimados", 6),
                              ("valor_por_animal", 400_000),
                              ("animales_por_evento", 6),
                              ("evitacion_por_deteccion", 0.95),
                              ("horas_vigilancia_ahorradas_semana", 8),
                              ("valor_hora_vigilancia", 15_000)):
            with self.subTest(parametro=nombre):
                self.assertGreater(
                    self._roi(**{nombre: valor}), base,
                    f"subir '{nombre}' deberia aumentar el ROI")

        # `detecciones_por_evento` es una fraccion y ya vale 1.0 por defecto,
        # asi que el extremo que mueve el ROI es BAJARLA.
        self.assertLess(self._roi(detecciones_por_evento=0.4), base,
                        "bajar la tasa de deteccion deberia reducir el ROI")

        # Supuestos que AUMENTAN el costo -> ROI baja.
        for nombre, valor in (("costo_camara", 900_000),
                              ("costo_equipo", 1_200_000),
                              ("costo_instalacion", 400_000),
                              ("costo_mantenimiento_anual", 300_000)):
            with self.subTest(parametro=nombre):
                self.assertLess(
                    self._roi(**{nombre: valor}), base,
                    f"subir '{nombre}' deberia reducir el ROI")

    def test_roi_moves_toward_the_annual_return_as_years_grow(self):
        """El ROI de N años tiende a `utilidad_neta / costo_anual`.

        No siempre baja al ampliar el horizonte: si el proyecto rinde mucho
        por año, ampliarlo acerca la utilidad al asintótico y el ROI SUBE. Lo
        que no puede hacer es dar saltos: tiene que moverse de forma monótona
        hacia ese límite. Por eso la prueba mira la tendencia, no una dirección
        inventada."""
        informe = ROIAnalysis(self.base).compute()
        asintotico = (informe["utilidad_neta_anual_clp"]
                      / informe["costo_anual_clp"] * 100)
        rois = [(n, self._roi(anios_analizados=n)) for n in (1, 3, 5, 10, 20)]

        for (n1, r1), (n2, r2) in zip(rois, rois[1:]):
            with self.subTest(anios=f"{n1}->{n2}"):
                if r1 < asintotico:
                    self.assertGreater(r2, r1,
                                       f"por debajo del limite ({r1:.1f} < "
                                       f"{asintotico:.1f}) el ROI deberia subir")
                else:
                    self.assertLess(r2, r1,
                                    f"por encima del limite ({r1:.1f} > "
                                    f"{asintotico:.1f}) el ROI deberia bajar")
        for n, roi in rois:
            with self.subTest(anios=n):
                distancia_inicial = abs(asintotico - rois[0][1])
                self.assertLessEqual(abs(asintotico - roi), distancia_inicial + 1e-6,
                                     f"con {n} años el ROI deberia estar mas "
                                     "cerca del limite que con 1 año")

    def test_implausible_roi_raises_a_warning(self):
        """Un ROI desproporcionado casi siempre es un supuesto erroneo. El
        aviso existe para que no se presente sin revisarlo."""
        report = ROIAnalysis(self.base).compute()
        if report["roi_porcentaje"] > ROI_PLAUSIBLE_MAX_PCT:
            self.assertTrue(
                any("supera" in w for w in report["advertencias"]),
                f"ROI de {report['roi_porcentaje']}% sin aviso de plausibilidad")

    def test_negative_utility_raises_a_warning(self):
        """Un ROI negativo significa que el sistema cuesta dinero. Tiene que
        decirlo, no devolver un porcentaje y dejar que parezca rentable."""
        report = ROIAnalysis(self.base, params={
            "valor_por_animal": 1,
            "eventos_anuales_estimados": 0,
            "horas_vigilancia_ahorradas_semana": 0,
        }).compute()
        self.assertLess(report["utilidad_neta_anual_clp"], 0)
        self.assertTrue(any("negativa" in w for w in report["advertencias"]),
                        f"sin aviso de utilidad negativa: {report['advertencias']}")
        self.assertIsNone(report["retorno_inversion_anos"],
                          "sin utilidad neta no hay retorno que mostrar")

    def test_scenarios_follow_the_alert_latency(self):
        """La latencia se toma de la cadena real (fps + timeout). Con menos
        minutos de permanencia, la alert llega tarde y la efectividad baja."""
        analysis = ROIAnalysis(self.base)
        report = analysis.compute()
        latencia = analysis.detection_latency()
        esc = report["escenarios"]

        # La latencia no depende del escenario: sale de la configuracion real.
        for nombre in ("optimista", "esperado", "pesimista"):
            with self.subTest(escenario=nombre):
                self.assertEqual(esc[nombre]["minutos_disponibles"], 3
                                 if nombre == "optimista" else
                                 6 if nombre == "esperado" else 12)
        # A mas tiempo en el perimetro, mas fraccion del tiempo se cubre.
        self.assertLessEqual(esc["optimista"]["efectividad"],
                             esc["esperado"]["efectividad"])
        self.assertLessEqual(esc["esperado"]["efectividad"],
                             esc["pesimista"]["efectividad"])
        for nombre, escen in esc.items():
            with self.subTest(escenario=nombre):
                self.assertGreaterEqual(escen["efectividad"], 0.0)
                self.assertLessEqual(escen["efectividad"], 1.0)
        # Si la alerta tarda mas que la permanencia, la evitacion es cero.
        if latencia["total_local_s"] / 60.0 >= 3:
            self.assertEqual(esc["optimista"]["efectividad"], 0.0)
            self.assertFalse(esc["optimista"]["llega_a_tiempo"])

    def test_latency_uses_real_chain_values_not_hardcoded_ones(self):
        """La latencia se compone de fps_limit y timeout de red reales. Si
        estan hardcodeados, el numero no describe ESTE despliegue."""
        base = copy.deepcopy(self.base)
        base["camera"]["fps_limit"] = 25
        base["alerts"]["remote"]["timeout_seconds"] = 2
        lat = ROIAnalysis(base).detection_latency()
        self.assertAlmostEqual(lat["inferencia_s"], 1 / 25, places=3)
        self.assertAlmostEqual(lat["total_local_s"], 1 / 25 + 2, places=3)

    def test_config_overrides_code_defaults(self):
        """Los supuestos se leen de `evaluation.roi` en config.yaml. Si un valor
        de config se ignorara, el README estaria mintiendo al decir que son
        editables."""
        custom = copy.deepcopy(self.base)
        custom.setdefault("evaluation", {})["roi"] = {
            "eventos_anuales_estimados": 20,
            "valor_por_animal": 1_000_000,
        }
        report = ROIAnalysis(custom).compute()
        self.assertEqual(report["parametros"]["eventos_anuales_estimados"], 20)
        self.assertEqual(report["parametros"]["valor_por_animal"], 1_000_000)

    def test_parameters_are_kept_in_the_report(self):
        """El informe debe guardar los supuestos usados. Un ROI sin los numeros
        que lo producen no es auditable."""
        report = ROIAnalysis(self.base, params={"eventos_anuales_estimados": 7}
                             ).compute()
        self.assertEqual(report["parametros"]["eventos_anuales_estimados"], 7)
        self.assertIn("nota_metodologica", report)
        self.assertTrue(report["nota_metodologica"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
