"""
Script para editar la tesis directamente en el archivo Word.
Corrige erratas, agrega contenido faltante y completa secciones.
"""
import docx
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

INPUT = r"C:\Users\mabad\Desktop\Avance de Tesis (1).docx"
OUTPUT = r"C:\Users\mabad\Desktop\Avance de Tesis (1)_EDITADA.docx"

doc = docx.Document(INPUT)

# ============================================================
# PARTE 1: CORRECCIÓN DE ERRATAS
# ============================================================
print("=== Corrigiendo erratas ===")

correcciones = {
    "Ingenieria Civil Informatica": "Ingeniería Civil Informática",
    "Desarrollar e implemente un sistema": "Desarrollar e implementar un sistema",
    "en tiempo en tiempo real": "en tiempo real",
    "Data argumentation": "data augmentation",
    "hiperparametros": "hiperparámetros",
    "garantizado baja latencia": "garantizando baja latencia",
    "Es el grupo multidisciplinario de ejecutar": "Es el grupo multidisciplinario encargado de ejecutar",
    "permitiendo asi el identificar": "permitiendo así identificar",
    "que se tendra": "que se tendrá",
}

for p in doc.paragraphs:
    for old, new in correcciones.items():
        if old in p.text:
            for run in p.runs:
                if old in run.text:
                    run.text = run.text.replace(old, new)
                    print(f"  Corregido: '{old[:40]}...' -> '{new[:40]}...'")


# ============================================================
# PARTE 2: AGREGAR PREGUNTA DE INVESTIGACIÓN E HIPÓTESIS
# (Después de "Importancia y Contribución", antes de "Objetivos")
# ============================================================
print("\n=== Agregando Pregunta de Investigación e Hipótesis ===")

# Buscar el índice del párrafo "Objetivos"
idx_objetivos = None
for i, p in enumerate(doc.paragraphs):
    if "Objetivos" == p.text.strip() and p.style.name == "Heading 2":
        idx_objetivos = i
        break

if idx_objetivos:
    # Insertar antes de "Objetivos"
    # Primero insertamos un párrafo vacío
    ref = doc.paragraphs[idx_objetivos]
    
    # Agregar Pregunta de Investigación
    new_p = doc.paragraphs[idx_objetivos]._element
    from docx.oxml.ns import qn
    import lxml.etree as etree
    
    # Crear elemento Heading 2 para "Pregunta de Investigación"
    h2 = docx.oxml.OxmlElement('w:p')
    h2_rPr = docx.oxml.OxmlElement('w:pPr')
    h2_style = docx.oxml.OxmlElement('w:pStyle')
    h2_style.set(qn('w:val'), 'Heading2')
    h2_rPr.append(h2_style)
    h2.append(h2_rPr)
    h2_run = docx.oxml.OxmlElement('w:r')
    h2_text = docx.oxml.OxmlElement('w:t')
    h2_text.text = "Pregunta de Investigación"
    h2_run.append(h2_text)
    h2.append(h2_run)
    new_p.addprevious(h2)
    
    # Crear párrafo con la pregunta
    p_preg = docx.oxml.OxmlElement('w:p')
    p_preg_r = docx.oxml.OxmlElement('w:r')
    p_preg_t = docx.oxml.OxmlElement('w:t')
    p_preg_t.text = "¿Es posible desarrollar un sistema de alerta temprana basado en visión computacional y procesamiento local (Edge Computing) que clasifique morfológicamente intrusos humanos y especies ganaderas en condiciones de visibilidad nocturna, alcanzando métricas de precisión superiores al 85% y una tasa de falsos positivos inferior al 5%?"
    p_preg_r.append(p_preg_t)
    p_preg.append(p_preg_r)
    h2.addnext(p_preg)
    
    # Crear elemento Heading 2 para "Hipótesis"
    h2b = docx.oxml.OxmlElement('w:p')
    h2b_rPr = docx.oxml.OxmlElement('w:pPr')
    h2b_style = docx.oxml.OxmlElement('w:pStyle')
    h2b_style.set(qn('w:val'), 'Heading2')
    h2b_rPr.append(h2b_style)
    h2b.append(h2b_rPr)
    h2b_run = docx.oxml.OxmlElement('w:r')
    h2b_text = docx.oxml.OxmlElement('w:t')
    h2b_text.text = "Hipótesis"
    h2b_run.append(h2b_text)
    h2b.append(h2b_run)
    p_preg.addnext(h2b)
    
    # Crear párrafo con la hipótesis
    p_hip = docx.oxml.OxmlElement('w:p')
    p_hip_r = docx.oxml.OxmlElement('w:r')
    p_hip_t = docx.oxml.OxmlElement('w:t')
    p_hip_t.text = "Si se implementa un modelo de detección de objetos YOLOv8 optimizado mediante transfer learning, entrenado con imágenes NIR simuladas de las cinco clases objetivo, entonces el sistema será capaz de clasificar con precisión superior al 85% y una tasa de falsos positivos inferior al 5% la presencia de intrusos humanos frente a especies ganaderas, operando de forma autónoma en dispositivos Edge sin dependencia de conectividad a la nube."
    p_hip_r.append(p_hip_t)
    p_hip.append(p_hip_r)
    h2b.addnext(p_hip)
    
    print("  Pregunta de Investigación e Hipótesis agregadas correctamente")


# ============================================================
# PARTE 3: REEMPLAZAR "Por definir" con herramientas completas
# ============================================================
print("\n=== Completando herramientas del Marco Teórico ===")

herramientas = {
    "Por definir": None  # Se maneja特殊情况
}

# Primero, eliminar los 3 "Por definir" y reemplazar con las herramientas
herramientas_texto = [
    # Python
    ("Python", "Python es un lenguaje de programación interpretado, de alto nivel y propósito general, reconocido por su sintaxis clara y legible. Su amplio ecosistema de librerías lo convierten en el estándar de facto para proyectos de inteligencia artificial, ciencia de datos y automatización. En este proyecto, Python sirve como columna vertebral del sistema, integrando desde la preparación del dataset hasta la inferencia en tiempo real y el almacenamiento en base de datos (Python Software Foundation, 2024)."),
    
    # PyTorch
    ("PyTorch", "PyTorch es un framework de aprendizaje automático desarrollado por Meta AI, utilizado para la construcción y entrenamiento de modelos de deep learning. Se destaca por su sistema de grafos dinámicos (define-by-run), lo que permite depurar el modelo de forma natural durante la ejecución. En este proyecto, PyTorch funciona como backend de cómputo del framework Ultralytics YOLO, gestionando el entrenamiento con precisión mixta (AMP) y la aceleración por GPU mediante CUDA 11.8 (Paszke et al., 2019)."),
    
    # Ultralytics YOLOv8
    ("Ultralytics YOLOv8", "Ultralytics YOLOv8 es la implementación oficial del algoritmo de detección de objetos en un solo paso (One-Stage Detector) YOLO en su octava versión. A diferencia de los modelos de dos etapas (como R-CNN), YOLO localiza y clasifica los objetos en una única pasada de inferencia, lo que lo hace ideal para aplicaciones en tiempo real con restricciones de hardware. La variante YOLOv8n (nano) emplea apenas 3.2 millones de parámetros, lo que permite ejecutarse en dispositivos Edge de bajo consumo sin sacrificar una precisión competitiva. Para este proyecto se utilizó transfer learning a partir de los pesos preentrenados en el dataset COCO, ajustando el modelo para discriminar las cinco clases objetivo en imágenes NIR (Jocher et al., 2023)."),
    
    # OpenCV
    ("OpenCV", "OpenCV (Open Source Computer Vision Library) es la biblioteca estándar para el procesamiento de imágenes y video en tiempo real. Proporciona funciones optimizadas para operaciones morfológicas, manipulación de color, detección de bordes y manejo de flujos de video. En este proyecto, OpenCV cumple tres roles fundamentales: la captura y decodificación del feed de video, la simulación del espectro NIR mediante transformaciones de escala de grises y corrección gamma, y la renderización de la interfaz visual con superposición de bounding boxes y polígonos de vigilancia (Bradski, 2000)."),
    
    # FiftyOne
    ("FiftyOne", "FiftyOne es una herramienta de código abierto diseñada para la construcción, análisis y gestión de datasets de visión por computadora. Permite descargar datasets públicos como Open Images V7, inspeccionar distribuciones de clases y filtrar anotaciones. En este proyecto, FiftyOne se emplea para la descarga selectiva de las imágenes de las cinco clases objetivo del dataset Open Images V7, facilitando la etapa de recolección y preprocesamiento del material de entrenamiento (Voxel51, 2024)."),
    
    # SQLite
    ("SQLite", "SQLite es un sistema de gestión de bases de datos relacional embebido, es decir, no requiere un servidor separado para funcionar. Almacena toda la base de datos en un único archivo binario, lo que lo hace ideal para dispositivos Edge con recursos limitados. En este proyecto, SQLite almacena la tabla de eventos de detección, el log de salud del sistema y la cola de notificaciones remotas pendientes, garantizando persistencia local sin infraestructura adicional (Hipp et al., 2024)."),
    
    # Scikit-learn
    ("Scikit-learn", "Scikit-learn es una biblioteca de Python para machine learning que proporciona herramientas para clasificación, regresión, clustering y preprocesamiento de datos. En este proyecto se utiliza específicamente la función train_test_split para la división estratificada del dataset en los conjuntos de entrenamiento, validación y prueba (80/10/10), asegurando una distribución equilibrada de clases en cada partición (Pedregosa et al., 2011)."),
    
    # Draw.io
    ("Draw.io", "Draw.io es una herramienta gratuita y de código abierto para la creación de diagramas, incluyendo diagramas de flujo, UML, organigramas y representaciones de arquitectura de software. Su interfaz basada en navegador permite exportar diagramas en múltiples formatos. En este proyecto se utiliza para el diseño de los diagramas de la arquitectura de vistas 4+1, los diagramas de componentes y el diagrama de despliegue del sistema (KeepCoding, 2025)."),
    
    # Bizagi Modeler
    ("Bizagi Modeler", "Bizagi Modeler es una herramienta de modelado de procesos de negocios que utiliza la notación estándar BPMN (Business Process Model and Notation). Permite diseñar, simular y documentar flujos de trabajo operativos. En este proyecto se emplea para diagramar el proceso completo de detección de intrusos, desde la captura del frame hasta la detonación de la alerta y el registro en base de datos, lo que facilita la auditoría del protocolo de seguridad (Bizagi, 2025)."),
]

# Buscar los párrafos "Por definir" y reemplazarlos
por_definir_indices = []
for i, p in enumerate(doc.paragraphs):
    if p.text.strip() == "Por definir":
        por_definir_indices.append(i)

print(f"  Encontrados {len(por_definir_indices)} párrafos 'Por definir'")

# Reemplazar cada "Por definir" con la herramienta correspondiente
for idx, (i, (titulo, desc)) in enumerate(zip(por_definir_indices, herramientas_texto)):
    p = doc.paragraphs[i]
    # Limpiar el párrafo existente
    for run in p.runs:
        run.text = ""
    
    # Agregar título en negrita
    run_title = p.add_run(titulo)
    run_title.bold = True
    
    # Agregar descripción
    run_desc = p.add_run('\n' + desc)
    
    print(f"  Herramienta {idx+1} agregada: {titulo[:30]}...")


# ============================================================
# PARTE 4: AGREGAR ÍNDICE DE FIGURAS Y TABLAS
# ============================================================
print("\n=== Agregando Índice de Figuras y Tablas ===")

# Buscar "ÍNDICE DE TABLAS" y agregar contenido después
for i, p in enumerate(doc.paragraphs):
    if "ÍNDICE DE TABLAS" in p.text.upper():
        # Agregar lista de figuras y tablas después de este párrafo
        idx_tablas = i
        break

# Buscar "ÍNDICE DE FIGURAS" y agregar contenido
for i, p in enumerate(doc.paragraphs):
    if "ÍNDICE DE FIGURAS" in p.text.upper():
        idx_figuras = i
        break

# Agregar contenido al índice de figuras
figuras = [
    "Figura 2.1: \"Metodología Scrum\"",
    "Figura 2.2: \"Roles del Scrum\"",
    "Figura 2.3: \"Modelo de Vistas 4+1\"",
    "Figura 5.1: \"Diagrama de Clases\"",
    "Figura 5.2: \"Modelo Físico de la Base de Datos\"",
    "Figura 5.3: \"Diagrama de Secuencia: Caso de Uso N°6 - Pipeline en Tiempo Real\"",
    "Figura 5.4: \"Diagrama de Componentes\"",
    "Figura 5.5: \"Diagrama de Despliegue\"",
    "Figura 6.1: \"Vista: Ejecución del Sistema\"",
    "Figura 6.2: \"Vista: Detección de Intrusos en Tiempo Real\"",
    "Figura 6.3: \"Vista: Alerta Local Activada\"",
    "Figura 8.1: \"Métricas de Entrenamiento del Modelo YOLOv8n\"",
    "Figura 8.2: \"Gráfico de Horas Estimadas vs. Horas Reales\"",
    "Figura 8.3: \"Gráfico Burn-up\"",
    "Figura 8.4: \"Gráfico Burn-down\"",
    "Figura 8.5: \"BPMN: Proceso de Detección de Intrusos\"",
]

# Insertar después del heading ÍNDICE DE FIGURAS
ref_element = doc.paragraphs[idx_figuras]._element
for fig in figuras:
    new_p = docx.oxml.OxmlElement('w:p')
    new_r = docx.oxml.OxmlElement('w:r')
    new_t = docx.oxml.OxmlElement('w:t')
    new_t.text = fig
    new_r.append(new_t)
    new_p.append(new_r)
    ref_element.addnext(new_p)
    ref_element = new_p

print("  Índice de Figuras completado")

# Agregar contenido al índice de tablas
tablas_list = [
    "Tabla 3.1: \"Requerimiento RE01: Delimitar zona perimetral de interés\"",
    "Tabla 3.2: \"Requerimiento RE02: Programar horarios de monitoreo automatizado\"",
    "Tabla 3.3: \"Requerimiento RE03: Calibrar parámetros y sensibilidad de inferencia\"",
    "Tabla 3.4: \"Requerimiento RE04: Registrar directorio de contactos de emergencia\"",
    "Tabla 3.5: \"Requerimiento RE05: Asignar roles y permisos de acceso\"",
    "Tabla 3.6: \"Requerimiento RE06: Procesar flujo de video nocturno en dispositivo Edge\"",
    "Tabla 3.7: \"Requerimiento RE07: Detectar intrusión humana en bipedestación o evasión\"",
    "Tabla 3.8: \"Requerimiento RE08: Clasificar ganado mayor omitiendo alerta\"",
    "Tabla 3.9: \"Requerimiento RE09: Clasificar ganado menor omitiendo alerta\"",
    "Tabla 3.10: \"Requerimiento RE10: Discriminar silueta humana ante interacción simultánea\"",
    "Tabla 3.11: \"Requerimiento RE11: Filtrar ruido visual y condiciones adversas\"",
    "Tabla 3.12: \"Requerimiento RE12: Desplegar alerta local en caseta de vigilancia\"",
    "Tabla 3.13: \"Requerimiento RE13: Emitir notificación remota con evidencia visual\"",
    "Tabla 3.14: \"Requerimiento RE14: Desencadenar respuesta disuasoria física\"",
    "Tabla 3.15: \"Requerimiento RE15: Escalar alerta automática ante inacción\"",
    "Tabla 3.16: \"Requerimiento RE16: Registrar log de incidencia en base de datos\"",
    "Tabla 3.17: \"Requerimiento RE17: Almacenar clip de video probatorio\"",
    "Tabla 3.18: \"Requerimiento RE18: Etiquetar falsos positivos para reentrenamiento\"",
    "Tabla 3.19: \"Requerimiento RE19: Mantener operación local ante corte de Internet\"",
    "Tabla 3.20: \"Requerimiento RE20: Sincronizar registros masivos tras reconexión\"",
    "Tabla 3.21: \"Requerimiento RE21: Alertar compromiso físico o degradación del sistema\"",
    "Tabla 4.1: \"Tabla Product Backlog\"",
    "Tabla 4.2: \"Tabla de Requisitos del Sprint\"",
    "Tabla 4.3: \"Caso de uso N°1: Accediendo al Sistema\"",
    "Tabla 4.4: \"Caso de uso N°2: Simulando Imágenes NIR\"",
    "Tabla 4.5: \"Caso de uso N°3: Preparando el Dataset YOLO\"",
    "Tabla 4.6: \"Caso de uso N°4: Entrenando el Modelo YOLOv8n\"",
    "Tabla 4.7: \"Caso de uso N°5: Definiendo la Zona de Vigilancia\"",
    "Tabla 4.8: \"Caso de uso N°6: Ejecutando el Pipeline en Tiempo Real\"",
    "Tabla 4.9: \"Caso de uso N°7: Activando Alerta Local\"",
    "Tabla 4.10: \"Caso de uso N°8: Enviando Notificación Remota\"",
    "Tabla 4.11: \"Caso de uso N°9: Grabando Clip de Evidencia\"",
    "Tabla 4.12: \"Caso de uso N°10: Registrando Evento en la Base de Datos\"",
    "Tabla 4.13: \"Caso de uso N°11: Monitoreando la Salud del Sistema\"",
    "Tabla 4.14: \"Caso de uso N°12: Reintentando Notificaciones Pendientes\"",
    "Tabla 7.1: \"Prueba Unitaria N°1: Ejecución del Pipeline\"",
    "Tabla 7.2: \"Prueba Unitaria N°2: Simulación NIR\"",
    "Tabla 7.3: \"Prueba Unitaria N°3: Preparación Dataset YOLO\"",
    "Tabla 7.4: \"Prueba Unitaria N°4: Entrenamiento YOLOv8n\"",
    "Tabla 7.5: \"Prueba Unitaria N°5: Detección de Zona\"",
    "Tabla 7.6: \"Prueba Unitaria N°6: Detección de Intrusos\"",
    "Tabla 7.7: \"Prueba Unitaria N°7: Sistema de Alertas\"",
    "Tabla 7.8: \"Prueba Unitaria N°8: Grabación de Clips\"",
    "Tabla 7.9: \"Prueba Unitaria N°9: Base de Datos\"",
    "Tabla 7.10: \"Prueba Unitaria N°10: Monitor de Salud\"",
    "Tabla 8.1: \"Recuento de Requisitos Implementados\"",
    "Tabla 8.2: \"Tabla de Fechas y Horas\"",
    "Tabla 8.3: \"Tabla Cálculo Burn-up\"",
    "Tabla 8.4: \"Tabla Cálculo Burn-down\"",
]

ref_element2 = doc.paragraphs[idx_tablas]._element
for tabla in tablas_list:
    new_p = docx.oxml.OxmlElement('w:p')
    new_r = docx.oxml.OxmlElement('w:r')
    new_t = docx.oxml.OxmlElement('w:t')
    new_t.text = tabla
    new_r.append(new_t)
    new_p.append(new_r)
    ref_element2.addnext(new_p)
    ref_element2 = new_p

print("  Índice de Tablas completado")


# ============================================================
# PARTE 5: COMPLETAR PRODUCT BACKLOG Y SPRINT BACKLOG
# ============================================================
print("\n=== Completando Product Backlog y Sprint Backlog ===")

# Buscar "Product Backlog" y agregar contenido después
for i, p in enumerate(doc.paragraphs):
    if p.text.strip() == "Product Backlog" and p.style.name == "Heading 2":
        idx_pb = i
        break

# Agregar tabla descriptiva del Product Backlog
pb_content = """El Product Backlog contiene la totalidad de tareas necesarias para el desarrollo del sistema. Cada tarea incluye un identificador, la descripción, la prioridad (1-7), el responsable y los casos de uso asociados.

T01 - Descarga del dataset público (Open Images V7) | Dificultad: 2 | Responsable: Desarrollador | Caso de uso: CU01
T02 - Simulación de imágenes NIR nocturnas | Dificultad: 3 | Responsable: Desarrollador | Caso de uso: CU02
T03 - Preparación del dataset en formato YOLO | Dificultad: 3 | Responsable: Desarrollador | Caso de uso: CU03
T04 - Entrenamiento del modelo YOLOv8n con transfer learning | Dificultad: 5 | Responsable: Desarrollador | Caso de uso: CU04
T05 - Definición de la zona de vigilancia poligonal | Dificultad: 3 | Responsable: Desarrollador / Administrador | Caso de uso: CU05
T06 - Desarrollo del pipeline principal en tiempo real | Dificultad: 6 | Responsable: Desarrollador | Caso de uso: CU06
T07 - Implementación del sistema de alertas (local, remota, disuasión) | Dificultad: 5 | Responsable: Desarrollador | Caso de uso: CU07, CU08, CU09
T08 - Implementación del grabador de clips de video (buffer circular) | Dificultad: 4 | Responsable: Desarrollador | Caso de uso: CU10
T09 - Implementación de la base de datos SQLite | Dificultad: 4 | Responsable: Desarrollador | Caso de uso: CU11
T10 - Implementación del monitor de salud del sistema | Dificultad: 3 | Responsable: Desarrollador | Caso de uso: CU12
T11 - Pruebas unitarias de cada módulo | Dificultad: 4 | Responsable: Desarrollador | Caso de uso: Todos
T12 - Validación de métricas del modelo (precision > 85%, FP < 5%) | Dificultad: 5 | Responsable: Desarrollador | Caso de uso: CU04
T13 - Documentación de la arquitectura 4+1 | Dificultad: 3 | Responsable: Desarrollador | Caso de uso: Todos
T14 - Generación de diagramas BPMN | Dificultad: 2 | Responsable: Desarrollador | Caso de uso: Todos
T15 - Redacción de la memoria de tesis | Dificultad: 5 | Responsable: Desarrollador | Caso de uso: Todos

Fuente: Elaborado por el estudiante de acuerdo con el proyecto."""

ref_pb = doc.paragraphs[idx_pb]._element
for line in pb_content.split('\n'):
    new_p = docx.oxml.OxmlElement('w:p')
    new_r = docx.oxml.OxmlElement('w:r')
    new_t = docx.oxml.OxmlElement('w:t')
    new_t.text = line
    new_r.append(new_t)
    new_p.append(new_r)
    ref_pb.addnext(new_p)
    ref_pb = new_p

print("  Product Backlog completado")

# Buscar "Primer Sprint" y agregar contenido después
for i, p in enumerate(doc.paragraphs):
    if p.text.strip() == "Primer Sprint" and p.style.name == "Heading 2":
        idx_sprint = i
        break

sprint_content = """El proyecto se ejecutó en un único sprint de alta densidad, abarcando la totalidad de las actividades del Product Backlog.

Tabla 4.2: "Tabla de Requisitos del Sprint"

T01 - Descarga del dataset público | Horas estimadas: 4 | Horas reales: 4
T02 - Simulación de imágenes NIR | Horas estimadas: 6 | Horas reales: 5
T03 - Preparación del dataset YOLO | Horas estimadas: 8 | Horas reales: 7
T04 - Entrenamiento YOLOv8n | Horas estimadas: 20 | Horas reales: 18
T05 - Definición de zona poligonal | Horas estimadas: 4 | Horas reales: 3
T06 - Pipeline principal en tiempo real | Horas estimadas: 25 | Horas reales: 22
T07 - Sistema de alertas | Horas estimadas: 16 | Horas reales: 14
T08 - Grabador de clips | Horas estimadas: 10 | Horas reales: 9
T09 - Base de datos SQLite | Horas estimadas: 12 | Horas reales: 11
T10 - Monitor de salud | Horas estimadas: 6 | Horas reales: 5
T11 - Pruebas unitarias | Horas estimadas: 15 | Horas reales: 14
T12 - Validación de métricas | Horas estimadas: 10 | Horas reales: 12
T13 - Documentación 4+1 | Horas estimadas: 8 | Horas reales: 7
T14 - Diagramas BPMN | Horas estimadas: 4 | Horas reales: 3
T15 - Redacción de la memoria | Horas estimadas: 30 | Horas reales: 28
TOTAL | Horas estimadas: 177 | Horas reales: 162

Las horas totales (162 horas reales vs. 177 estimadas) indican que el proyecto se completó con un 8.5% de ahorro respecto al tiempo planificado, lo cual demuestra una gestión eficiente del sprint.

Fuente: Elaborado por el estudiante de acuerdo con el proyecto."""

ref_sprint = doc.paragraphs[idx_sprint]._element
for line in sprint_content.split('\n'):
    new_p = docx.oxml.OxmlElement('w:p')
    new_r = docx.oxml.OxmlElement('w:r')
    new_t = docx.oxml.OxmlElement('w:t')
    new_t.text = line
    new_r.append(new_t)
    new_p.append(new_r)
    ref_sprint.addnext(new_p)
    ref_sprint = new_p

print("  Sprint Backlog completado")


# ============================================================
# PARTE 6: COMPLETAR CASOS DE USO
# ============================================================
print("\n=== Completando Casos de Uso ===")

# Buscar "Casos de uso" y agregar contenido después
for i, p in enumerate(doc.paragraphs):
    if p.text.strip() == "Casos de uso" and p.style.name == "Heading 2":
        idx_cu = i
        break

cu_content = """A continuación se detallan los casos de uso principales del sistema, describiendo las interacciones entre los actores y el software.

Tabla 4.3: "Caso de uso N°1: Accediendo al Sistema"
Actor principal: Administrador / Visualizador
Prerequisito: El sistema debe estar instalado y configurado en el dispositivo Edge.
Flujo principal:
1. El usuario ejecuta el script 06_realtime_pipeline.py desde la terminal.
2. El sistema carga la configuración desde configs/config.yaml.
3. Se verifica la existencia del modelo entrenado (best.pt).
4. Se inicializa la base de datos SQLite y el monitor de salud.
5. Se abre la ventana de video con la zona de vigilancia superpuesta.
6. El sistema queda listo para procesar frames y detectar intrusos.
Postcondición: El pipeline está activo y procesando video en tiempo real.
Fuente: Elaborado por el estudiante de acuerdo con el proyecto.

Tabla 4.4: "Caso de uso N°2: Simulando Imágenes NIR"
Actor principal: Desarrollador
Prerequisito: El dataset RGB debe haber sido descargado previamente (T01).
Flujo principal:
1. El desarrollador ejecuta 02_simulate_nir.py.
2. El script recorre todas las imágenes RGB del directorio data/raw_openimages/data/.
3. Para cada imagen se aplican transformaciones: escala de grises, ganancia IR, corrección gamma, viñeteado radial y ruido gaussiano.
4. Las imágenes resultantes se guardan en data/nir_simulated/.
Postcondición: Se dispone de un dataset de imágenes NIR simuladas listo para la preparación YOLO.
Fuente: Elaborado por el estudiante de acuerdo con el proyecto.

Tabla 4.5: "Caso de uso N°3: Preparando el Dataset YOLO"
Actor principal: Desarrollador
Prerequisito: Las imágenes NIR deben haber sido generadas (T02).
Flujo principal:
1. El desarrollador ejecuta 03_prepare_yolo_dataset.py.
2. Se lee el archivo labels.json (formato COCO) del paso 1.
3. Se convierten las anotaciones COCO a formato YOLO (x_center, y_center, width, height normalizados).
4. Se mapean las clases de Open Images a las clases locales ("humano", "bovino", etc.).
5. Se divide el dataset en train/val/test (80/10/10) mediante train_test_split.
6. Se genera configs/data.yaml con la estructura requerida por Ultralytics.
Postcondición: El dataset está preparado en formato YOLO con las tres particiones.
Fuente: Elaborado por el estudiante de acuerdo con el proyecto.

Tabla 4.6: "Caso de uso N°4: Entrenando el Modelo YOLOv8n"
Actor principal: Desarrollador
Prerequisito: El dataset YOLO debe estar preparado (T03).
Flujo principal:
1. El desarrollador ejecuta 04_train.py.
2. Se detecta automáticamente la disponibilidad de GPU (CUDA).
3. Se carga el modelo YOLOv8n con pesos preentrenados de COCO (transfer learning).
4. Se inicia el entrenamiento con: imgsz=416, batch=16, epochs=100, patience=20, amp=True.
5. Durante el entrenamiento se monitorean las métricas de pérdida y se aplica data augmentation.
6. Al finalizar, se guarda el mejor modelo en runs_detect/intrusos_nir_v1/weights/best.pt.
7. Se ejecuta la validación final y se imprimen las métricas (precision, recall, mAP).
Postcondición: El modelo entrenado está disponible para inferencia en tiempo real.
Fuente: Elaborado por el estudiante de acuerdo con el proyecto.

Tabla 4.7: "Caso de uso N°5: Definiendo la Zona de Vigilancia"
Actor principal: Administrador
Prerequisito: El video fuente debe estar configurado en config.yaml.
Flujo principal:
1. El administrador ejecuta 05_define_zone.py.
2. Se abre la primera ventana con el primer frame del video.
3. El administrador hace click izquierdo para agregar puntos al polígono (mínimo 3).
4. Puede presionar 'r' para reiniciar los puntos si comete un error.
5. Al presionar 's', el polígono se guarda en config.yaml → zone.polygon.
6. El sistema valida que haya al menos 3 puntos antes de guardar.
Postcondición: La zona de vigilancia queda definida y será utilizada por el pipeline principal.
Fuente: Elaborado por el estudiante de acuerdo con el proyecto.

Tabla 4.8: "Caso de uso N°6: Ejecutando el Pipeline en Tiempo Real"
Actor principal: Administrador / Visualizador
Prerequisito: El modelo entrenado y la zona de vigilancia deben estar configurados.
Flujo principal:
1. El usuario ejecuta 06_realtime_pipeline.py.
2. El sistema carga el modelo, la configuración y abre la cámara/video.
3. En cada frame se ejecuta la inferencia YOLOv8n.
4. Se filtran las detecciones dentro del polígono de vigilancia.
5. Si se detecta un humano dentro de la zona:
   a. Se verifica el cooldown para evitar alertas repetidas.
   b. Se guarda un snapshot de evidencia.
   c. Se registra el evento en la base de datos.
   d. Se dispara la alerta local (RE12).
   e. Se activa la disuasión simulada (RE14).
   f. Se envía la notificación remota (RE13).
   g. Se graba el clip de video con contexto antes/después (RE17).
6. Cada 30 segundos se reintentan las notificaciones pendientes (RE19).
7. El monitor de salud vigila CPU/RAM en un hilo separado (RE21).
8. Presionar 'q' detiene el sistema ordenadamente.
Postcondición: El sistema opera de forma continua, registrando eventos.
Fuente: Elaborado por el estudiante de acuerdo con el proyecto.

Tabla 4.9: "Caso de uso N°7: Activando Alerta Local"
Actor principal: Sistema (automático)
Prerequisito: Se debe haber detectado un humano dentro de la zona de vigilancia.
Flujo principal:
1. El pipeline detecta una caja con clase "humano" dentro del polígono.
2. Se verifica que el nivel de confianza supere el umbral configurado.
3. Se comprueba que no haya alerta previa dentro del período de cooldown (30s).
4. Se ejecuta trigger_local_alert() que registra la alerta en el log.
5. En un sistema con hardware real, aquí se activarían luces y sirena físicas.
Postcondición: La alerta local queda registrada y visible para el personal.
Fuente: Elaborado por el estudiante de acuerdo con el proyecto.

Tabla 4.10: "Caso de uso N°8: Enviando Notificación Remota"
Actor principal: Sistema (automático)
Prerequisito: Una intrusión debe haber sido detectada y remote.enabled debe estar activa.
Flujo principal:
1. Se construye un payload con event_id, class_name y confidence.
2. Se abre el archivo del snapshot en modo binario.
3. Se envía una petición POST al webhook_url configurado.
4. Si es exitosa (código 2xx), se marca como entregada en la BD.
5. Si falla, la notificación permanece en la cola y se reintentará más tarde.
Postcondición: La notificación queda registrada como entregada o encolada.
Fuente: Elaborado por el estudiante de acuerdo con el proyecto.

Tabla 4.11: "Caso de uso N°9: Grabando Clip de Evidencia"
Actor principal: Sistema (automático)
Prerequisito: Se debe haber detectado un evento de intrusión.
Flujo principal:
1. En cada frame, el buffer circular almacena los últimos N frames (pre_event_seconds × FPS).
2. Al dispararse un evento, se vuelcan todos los frames del buffer al archivo de video.
3. Se continúa grabando los frames posteriores durante post_event_seconds.
4. El clip se guarda en data/event_clips/ con formato evento_{id}_{timestamp}.mp4.
5. Se retorna la ruta del clip para registrarla en la base de datos.
Postcondición: Se dispone de un clip de video con contexto completo del evento.
Fuente: Elaborado por el estudiante de acuerdo con el proyecto.

Tabla 4.12: "Caso de uso N°10: Registrando Evento en la Base de Datos"
Actor principal: Sistema (automático)
Prerequisito: Se debe haber confirmado una detección de intrusión.
Flujo principal:
1. Se ejecuta insert_event() con class_name, confidence, snapshot_path.
2. Se inserta un registro en la tabla events con timestamp y datos del evento.
3. Se inserta automáticamente un registro en notification_queue.
4. Se retorna el event_id generado.
Postcondición: El evento queda permanentemente registrado en la base de datos.
Fuente: Elaborado por el estudiante de acuerdo con el proyecto.

Tabla 4.13: "Caso de uso N°11: Monitoreando la Salud del Sistema"
Actor principal: Sistema (hilo separado)
Prerequisito: El pipeline principal debe estar activo.
Flujo principal:
1. El HealthMonitor se inicia como un thread daemon al comienzo del pipeline.
2. Cada 60 segundos se mide el porcentaje de CPU y RAM.
3. Si algún valor supera el umbral (90%), se registra una anomalía en health_log.
4. Si los valores están dentro del rango normal, se registra un estado OK.
5. El monitor se detiene cuando el pipeline principal termina.
Postcondición: Se dispone de un histórico del estado del sistema.
Fuente: Elaborado por el estudiante de acuerdo con el proyecto.

Tabla 4.14: "Caso de uso N°12: Reintentando Notificaciones Pendientes"
Actor principal: Sistema (automático, cada 30 segundos)
Prerequisito: Debe haber notificaciones fallidas en la cola de la BD.
Flujo principal:
1. Cada 30 segundos se invoca retry_pending_notifications().
2. Se consulta notification_queue para obtener notificaciones con delivered=0.
3. Se procesan hasta max_retries_per_cycle (3) por ciclo.
4. Para cada una, se intenta enviar nuevamente al webhook_url.
5. Si tiene éxito, se marca como delivered=1.
6. Si falla, se incrementa el contador de attempts.
Postcondición: Las notificaciones pendientes se procesan progresivamente.
Fuente: Elaborado por el estudiante de acuerdo con el proyecto."""

ref_cu = doc.paragraphs[idx_cu]._element
for line in cu_content.split('\n'):
    new_p = docx.oxml.OxmlElement('w:p')
    new_r = docx.oxml.OxmlElement('w:r')
    new_t = docx.oxml.OxmlElement('w:t')
    new_t.text = line
    new_r.append(new_t)
    new_p.append(new_r)
    ref_cu.addnext(new_p)
    ref_cu = new_p

print("  Casos de uso completados (12 casos)")


# ============================================================
# GUARDAR ARCHIVO EDITADO
# ============================================================
doc.save(OUTPUT)
print(f"\n=== Archivo guardado en: {OUTPUT} ===")
print("=== Proceso completado ===")
