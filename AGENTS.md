# Agent notes

# OBJETIVO DEL PROYECTO

Desarrollar un sistema de reconocimiento de gestos de manos robusto y escalable
en tiempo real utilizando:

- Python
- OpenCV
- MediaPipe 1.x
- MediaPipe Tasks

El sistema debe ser robusto ante:

- diferentes posiciones de la mano
- diferentes orientaciones
- diferentes escalas/distancias
- mano izquierda y derecha
- pequeñas variaciones de postura
- movimiento moderado
- cambios razonables de iluminación

---

# REGLA CRITICA: NO MODIFICAR EL CÓDIGO SIN AUTORIZACIÓN

El agente NO debe modificar, crear, eliminar, mover ni sobrescribir
archivos del proyecto a menos que el usuario lo solicite
explícitamente.


Si el usuario está:

- haciendo una pregunta
- pidiendo una explicación
- preguntando cómo funciona algo
- preguntando qué opciones existen
- preguntando qué métodos tiene una clase o función
- preguntando cómo debería implementarse algo
- solicitando una comparación
- solicitando recomendaciones
- mostrando un error
- preguntando por una posible solución

NO modificar ningún archivo.

Cuando exista más de una forma razonable de implementar algo,
informar al usuario de las alternativas antes de modificar el código.

No seleccionar silenciosamente una alternativa importante
sin informar al usuario.

## Informar sobre métodos y opciones disponibles

Cuando el usuario pregunte por una clase, función, API o componente,
no limitarse a explicar únicamente el método que parece necesario.

Informar también sobre las principales opciones disponibles
cuando sean relevantes.

Por ejemplo, si el usuario pregunta por:

`HandLandmarker`

informar sobre elementos relevantes como:

- `detect()`
- `detect_for_video()`
- `detect_async()`
- `HandLandmarkerOptions`
- `RunningMode`
- `num_hands`
- `min_hand_detection_confidence`
- `min_hand_presence_confidence`
- `min_tracking_confidence`
- `result_callback`

Explicar brevemente para qué sirve cada uno y cuál es el más
adecuado para el proyecto.

No es necesario enumerar absolutamente todos los elementos
internos de una API. Priorizar los que sean útiles para la decisión
actual.

---

# REGLA CRITICA: MEDIAPIPE

Este proyecto utiliza la API MODERNA de MediaPipe.

NO utilizar APIs antiguas de MediaPipe Solutions.

Como por ejemplo:

- `mp.solutions`

No utilizar tutoriales antiguos basados en MediaPipe Solutions
sin comprobar primero su compatibilidad con MediaPipe 1.x.

La arquitectura podria utilizar:

- `mp.tasks`
- `mp.tasks.vision`
- `HandLandmarker`
- `HandLandmarkerOptions`
- `GestureRecognizer`
- `GestureRecognizerOptions`
- `BaseOptions`
- `RunningMode`

Antes de utilizar una API de MediaPipe, verificar la documentación
oficial actual.

Fuentes prioritarias:

1. Documentación oficial de Google AI Edge / MediaPipe
2. Repositorio oficial de MediaPipe
3. Ejemplos oficiales
4. Otras fuentes únicamente como referencia secundaria

Si existe conflicto entre un tutorial externo y la documentación
oficial actual, utilizar la documentación oficial.


---

# Arquitectura

Separar:

1. Captura de cámara
2. Conversión de imagen
3. Detección de mano
4. Extracción de landmarks
5. Normalización
6. Análisis geométrico
7. Clasificación del gesto
8. Filtrado temporal
9. Interfaz / visualización

No mezclar toda la lógica en un único archivo.

---

# Landmarks

Utilizar los 21 landmarks de la mano.

No clasificar el gesto únicamente mediante coordenadas
absolutas de píxeles.

La clasificación debe ser robusta frente a:

- traslación
- escala
- orientación
- mano izquierda/derecha

Preferir distancias normalizadas, ángulos y relaciones geométricas.

---

# Tiempo real

La aplicación debe estar preparada para webcam.

Cuando se utilice LIVE_STREAM:

- utilizar detect_async()
- utilizar timestamps monotónicamente crecientes
- utilizar result_callback
- no asumir que cada frame produce necesariamente un resultado

---

# Rendimiento

No realizar operaciones costosas innecesarias por frame.

Evitar:

- crear modelos repetidamente
- crear objetos MediaPipe en cada frame
- cargar modelos continuamente
- cálculos redundantes

Inicializar los modelos una sola vez.

Liberar correctamente los recursos.

---

# Calidad

Cada cambio debe:

1. explicar qué se modificó
2. verificar que el código compila
3. comprobar compatibilidad con la versión instalada
4. comprobar errores
5. evitar APIs deprecated 

No modificar dependencias sin justificarlo.

---

# Investigación

Cuando exista una duda sobre MediaPipe:

PRIORIDAD:

1. documentación oficial de Google AI Edge / MediaPipe
2. repositorio oficial de MediaPipe
3. ejemplos oficiales
4. otras fuentes

No utilizar blogs antiguos como fuente principal.

Si una fuente utiliza mp.solutions, comprobar primero
si corresponde a una versión antigua.