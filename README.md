# Gesture Control

Sistema de reconocimiento de gestos de manos en tiempo real utilizando **Python**, **OpenCV** y la API moderna de **MediaPipe Tasks**.

El sistema esta disenado para detectar manos a distancia de forma robusta, combinando deteccion de pose corporal (para estimar la posicion de las munecas) con deteccion especifica de manos sobre regiones de interes (ROI) dinamicas.

---

## Requisitos

- Python 3.10+
- OpenCV
- NumPy
- MediaPipe 0.10.35+
- Webcam funcional

Instalacion de dependencias:

```bash
pip install -r requirements.txt
```

---

## Estructura del proyecto

```
Gesture-control/
├── main.py                    # Punto de entrada principal
├── test_full_frame.py         # Prueba de deteccion sobre frame completo (sin ROI)
├── requirements.txt           # Dependencias
├── AGENTS.md                  # Notas para agentes de desarrollo
├── README.md                  # Este archivo
│
├── config/
│   └── settings.py            # Configuracion centralizada (dataclasses inmutables)
│
├── core/                      # Dominio: logica de negocio y tipos
│   ├── types.py               # Enums (TrackingState, TrackingMode) y HandState
│   ├── roi_estimator.py       # Calculo de regiones de interes (ROI)
│   └── state_machine.py       # Maquina de estados de seguimiento (thread-safe)
│
├── detection/                 # Deteccion y pipeline de imagenes
│   ├── base_detector.py       # Protocolo comun para detectores asincronos
│   ├── hand_detector.py       # Wrapper de HandLandmarker (MediaPipe)
│   ├── pose_detector.py       # Wrapper de PoseLandmarker (MediaPipe)
│   └── image_pipeline.py      # Conversiones BGR->RGB, crop ROI, transformacion de landmarks
│
├── infrastructure/            # Infraestructura tecnica
│   ├── camera.py              # Captura de video via OpenCV
│   └── timing.py              # Control de FPS (limitador y contador)
│
├── presentation/              # Capa de presentacion
│   └── visualizer.py          # Dibujo de landmarks, ROI, brazos y HUD
│
├── utils/                     # Utilidades transversales
│   └── logging_config.py      # Configuracion de logging
│
└── tests/                     # Espacio reservado para tests
    └── __init__.py
```

---

## Como ejecutar

### Modo principal (con ROI dinamico y pose)

```bash
python main.py
```

Opciones disponibles:

```bash
python main.py --camera 0 --fps 30 --width 640 --height 480 --log-level INFO
```

| Parametro     | Descripcion                  | Default  |
|---------------|------------------------------|----------|
| -c, --camera  | Indice de la camara          | 0        |
| --fps         | FPS objetivo del bucle       | 30       |
| --width       | Ancho del frame              | 640      |
| --height      | Alto del frame               | 480      |
| --log-level   | Nivel de logging             | INFO     |

Durante la ejecucion:
- Presiona **q** o **ESC** para salir.
- La imagen se muestra con efecto espejo por defecto (mirror=True).

### Modo test full-frame (sin ROI, sin pose)

```bash
python test_full_frame.py
```

Este modo pasa el frame completo directamente a HandLandmarker, util para comparar rendimiento o validar el modelo de manos sin la capa de tracking.

---

## Flujo de funcionamiento completo

### 1. Inicio y configuracion

1. main.py parsea los argumentos de linea de comandos (argparse).
2. Se configura el logging con formato consistente (utils/logging_config.py).
3. Se construye el objeto Config (config/settings.py), que agrupa subconfiguraciones:
   - CameraConfig: indice, resolucion, espejo, FPS.
   - ModelConfig: rutas a modelos y umbrales de confianza.
   - ROIConfig: factores de escala, margenes y limites de ROI.
   - TrackingConfig: umbrales de histeresis para cambios de modo y perdida de tracking.

### 2. Inicializacion de componentes

Se crean las siguientes instancias **una sola vez** (no por frame):

| Componente                 | Archivo                        | Proposito                                                            |
|----------------------------|--------------------------------|----------------------------------------------------------------------|
| Camera                     | infrastructure/camera.py       | Abre la webcam via cv2.VideoCapture, fuerza codec MJPEG.             |
| FPSLimiter                 | infrastructure/timing.py       | Duerme el hilo principal para mantener el target FPS.                |
| FPSCounter                 | infrastructure/timing.py       | Cuenta los FPS reales cada segundo.                                  |
| HandTrackingStateMachine   | core/state_machine.py          | Decide que ROI usar y empareja manos entre frames.                   |
| PoseDetector               | detection/pose_detector.py     | Carga pose_landmarker_full.task en modo LIVE_STREAM.                 |
| HandDetector               | detection/hand_detector.py     | Carga hand_landmarker.task en modo LIVE_STREAM.                      |
| Visualizer                 | presentation/visualizer.py     | Prepara el renderer de debug.                                        |

### 3. Callbacks asincronos

MediaPipe ejecuta la inferencia en hilos internos. Cuando termina, invoca:

- pose_callback(result) -> state_machine.on_pose_result(result)
- hand_callback(result) -> state_machine.on_hand_result(result)

Estos callbacks actualizan el estado interno de la maquina de estados, que esta protegido por un threading.Lock.

### 4. Bucle principal (por frame)

```
while True:
    1. fps_limiter.wait()               # Respeta el target FPS
    2. frame = camera.read()            # Lee frame BGR de OpenCV
    3. cv2.flip(frame, 1) si mirror     # Efecto espejo
    4. rgb_full = bgr_to_rgb(frame)     # BGR -> RGB contiguo
    5. mp_image_full = create_mp_image(rgb_full)  # Envuelve en MPImage
    6. timestamp_ms = (time_ns - start) // 1_000_000

    7. if state_machine.should_run_pose_detection():
           pose_detector.detect_async(mp_image_full, timestamp_ms)

    8. roi = state_machine.get_roi_for_hand_detection(frame.shape)
    9. state_machine.set_current_hand_roi(roi)

    10. roi_rgb = crop_roi(rgb_full, roi)
    11. mp_image_roi = create_mp_image(roi_rgb)
    12. hand_detector.detect_async(mp_image_roi, timestamp_ms)

    13. fps_counter.tick()
    14. display = visualizer.draw(frame, hands, pose, roi, state, mode, fps)
    15. cv2.imshow("Gesture Control", display)
    16. if q/ESC: break
```

### 5. Maquina de estados y modos de tracking

El HandTrackingStateMachine opera en dos modos de ROI:

#### Modo WRISTS (deteccion a distancia)
- Usa las munecas detectadas por PoseLandmarker para crear un ROI amplio.
- Ideal cuando las manos estan lejos o aun no se han detectado.
- Requiere pose fresca en cada frame.

#### Modo HANDS (seguimiento cercano)
- Usa las posiciones conocidas de las 2 manos para generar ROIs individuales y luego unificarlos.
- Activa despues de 3 frames consecutivos con 2 manos detectadas.
- Pose se ejecuta solo cada 30 frames (recalibracion periodica).

#### Estados posibles

| Estado     | Significado                                       | Transicion tipica                        |
|------------|---------------------------------------------------|------------------------------------------|
| DETECTING  | Buscando manos, usa ROI de munecas                | 2 manos detectadas -> TRACKING           |
| TRACKING   | 2 manos estables, sigue en modo HANDS             | Pierde manos -> LOST o DETECTING         |
| LOST       | Perdido tracking, vuelve a WRISTS y busca de nuevo| Detecta pose/munecas -> DETECTING        |

#### Histeresis de cambio de modo

- WRISTS -> HANDS: requiere 3 frames consecutivos con 2 manos.
- HANDS -> WRISTS: requiere 5 frames consecutivos sin 2 manos.
- Esto evita oscilaciones rapidas entre modos.

### 6. Mini-tracker por proximidad

Si una mano no se detecta en un frame concreto, no desaparece instantaneamente:

- Se incrementa `lost_frames`.
- La mano se mantiene en su ultima posicion conocida.
- Si supera `hand_max_lost_frames` (3), se descarta.

Cuando llegan nuevas detecciones, se emparejan con las manos existentes por distancia euclidea minima (emparejamiento greedy). Las manos se ordenan siempre de izquierda a derecha para identidad visual estable.

### 7. Pipeline de imagen

El modulo detection/image_pipeline.py centraliza toda la transformacion de imagenes:

| Funcion                        | Entrada                | Salida                                      |
|--------------------------------|------------------------|---------------------------------------------|
| bgr_to_rgb(frame)              | np.ndarray BGR         | np.ndarray RGB contiguo                     |
| create_mp_image(rgb)           | np.ndarray RGB         | MPImage (MediaPipe)                         |
| crop_roi(frame, roi)           | frame + (x1,y1,x2,y2)  | np.ndarray recortado (copia)                |
| denormalize_landmarks_to_roi   | landmarks [0,1] + roi  | (x,y,z) en coordenadas absolutas del frame  |

### 8. Visualizacion

El Visualizer dibuja sobre una copia del frame BGR:

- **ROI**: rectangulo cyan.
- **Pose**: lineas azules de hombro-codo-muneca (si visibilidad > 0.5).
- **Manos**: 21 landmarks con conexiones (verde para mano 1, naranja para mano 2).
- **HUD**: estado (DETECTING/TRACKING/LOST), modo (WRISTS/HANDS), FPS, confianza por mano.

### 9. Cierre y liberacion de recursos

Los tres componentes principales (Camera, PoseDetector, HandDetector) se abren con context managers (`with`), garantizando que:

- La camara se libera (`cap.release()`).
- Los modelos MediaPipe se cierran (`landmarker.close()`).
- Las ventanas de OpenCV se destruyen (`cv2.destroyAllWindows()`).

---

## Configuracion avanzada

Puedes modificar los valores por defecto editando `config/settings.py`:

| Parametro                            | Default | Descripcion                                         |
|--------------------------------------|---------|-----------------------------------------------------|
| roi_size_factor                      | 2.2     | Multiplicador del tamano de mano para ROI en HANDS  |
| roi_min_size                         | 96      | Tamano minimo del ROI en pixeles                    |
| roi_max_size_ratio                   | 0.8     | Maximo ratio del ROI respecto al frame              |
| hand_lost_threshold                  | 5       | Frames sin manos antes de marcar LOST               |
| pose_recalibration_interval          | 30      | Frames entre redetecciones de pose en modo HANDS    |
| mode_switch_to_hands_threshold       | 3       | Frames con 2 manos para cambiar a HANDS             |
| mode_switch_to_wrists_threshold      | 5       | Frames sin 2 manos para volver a WRISTS             |
| hand_max_lost_frames                 | 3       | Frames que una mano persiste sin deteccion real     |

---

## Notas para desarrolladores

- **Thread-safety**: Los callbacks de MediaPipe corren en hilos internos. Todo acceso a `HandTrackingStateMachine` pasa por un `threading.Lock`.
- **Timestamps**: MediaPipe LIVE_STREAM requiere timestamps monotonicamente crecientes en milisegundos. Se calculan con `time.time_ns()` relativo al inicio del programa.
- **No bloqueante**: `detect_async()` retorna inmediatamente. El resultado llega al callback cuando la inferencia termina, posiblemente varios frames despues.
- **Modelos**: Deben descargarse previamente en la carpeta `models/`:
  - `pose_landmarker_full.task`
  - `hand_landmarker.task`
