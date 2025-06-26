# Head Tracker con LSL y MediaPipe

Este proyecto implementa un sistema de seguimiento de cabeza en tiempo real utilizando MediaPipe y OpenCV. El script está diseñado para capturar la orientación de la cabeza (Yaw, Pitch, Roll) y la distancia estimada a la cámara, transmitiéndolos mediante un stream LSL (Lab Streaming Layer). Es ideal para experimentos de interacción humano-computadora, estudios de atención visual o sincronización con otros dispositivos biométricos.

## Características

- Seguimiento facial en tiempo real con MediaPipe.
- Cálculo de orientación de la cabeza: Yaw, Pitch, Roll.
- Estimación de distancia en milímetros con base en la separación ocular.
- Transmisión de datos vía LSL en un stream llamado `HeadTracking`.
- Captura de eventos desde un stream LSL opcional (por ejemplo, triggers experimentales).
- Calibración inicial automática del punto neutro de orientación.
- Suavizado exponencial de los valores de salida.
- Visualización en vivo con anotaciones superpuestas.
- Registro de datos en archivos `.parquet` de alta eficiencia:
  - `*_head_tracking_data.parquet`
  - `*_landmarks_data.parquet`

## Requisitos

Instala las siguientes dependencias usando pip:

```bash
pip install opencv-python mediapipe pandas pyarrow pylsl numpy
```

## Instrucciones rápidas

Instala dependencias (si usarás el código fuente):

```bash

pip install opencv-python mediapipe pandas pyarrow pylsl numpy
```

### Ejecuta el sistema:

```bash
python head_tracker.py
```

**También puedes usar el ejecutable incluido si estás en Windows (head_tracker.exe)**