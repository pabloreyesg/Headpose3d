# === Verificación de dependencias ===
import importlib
import sys

required_modules = {
    'cv2': 'opencv-python',
    'mediapipe': 'mediapipe',
    'pandas': 'pandas',
    'pyarrow': 'pyarrow',
    'pylsl': 'pylsl',
    'numpy': 'numpy'
}

missing = [pkg for mod, pkg in required_modules.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("🚫 Faltan librerías necesarias:")
    print(f"pip install {' '.join(missing)}")
    sys.exit(1)

# === Librerías ===
import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_streams

# === Configuración ===
prefix = input("📂 Nombre del experimento o prefijo de archivo: ").strip() or "session"
save_every_n = 2  # ~15 FPS

# === Funciones ===
def get_head_orientation(landmarks, width, height):
    indices = [33, 263, 1, 61, 291, 199]
    image_points = np.array([[landmarks[i].x * width, landmarks[i].y * height] for i in indices], dtype="double")
    model_points = np.array([
        [-30.0, 0.0, -30.0], [30.0, 0.0, -30.0], [0.0, 0.0, 0.0],
        [-25.0, -30.0, -20.0], [25.0, -30.0, -20.0], [0.0, -60.0, -10.0]
    ])
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                               [0, focal_length, center[1]],
                               [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    try:
        success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        if not success:
            return 0.0, 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        x = np.arctan2(rmat[2, 1], rmat[2, 2])
        y = np.arctan2(-rmat[2, 0], sy)
        z = np.arctan2(rmat[1, 0], rmat[0, 0])
        return np.degrees(y), np.degrees(x), np.degrees(z)  # yaw, pitch, roll
    except:
        return 0.0, 0.0, 0.0

def get_distance(landmarks, width, height):
    l = np.array([landmarks[33].x * width, landmarks[33].y * height])
    r = np.array([landmarks[263].x * width, landmarks[263].y * height])
    d_px = np.linalg.norm(l - r)
    return (63.0 * 850 / d_px) if d_px > 0 else 0.0

def crop_to_square(frame):
    h, w = frame.shape[:2]
    min_dim = min(h, w)
    return frame[(h - min_dim)//2:(h + min_dim)//2, (w - min_dim)//2:(w + min_dim)//2]

def smooth(prev, curr, alpha=0.9):
    return alpha * prev + (1 - alpha) * curr

def initialize_trigger_listener(timeout=300):
    print("⏳ Esperando trigger 'DataSyncMarker487'...")
    start = time.time()
    while (time.time() - start) < timeout:
        for s in resolve_streams():
            if s.name() == "DataSyncMarker487" and s.type() == "Tags" and s.source_id() == "487":
                print("✅ Trigger conectado.")
                return StreamInlet(s)
        time.sleep(1)
    print("⚠️ No se detectó trigger. Continuando sin él.")
    return None

def calibrate_orientation(face_mesh, cap, duration=5):
    print(f"🧭 Calibración: mire al frente {duration} segundos...")
    values = []
    start = time.time()
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret: continue
        frame = crop_to_square(frame)
        h, w = frame.shape[:2]
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                yaw, pitch, roll = get_head_orientation(face.landmark, w, h)
                values.append([yaw, pitch, roll])
        cv2.putText(frame, "Calibrando: mire al frente...", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Calibración", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break
    cv2.destroyWindow("Calibración")
    if values:
        avg = np.mean(values, axis=0)
        print(f"📏 Offset calibrado: Yaw={avg[0]:.2f}, Pitch={avg[1]:.2f}, Roll={avg[2]:.2f}")
        return avg
    print("⚠️ Calibración fallida. Usando offset cero.")
    return np.array([0.0, 0.0, 0.0])

# === Main ===
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return

    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    yaw_offset, pitch_offset, roll_offset = calibrate_orientation(face_mesh, cap, duration=5)

    inlet = initialize_trigger_listener()
    outlet = StreamOutlet(StreamInfo('HeadTracking', 'HeadPose', 4, 30, 'float32', 'myuid34234'))

    data_log, landmarks_log = [], []
    prev_yaw = prev_pitch = prev_roll = prev_distance = 0.0
    first = True
    frame_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_counter += 1
            if frame_counter % save_every_n != 0: continue

            frame = crop_to_square(frame)
            h, w = frame.shape[:2]
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            trigger = None
            if inlet:
                sample, _ = inlet.pull_sample(timeout=0.0)
                if sample: trigger = sample[0]

            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, face, mp.solutions.face_mesh.FACEMESH_CONTOURS)

                    yaw_r, pitch_r, roll_r = get_head_orientation(face.landmark, w, h)
                    yaw_r -= yaw_offset
                    pitch_r -= pitch_offset
                    roll_r -= roll_offset
                    dist = get_distance(face.landmark, w, h)

                    if first:
                        yaw, pitch, roll, distance = yaw_r, pitch_r, roll_r, dist
                        first = False
                    else:
                        yaw = smooth(prev_yaw, yaw_r)
                        pitch = smooth(prev_pitch, pitch_r)
                        roll = smooth(prev_roll, roll_r)
                        distance = smooth(prev_distance, dist)

                    prev_yaw, prev_pitch, prev_roll, prev_distance = yaw, pitch, roll, distance
                    timestamp = time.time()

                    outlet.push_sample([yaw, pitch, roll, distance])
                    data_log.append([timestamp, yaw, pitch, roll, distance, trigger])

                    landmarks_flat = [timestamp, trigger] + [
                        coord for lm in face.landmark for coord in (lm.x * w, lm.y * h, lm.z)
                    ]
                    landmarks_log.append(landmarks_flat)

                    cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.putText(frame, f"Roll: {roll:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.putText(frame, f"Dist: {distance:.0f}mm", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break

    except KeyboardInterrupt:
        print("\n🛑 Interrupción manual.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if data_log:
            df = pd.DataFrame(data_log, columns=["Timestamp", "Yaw", "Pitch", "Roll", "Distance", "Trigger"])
            df.to_parquet(f"{prefix}_head_tracking_data.parquet", index=False)
            print(f"✅ {prefix}_head_tracking_data.parquet guardado.")
        if landmarks_log:
            cols = ["Timestamp", "Trigger"] + [f"L{i}_{a}" for i in range(468) for a in ("x", "y", "z")]
            df = pd.DataFrame(landmarks_log, columns=cols)
            df.to_parquet(f"{prefix}_landmarks_data.parquet", index=False)
            print(f"✅ {prefix}_landmarks_data.parquet guardado.")

if __name__ == "__main__":
    main()
