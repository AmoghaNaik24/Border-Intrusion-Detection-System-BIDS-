from modules.motion_detection.motion_detector import run_motion_detection
from modules.object_detection.yolo_detector import run_object_detection
from modules.face_detection.face_detector import detect_and_extract_faces
from modules.face_recognition.face_recognizer import recognize_faces

def run_pipeline():
    print("\n=== PHASE 1: Motion Detection ===")
    run_motion_detection()

    print("\n=== PHASE 2: Object Detection ===")
    run_object_detection()

    print("\n=== PHASE 3: Face Detection ===")
    detect_and_extract_faces()

    print("\n=== PHASE 4: Face Recognition ===")
    recognize_faces()

    print("\n✅ PIPELINE COMPLETED")

if __name__ == "__main__":
    run_pipeline()