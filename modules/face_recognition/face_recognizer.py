import os
import cv2
import json
import numpy as np
import logging
from deepface import DeepFace
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "face_recognition_log.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)

def save_recognition_log(record: dict):
    """
    Append one recognition record to file.
    """
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------- LOGGER ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceRecognition")

# ---------------- PATHS ----------------
FACES_DIR = "data/processed/faces"
FACE_DATABASE_DIR = "data/face_database"

# Similarity threshold (tune if needed)
THRESHOLD = 0.35   # start low for live webcam, increase later

# ---------------- COSINE SIMILARITY ----------------
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def cosine_sim(a, b):
    """
    Compute cosine similarity between two vectors.
    """
    if SKLEARN_AVAILABLE:
        return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]
    else:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------------- FACE EMBEDDING ----------------
def get_face_embedding(face_img):
    """
    Extract normalized FaceNet embedding from a face image.
    Raises error if no face is detected.
    """
    try:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        result = DeepFace.represent(
            img_path=face_img,
            model_name="Facenet",
            detector_backend="mtcnn",
            enforce_detection=True
        )

        embedding = np.array(result[0]["embedding"], dtype="float32")
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    except Exception:
        raise RuntimeError("Face not detected")


# ---------------- BUILD FACE DATABASE ----------------
def build_face_database():
    """
    Loads known faces and their metadata into memory.
    """
    database = {}

    if not os.path.exists(FACE_DATABASE_DIR):
        logger.warning("Face database directory not found")
        return database

    for person_id in os.listdir(FACE_DATABASE_DIR):
        person_path = os.path.join(FACE_DATABASE_DIR, person_id)

        if not os.path.isdir(person_path):
            continue

        info_path = os.path.join(person_path, "info.json")
        if not os.path.exists(info_path):
            logger.warning(f"Missing info.json for {person_id}, skipping")
            continue

        with open(info_path, "r") as f:
            person_info = json.load(f)

        embeddings = []

        for img_name in os.listdir(person_path):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            try:
                emb = get_face_embedding(image)
                embeddings.append(emb)
            except:
                logger.warning(f"No face detected in DB image: {img_name}")

        if embeddings:
            database[person_id] = {
                "info": person_info,
                "embeddings": embeddings
            }
            logger.info(
                f"Loaded {len(embeddings)} embeddings for {person_info['name']}"
            )

    return database


# ---------------- FACE RECOGNITION ----------------
def recognize_faces():
    """
    Recognize faces extracted in Phase-3 against known database.
    """
    database = build_face_database()
    logger.info(f"Known identities loaded: {len(database)}")

    if not database:
        logger.warning("No known faces available for recognition")
        return

    if not os.path.exists(FACES_DIR):
        logger.warning("Faces directory not found")
        return

    for face_img_name in os.listdir(FACES_DIR):
        face_path = os.path.join(FACES_DIR, face_img_name)
        face = cv2.imread(face_path)

        if face is None:
            logger.warning(f"Could not read face image: {face_img_name}")
            continue

        try:
            query_emb = get_face_embedding(face)
        except:
            logger.warning(f"No face detected in {face_img_name}, skipping")
            continue

        best_score = 0.0
        best_match_info = None

        # 🔑 CORRECT MATCHING LOOP
        for person_id, data in database.items():
            for db_emb in data["embeddings"]:
                score = cosine_sim(query_emb, db_emb)

                if score > best_score:
                    best_score = score
                    best_match_info = data["info"]

        # Debug similarity score
        logger.info(f"{face_img_name} best similarity = {best_score:.2f}")

        # Final decision
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if best_score >= THRESHOLD and best_match_info is not None:
            record = {
                "timestamp": timestamp,
                "face_image": face_img_name,
                "name": best_match_info.get("name"),
                "dob": best_match_info.get("dob"),
                "region": best_match_info.get("region"),
                "similarity": round(float(best_score), 2),
                "status": "KNOWN"
            }

            logger.info(
                f"{face_img_name} → {record['name']} "
                f"(DOB={record['dob']}, Region={record['region']}) ✅"
            )

        else:
            record = {
                "timestamp": timestamp,
                "face_image": face_img_name,
                "name": None,
                "dob": None,
                "region": None,
                "similarity": round(float(best_score), 2),
                "status": "UNKNOWN"
            }

            logger.warning(f"{face_img_name} → UNKNOWN 🚨")

        # 🔐 SAVE TO FILE (for both cases)
        save_recognition_log(record)


# ---------------- RUN ----------------
if __name__ == "__main__":
    recognize_faces()