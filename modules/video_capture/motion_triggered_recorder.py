import cv2
import time
import os
from core.logger import logger


def capture_on_motion(record_duration=5):
    """
    Live webcam capture.
    Motion triggers recording of fixed-duration video (5 sec).
    """

    output_dir = "data/raw/chunks"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Webcam not accessible")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=300,
        varThreshold=50,
        detectShadows=False
    )

    recording = False
    out = None
    start_record_time = None
    video_path = None

    logger.info("Live motion-triggered recording started")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = bg_subtractor.apply(gray)

        motion_area = cv2.countNonZero(fg_mask)

        # ---------- MOTION TRIGGER ----------
        if motion_area > 5000 and not recording:
            timestamp = int(time.time())
            video_path = os.path.join(output_dir, f"motion_{timestamp}.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            recording = True
            start_record_time = time.time()

            logger.info(f"Motion detected → Recording started: {video_path}")

        # ---------- RECORD FIXED 5 SECONDS ----------
        if recording:
            out.write(frame)

            if time.time() - start_record_time >= record_duration:
                out.release()
                recording = False
                logger.info(f"Recording completed (5 sec): {video_path}")

                break  # 🔑 EXIT → PIPELINE CONTINUES

        cv2.imshow("Live Feed", frame)
        cv2.imshow("Motion Mask", fg_mask)

        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    return video_path
