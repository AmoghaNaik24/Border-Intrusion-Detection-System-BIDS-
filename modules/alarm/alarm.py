import time
import cv2
import os
from playsound import playsound
from core.logger import logger

ALERT_DIR = "alerts"
ALARM_PATH = "assets/alarm.wav"

os.makedirs(ALERT_DIR, exist_ok=True)


def trigger_alarm(frame):
    logger.critical("🚨🚨 ALARM TRIGGERED 🚨🚨")

    try:
        playsound(ALARM_PATH, block=False)
    except:
        logger.warning("Alarm sound failed")

    ts = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"{ALERT_DIR}/intrusion_{ts}.jpg", frame)