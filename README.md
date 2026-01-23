

🚨 Project Showcase | Smart Border Security System 🚨


🔍 Problem Statement

Traditional border surveillance systems rely heavily on manual monitoring and static sensors, which makes them prone to delays, human error, and false alarms (animals, empty backgrounds, or environmental changes).
Our goal was to design an intelligent, automated system that can detect real intrusions accurately, reduce false positives, and collect proper evidence for further investigation.


---

🛠️ Solution Overview (4‑Phase Architecture)

The system works in four sequential phases, ensuring efficiency and accuracy:

Phase 1: Motion Detection

Detects movement using background subtraction

Extracts Region of Interest (ROI) only when motion is present

Prevents unnecessary computation when the area is empty

Phase 2: Object Detection

Uses YOLOv8n for real‑time object detection

Classifies objects into:

Intrusion threats (person, vehicles, aircraft, etc.)

Animals

Other objects

Only genuine threats move to the next phase

Phase 3: Face Detection

Activated only when a person is detected

Detects faces from intrusion frames

Triggers an alarm during confirmed intrusion

Phase 4: Face Recognition

Compares detected faces with a preloaded criminal/authorized database

Classifies individuals as Known or Unknown

For unknown individuals:

Captures image

Stores video & snapshot as evidence

Logs intrusion details for future investigation


---

🎯 Key Outcomes

✅ Empty background → No threat

🚶 Person detected → Intrusion alert

🧑 Known face → Alarm+ Identified & logged

❓ Unknown face → Alarm + evidence stored
-----
🚀 Tech Stack
Python | OpenCV | YOLOv8 | DeepFace

