"""
VitalTwin AI - Automated Video Demonstration Generator
Captures interactive browser frames via Selenium and compiles a complete,
smooth demonstration MP4 video synced with the challenge presentation script.
"""

import os
import sys
import time
from typing import List, Tuple
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def setup_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1600,900")
    driver = webdriver.Chrome(options=options)
    return driver


def overlay_subtitle(img: np.ndarray, title: str, subtitle: str, time_str: str = "") -> np.ndarray:
    """Overlays a professional healthcare presentation banner with speech script subtitles."""
    h, w, _ = img.shape
    overlay = img.copy()

    # Bottom banner for speech narration
    banner_h = 105
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (15, 23, 42), -1)  # Navy #0F172A
    cv2.addWeighted(overlay, 0.92, img, 0.08, 0, img)

    # Accent line
    cv2.line(img, (0, h - banner_h), (w, h - banner_h), (13, 148, 136), 3)  # Teal

    # Narration Title
    cv2.putText(
        img,
        title,
        (30, h - banner_h + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 184, 166),  # Teal
        2,
        cv2.LINE_AA,
    )

    # Subtitle / Spoken Script Line
    cv2.putText(
        img,
        subtitle,
        (30, h - banner_h + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (241, 245, 249),  # White
        1,
        cv2.LINE_AA,
    )

    if time_str:
        cv2.putText(
            img,
            time_str,
            (w - 180, h - banner_h + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (148, 163, 184),
            1,
            cv2.LINE_AA,
        )

    return img


def record_demo(output_video_path: str = "docs/vitaltwin_demo_video.mp4"):
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    driver = setup_driver()

    try:
        driver.get("http://127.0.0.1:8501")
        time.sleep(3)

        scenes = [
            (
                "1. Welcome & Project Introduction",
                "VitalTwin AI: An Explainable Healthcare Digital Twin for Predictive Healthcare (Happiest Health 2026).",
                5,
                None,
            ),
            (
                "2. Virtual Patient Profile (PT-1001)",
                "Combining historical EHR baselines with continuous CGM, activity, and heart rate telemetry.",
                5,
                lambda d: None,
            ),
            (
                "3. Live Sensor Simulation & Clock Advance",
                "Simulating 5-minute continuous glucose monitoring steps with causal physiological responses.",
                6,
                lambda d: click_button_text(d, "Step +5 Min"),
            ),
            (
                "4. Fast-Forward Observation (+15 Min)",
                "Updating digital twin state in real-time. Forecast model computes 2-hour excursion risk.",
                6,
                lambda d: click_button_text(d, "Step +15 Min"),
            ),
            (
                "5. Postprandial Meal Ingestion Event",
                "Logging a 50g carbohydrate meal. Biphasic gut absorption curve raises predicted risk.",
                7,
                lambda d: click_button_text(d, "Ingest Meal Now"),
            ),
            (
                "6. Continuous Glucose & Activity Dynamics",
                "Dual-axis interactive charts displaying target band (70-180 mg/dL), meals, and heart rate.",
                6,
                lambda d: scroll_page(d, 350),
            ),
            (
                "7. Explainable AI & Risk Drivers",
                "Interpretable Random Forest feature importance highlighting glycemic trend and carb influx.",
                6,
                lambda d: scroll_page(d, 550),
            ),
            (
                "8. Interactive What-If Scenario Sandbox",
                "Simulating hypothetical interventions (post-meal walking) without altering baseline records.",
                7,
                lambda d: scroll_page(d, 650),
            ),
            (
                "9. Model Transparency & Evaluation Tab",
                "Held-out test evaluation: Accuracy 94.1%, Precision 95.8%, Recall 93.4%, ROC-AUC 0.985.",
                7,
                lambda d: click_tab_text(d, "Model Evaluation & Metrics"),
            ),
            (
                "10. Standardized CSV Data Export",
                "Downloadable synthetic sensor streams and prediction logs for open auditability.",
                6,
                lambda d: click_tab_text(d, "Data Export (CSV)"),
            ),
            (
                "11. Conclusion & Non-Clinical Research Disclaimer",
                "Research proof of concept. Synthetic data only. Not intended for clinical medical treatment.",
                5,
                lambda d: scroll_page(d, 0),
            ),
        ]

        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None

        temp_img_path = "data/temp_frame.png"

        for idx, (title, subtitle, duration_secs, action) in enumerate(scenes):
            if action:
                try:
                    action(driver)
                    time.sleep(1.0)
                except Exception as e:
                    print(f"Scene action note: {e}")

            # Capture screenshot
            driver.save_screenshot(temp_img_path)
            frame_raw = cv2.imread(temp_img_path)
            if frame_raw is None:
                continue

            # Overlay subtitle banner
            frame_annotated = overlay_subtitle(
                frame_raw,
                f"VitalTwin AI — {title}",
                subtitle,
                f"Part {idx+1}/{len(scenes)}",
            )

            h, w, _ = frame_annotated.shape
            if writer is None:
                writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

            total_frames = duration_secs * fps
            for _ in range(total_frames):
                writer.write(frame_annotated)

        if writer:
            writer.release()

        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

        print(f"Video successfully created: {output_video_path} (Size: {os.path.getsize(output_video_path)} bytes)")
        return output_video_path

    finally:
        driver.quit()


def click_button_text(driver, text_snippet: str):
    buttons = driver.find_elements(By.TAG_NAME, "button")
    for b in buttons:
        if text_snippet.lower() in b.text.lower():
            b.click()
            return True
    return False


def click_tab_text(driver, text_snippet: str):
    tabs = driver.find_elements(By.CSS_SELECTOR, "button[data-baseweb='tab']")
    for t in tabs:
        if text_snippet.lower() in t.text.lower():
            t.click()
            return True
    return False


def scroll_page(driver, y_offset: int):
    driver.execute_script(f"window.scrollTo(0, {y_offset});")


if __name__ == "__main__":
    record_demo()
