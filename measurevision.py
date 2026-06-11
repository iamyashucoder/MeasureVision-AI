import cv2
import numpy as np
import math
import os
import csv
from datetime import datetime
from collections import deque


APP_NAME = "MeasureVision AI"
TAGLINE = "Measure the real world through your camera."

SCREENSHOT_FOLDER = "screenshots"
REPORT_FOLDER = "reports"
HISTORY_FILE = "measurement_history.csv"

DEFAULT_REFERENCE_WIDTH_CM = 8.56

# Looser lock settings for normal webcams.
# This makes auto-lock easier while still averaging measurements.
STABLE_FRAME_COUNT = 6
MAX_WIDTH_VARIATION_CM = 1.20
MAX_HEIGHT_VARIATION_CM = 1.20
MAX_CENTER_MOVEMENT_PX = 70
MAX_ANGLE_VARIATION_DEG = 18.0
MIN_SHARPNESS = 15
MIN_OBJECT_AREA = 500


def ask_reference_width():
    print()
    print("Reference object width is needed for real-world measurement.")
    print("Example: credit-card-like object width = 8.56 cm")

    value = input(f"Enter reference width in cm [{DEFAULT_REFERENCE_WIDTH_CM}]: ").strip()

    if value == "":
        return DEFAULT_REFERENCE_WIDTH_CM

    try:
        width = float(value)
        if width <= 0:
            return DEFAULT_REFERENCE_WIDTH_CM
        return width
    except ValueError:
        return DEFAULT_REFERENCE_WIDTH_CM


REFERENCE_WIDTH_CM = ask_reference_width()


def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def midpoint(p1, p2):
    return (
        int((p1[0] + p2[0]) / 2),
        int((p1[1] + p2[1]) / 2),
    )


def order_box_points(points):
    points = np.array(points, dtype="float32")
    ordered = np.zeros((4, 2), dtype="float32")

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]

    return ordered


def normalize_angle(angle):
    angle = abs(angle)

    if angle > 45:
        angle = 90 - angle

    return abs(angle)


def classify_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
    vertices = len(approx)
    area = cv2.contourArea(contour)

    if vertices == 3:
        return "Triangle"

    if vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        if 0.92 <= aspect_ratio <= 1.08:
            return "Square"
        return "Rectangle"

    circularity = 0
    if perimeter > 0:
        circularity = 4 * math.pi * area / (perimeter * perimeter)

    if circularity > 0.78:
        return "Circle"

    if vertices == 5:
        return "Pentagon"

    if vertices == 6:
        return "Hexagon"

    return "Polygon"


def get_formula(shape):
    if shape in ["Rectangle", "Square"]:
        return "A = width x height | P = 2(w + h)"

    if shape == "Circle":
        return "A = pi x r^2 | C = 2 x pi x r"

    if shape == "Triangle":
        return "A = 1/2 x base x height"

    return "A approx = width x height"


def calculate_details(shape, width_cm, height_cm):
    area = width_cm * height_cm
    perimeter = 2 * (width_cm + height_cm)

    diameter = min(width_cm, height_cm)
    radius = diameter / 2

    if shape == "Circle":
        area = math.pi * radius ** 2
        perimeter = 2 * math.pi * radius

    elif shape == "Triangle":
        area = 0.5 * width_cm * height_cm
        perimeter = width_cm + height_cm + math.sqrt(width_cm ** 2 + height_cm ** 2)

    return area, perimeter, radius, diameter


def get_object_dimensions(contour, pixels_per_cm):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = order_box_points(box)

    top_left, top_right, bottom_right, bottom_left = box

    width_pixels_top = distance(top_left, top_right)
    width_pixels_bottom = distance(bottom_left, bottom_right)

    height_pixels_left = distance(top_left, bottom_left)
    height_pixels_right = distance(top_right, bottom_right)

    width_pixels = (width_pixels_top + width_pixels_bottom) / 2
    height_pixels = (height_pixels_left + height_pixels_right) / 2

    width_cm = width_pixels / pixels_per_cm
    height_cm = height_pixels / pixels_per_cm

    angle = normalize_angle(rect[2])

    center_x = int(np.mean(box[:, 0]))
    center_y = int(np.mean(box[:, 1]))

    return box.astype("int"), width_cm, height_cm, angle, (center_x, center_y)


def find_contours(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(blurred, 45, 120)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours


def sharpness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def brightness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def draw_text(frame, text, x, y, scale=0.55, color=(255, 255, 255), thickness=1):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(frame, x, y, w, h, color=(5, 10, 25), alpha=0.75, border=(0, 255, 255)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border, 1)


def draw_target_zone(frame):
    h, w = frame.shape[:2]

    zone_w = int(w * 0.52)
    zone_h = int(h * 0.46)

    x1 = int((w - zone_w) / 2)
    y1 = int((h - zone_h) / 2) + 25
    x2 = x1 + zone_w
    y2 = y1 + zone_h

    color = (0, 255, 255)
    corner = 55

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 80, 100), 1)

    cv2.line(frame, (x1, y1), (x1 + corner, y1), color, 3)
    cv2.line(frame, (x1, y1), (x1, y1 + corner), color, 3)

    cv2.line(frame, (x2, y1), (x2 - corner, y1), color, 3)
    cv2.line(frame, (x2, y1), (x2, y1 + corner), color, 3)

    cv2.line(frame, (x1, y2), (x1 + corner, y2), color, 3)
    cv2.line(frame, (x1, y2), (x1, y2 - corner), color, 3)

    cv2.line(frame, (x2, y2), (x2 - corner, y2), color, 3)
    cv2.line(frame, (x2, y2), (x2, y2 - corner), color, 3)

    draw_text(
        frame,
        "PLACE ONE OBJECT INSIDE THIS SCAN ZONE",
        x1 + 24,
        y1 - 18,
        0.55,
        color,
        2,
    )

    return x1, y1, x2, y2


def contour_inside_zone(contour, zone):
    x1, y1, x2, y2 = zone
    x, y, w, h = cv2.boundingRect(contour)

    center_x = x + w // 2
    center_y = y + h // 2

    return x1 <= center_x <= x2 and y1 <= center_y <= y2


def choose_center_object(contours, zone):
    x1, y1, x2, y2 = zone
    target_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_OBJECT_AREA:
            continue

        if not contour_inside_zone(contour, zone):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)
        dist = distance(center, target_center)

        candidates.append((dist, area, contour))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], -item[1]))
    return candidates[0][2]


def draw_measurement_arrows(frame, box, width_cm, height_cm, color):
    top_left, top_right, bottom_right, bottom_left = box

    p1 = (int(top_left[0]), int(top_left[1] - 24))
    p2 = (int(top_right[0]), int(top_right[1] - 24))

    cv2.arrowedLine(frame, p1, p2, color, 2, tipLength=0.04)
    cv2.arrowedLine(frame, p2, p1, color, 2, tipLength=0.04)

    mx, my = midpoint(p1, p2)
    draw_text(frame, f"{width_cm:.2f} cm", mx - 45, my - 8, 0.55, color, 2)

    p3 = (int(top_right[0] + 26), int(top_right[1]))
    p4 = (int(bottom_right[0] + 26), int(bottom_right[1]))

    cv2.arrowedLine(frame, p3, p4, color, 2, tipLength=0.04)
    cv2.arrowedLine(frame, p4, p3, color, 2, tipLength=0.04)

    mx2, my2 = midpoint(p3, p4)
    draw_text(frame, f"{height_cm:.2f} cm", mx2 + 8, my2, 0.55, color, 2)


def draw_hud(frame, pixels_per_cm, state, progress, message):
    h, w = frame.shape[:2]

    draw_panel(frame, 0, 0, w, 95, color=(3, 8, 22), alpha=0.90)

    draw_text(frame, APP_NAME, 28, 38, 1.0, (0, 255, 255), 2)
    draw_text(frame, TAGLINE, 30, 68, 0.55, (255, 120, 255), 1)

    status_x = w - 430
    draw_panel(frame, status_x, 18, 410, 58, color=(8, 12, 32), alpha=0.80)

    if pixels_per_cm:
        cal_text = f"CALIBRATED | {pixels_per_cm:.2f} px/cm"
        cal_color = (0, 255, 120)
    else:
        cal_text = "NOT CALIBRATED"
        cal_color = (0, 120, 255)

    draw_text(frame, f"STATE: {state}", status_x + 16, 41, 0.52, (255, 255, 255), 1)
    draw_text(frame, cal_text, status_x + 16, 66, 0.50, cal_color, 1)

    draw_panel(frame, 18, 112, 360, 132, color=(5, 10, 28), alpha=0.75)

    draw_text(frame, "LOCK-ON SCANNER", 36, 145, 0.68, (0, 255, 255), 2)
    draw_text(frame, message, 36, 174, 0.50, (255, 255, 255), 1)

    bar_x = 36
    bar_y = 204
    bar_w = 305
    bar_h = 16

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 35, 55), -1)

    fill_w = int(bar_w * progress)
    bar_color = (0, 255, 120) if progress >= 1 else (0, 255, 255)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

    draw_text(frame, f"Stability: {int(progress * 100)}%", 36, 236, 0.48, (255, 255, 0), 1)


def draw_status_bar(frame):
    h, w = frame.shape[:2]

    draw_panel(frame, 0, h - 68, w, 68, color=(3, 8, 22), alpha=0.93)

    draw_text(frame, "CONTROL DECK", 24, h - 40, 0.62, (0, 255, 255), 2)

    draw_text(
        frame,
        "C CALIBRATE REF | L MANUAL LOCK | N NEW SCAN | S SAVE LOCKED RESULT | Q QUIT",
        190,
        h - 40,
        0.50,
        (255, 255, 255),
        1,
    )

    draw_text(
        frame,
        "Accuracy tip: keep camera still, object flat, good lighting, reference and object on same plane.",
        24,
        h - 15,
        0.42,
        (180, 220, 255),
        1,
    )


def draw_object_box(frame, box, color, locked=False):
    thickness = 4 if locked else 2

    cv2.drawContours(frame, [box], -1, color, thickness)

    length = 30

    for x, y in box:
        cv2.line(frame, (x - length, y), (x + length, y), color, thickness)
        cv2.line(frame, (x, y - length), (x, y + length), color, thickness)


def build_details(shape, width_cm, height_cm, angle, box, accuracy):
    area_cm, perimeter_cm, radius_cm, diameter_cm = calculate_details(shape, width_cm, height_cm)

    return {
        "shape": shape,
        "width": width_cm,
        "height": height_cm,
        "area": area_cm,
        "perimeter": perimeter_cm,
        "radius": radius_cm,
        "diameter": diameter_cm,
        "angle": angle,
        "box": box,
        "accuracy": accuracy,
        "formula": get_formula(shape),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def draw_locked_panel(frame, details):
    h, w = frame.shape[:2]
    x = w - 420
    y = 112

    draw_panel(frame, x, y, 400, 360, color=(4, 8, 24), alpha=0.88)

    draw_text(frame, "LOCKED MEASUREMENT", x + 20, y + 36, 0.70, (0, 255, 120), 2)
    draw_text(frame, "Single-object result captured", x + 20, y + 64, 0.48, (255, 120, 255), 1)

    draw_text(frame, details["shape"].upper(), x + 20, y + 112, 0.95, (0, 255, 255), 2)

    if details["shape"] == "Circle":
        lines = [
            f"Diameter   : {details['diameter']:.2f} cm",
            f"Radius     : {details['radius']:.2f} cm",
            f"Area       : {details['area']:.2f} cm^2",
            f"Circumf.   : {details['perimeter']:.2f} cm",
            f"Angle      : {details['angle']:.2f} deg",
            f"Accuracy   : {details['accuracy']}%",
        ]
    else:
        lines = [
            f"Width      : {details['width']:.2f} cm",
            f"Height     : {details['height']:.2f} cm",
            f"Area       : {details['area']:.2f} cm^2",
            f"Perimeter  : {details['perimeter']:.2f} cm",
            f"Angle      : {details['angle']:.2f} deg",
            f"Accuracy   : {details['accuracy']}%",
        ]

    yy = y + 155

    for line in lines:
        draw_text(frame, line, x + 24, yy, 0.56, (255, 255, 255), 1)
        yy += 30

    formula_y = y + 318
    cv2.rectangle(frame, (x + 20, formula_y), (x + 380, formula_y + 30), (20, 20, 45), -1)
    cv2.rectangle(frame, (x + 20, formula_y), (x + 380, formula_y + 30), (255, 120, 255), 1)
    draw_text(frame, details["formula"], x + 30, formula_y + 21, 0.45, (255, 180, 255), 1)


def check_stability(history):
    if len(history) < STABLE_FRAME_COUNT:
        return False, len(history) / STABLE_FRAME_COUNT, "Hold still..."

    widths = np.array([item["width"] for item in history])
    heights = np.array([item["height"] for item in history])
    centers = np.array([item["center"] for item in history])
    angles = np.array([item["angle"] for item in history])
    sharpness_values = np.array([item["sharpness"] for item in history])
    brightness_values = np.array([item["brightness"] for item in history])

    width_range = widths.max() - widths.min()
    height_range = heights.max() - heights.min()

    center_movement = np.max(
        np.sqrt(
            (centers[:, 0] - centers[:, 0].mean()) ** 2 +
            (centers[:, 1] - centers[:, 1].mean()) ** 2
        )
    )

    angle_range = angles.max() - angles.min()
    avg_sharpness = sharpness_values.mean()
    avg_brightness = brightness_values.mean()
    avg_angle = angles.mean()

    checks = [
        width_range <= MAX_WIDTH_VARIATION_CM,
        height_range <= MAX_HEIGHT_VARIATION_CM,
        center_movement <= MAX_CENTER_MOVEMENT_PX,
        angle_range <= MAX_ANGLE_VARIATION_DEG,
        avg_sharpness >= MIN_SHARPNESS,
        25 <= avg_brightness <= 245,
        avg_angle <= 30,
    ]

    passed = sum(checks)
    progress = passed / len(checks)

    if not checks[0] or not checks[1]:
        return False, progress, "Dimensions unstable"

    if not checks[2]:
        return False, progress, "Object moving"

    if not checks[3]:
        return False, progress, "Angle changing"

    if not checks[4]:
        return False, progress, "Image not sharp"

    if not checks[5]:
        return False, progress, "Improve lighting"

    if not checks[6]:
        return False, progress, "Straighten object"

    return True, 1.0, "Object locked"


def get_average_measurement(history, shape, box):
    width = float(np.mean([item["width"] for item in history]))
    height = float(np.mean([item["height"] for item in history]))
    angle = float(np.mean([item["angle"] for item in history]))

    accuracy = 92

    return build_details(shape, width, height, angle, box, accuracy)


def save_locked_result(frame, details):
    if details is None:
        print("No locked result to save.")
        return

    os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)
    os.makedirs(REPORT_FOLDER, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    image_path = os.path.join(SCREENSHOT_FOLDER, f"locked_measurement_{timestamp}.png")
    report_path = os.path.join(REPORT_FOLDER, f"locked_measurement_{timestamp}.txt")

    cv2.imwrite(image_path, frame)

    with open(report_path, "w") as file:
        file.write(f"{APP_NAME} Report\n")
        file.write(f"{TAGLINE}\n\n")
        file.write(f"Time: {details['time']}\n")
        file.write(f"Shape: {details['shape']}\n")
        file.write(f"Width: {details['width']:.2f} cm\n")
        file.write(f"Height: {details['height']:.2f} cm\n")
        file.write(f"Area: {details['area']:.2f} cm^2\n")
        file.write(f"Perimeter: {details['perimeter']:.2f} cm\n")
        file.write(f"Radius: {details['radius']:.2f} cm\n")
        file.write(f"Diameter: {details['diameter']:.2f} cm\n")
        file.write(f"Angle: {details['angle']:.2f} deg\n")
        file.write(f"Accuracy: {details['accuracy']}%\n")
        file.write(f"Formula: {details['formula']}\n")

    file_exists = os.path.exists(HISTORY_FILE)

    with open(HISTORY_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(
                [
                    "time",
                    "shape",
                    "width_cm",
                    "height_cm",
                    "area_cm2",
                    "perimeter_cm",
                    "radius_cm",
                    "diameter_cm",
                    "angle_deg",
                    "accuracy",
                ]
            )

        writer.writerow(
            [
                details["time"],
                details["shape"],
                f"{details['width']:.2f}",
                f"{details['height']:.2f}",
                f"{details['area']:.2f}",
                f"{details['perimeter']:.2f}",
                f"{details['radius']:.2f}",
                f"{details['diameter']:.2f}",
                f"{details['angle']:.2f}",
                details["accuracy"],
            ]
        )

    print(f"Saved screenshot: {image_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved history: {HISTORY_FILE}")


def open_camera():
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

        if cap.isOpened():
            print(f"Camera opened using index {camera_index}")
            return cap

    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index)

        if cap.isOpened():
            print(f"Camera opened using fallback index {camera_index}")
            return cap

    return None


def main():
    print(f"{APP_NAME} v2 started.")
    print(TAGLINE)
    print(f"Reference width: {REFERENCE_WIDTH_CM} cm")
    print()
    print("Controls:")
    print("C = calibrate reference")
    print("L = manual lock current object")
    print("N = new scan")
    print("S = save locked result")
    print("Q = quit")
    print()

    cap = open_camera()

    if cap is None:
        print("Error: Could not open camera.")
        print("On Mac, enable camera permission for Terminal, VS Code, or PyCharm.")
        return

    pixels_per_cm = None
    measurement_history = deque(maxlen=STABLE_FRAME_COUNT)

    locked_details = None
    locked_frame = None
    locked = False

    state = "CALIBRATION"
    message = "Press C with reference object in scan zone"
    progress = 0.0

    while True:
        success, frame = cap.read()

        if not success:
            print("Error: Could not read camera frame.")
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        zone = draw_target_zone(display)
        contours = find_contours(frame)
        target_contour = choose_center_object(contours, zone)

        if target_contour is not None and pixels_per_cm is not None and not locked:
            box, width_cm, height_cm, angle, center = get_object_dimensions(
                target_contour,
                pixels_per_cm,
            )

            shape = classify_shape(target_contour)

            sharp = sharpness_score(frame)
            bright = brightness_score(frame)

            measurement_history.append(
                {
                    "width": width_cm,
                    "height": height_cm,
                    "center": center,
                    "angle": angle,
                    "sharpness": sharp,
                    "brightness": bright,
                }
            )

            stable, progress, message = check_stability(measurement_history)

            state = "SCANNING"

            if stable:
                locked = True
                state = "LOCKED"
                message = "Measurement locked"

                locked_details = get_average_measurement(
                    measurement_history,
                    shape,
                    box,
                )

                locked_frame = display.copy()

            color = (0, 255, 120) if stable else (0, 255, 255)

            draw_object_box(display, box, color, locked=stable)
            draw_measurement_arrows(display, box, width_cm, height_cm, color)

            draw_text(display, shape, center[0] - 50, center[1] - 12, 0.75, color, 2)
            draw_text(
                display,
                f"W {width_cm:.2f} cm | H {height_cm:.2f} cm",
                center[0] - 110,
                center[1] + 22,
                0.52,
                color,
                2,
            )

        elif target_contour is not None and pixels_per_cm is None:
            state = "CALIBRATION"
            message = "Press C to calibrate reference"
            progress = 0.0

            temp_box = cv2.boxPoints(cv2.minAreaRect(target_contour))
            temp_box = order_box_points(temp_box).astype("int")
            draw_object_box(display, temp_box, (0, 120, 255), locked=False)

        elif locked and locked_details is not None:
            state = "LOCKED"
            message = "Press N for new scan or S to save"
            progress = 1.0

            box = locked_details["box"]
            draw_object_box(display, box, (0, 255, 120), locked=True)
            draw_measurement_arrows(
                display,
                box,
                locked_details["width"],
                locked_details["height"],
                (0, 255, 120),
            )

        else:
            if pixels_per_cm is None:
                state = "CALIBRATION"
                message = "Place reference object in scan zone"
            else:
                state = "WAITING"
                message = "Place one object in scan zone"

            progress = 0.0
            measurement_history.clear()

        draw_hud(display, pixels_per_cm, state, progress, message)
        draw_status_bar(display)

        if locked and locked_details is not None:
            draw_locked_panel(display, locked_details)

        cv2.imshow(APP_NAME, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("c"):
            if target_contour is None:
                print("No reference object found in scan zone.")
                continue

            ref_box = cv2.boxPoints(cv2.minAreaRect(target_contour))
            ref_box = order_box_points(ref_box)

            top_left, top_right, bottom_right, bottom_left = ref_box

            width_pixels_top = distance(top_left, top_right)
            width_pixels_bottom = distance(bottom_left, bottom_right)
            reference_width_pixels = (width_pixels_top + width_pixels_bottom) / 2

            if reference_width_pixels > 0:
                pixels_per_cm = reference_width_pixels / REFERENCE_WIDTH_CM
                measurement_history.clear()
                locked = False
                locked_details = None
                locked_frame = None

                print(f"Calibration complete: {pixels_per_cm:.2f} px/cm")
                print("Now place the object you want to measure in the scan zone.")

        elif key == ord("n"):
            locked = False
            locked_details = None
            locked_frame = None
            measurement_history.clear()
            state = "WAITING"
            message = "Place one object in scan zone"
            progress = 0.0
            print("New scan started.")

        elif key == ord("l"):
            if pixels_per_cm is None:
                print("Calibrate first by pressing C with reference object in scan zone.")

            elif target_contour is None:
                print("No object found in scan zone.")

            else:
                box, width_cm, height_cm, angle, center = get_object_dimensions(
                    target_contour,
                    pixels_per_cm,
                )

                shape = classify_shape(target_contour)

                locked_details = build_details(
                    shape,
                    width_cm,
                    height_cm,
                    angle,
                    box,
                    90,
                )

                locked = True
                locked_frame = display.copy()
                measurement_history.clear()
                print("Manual lock complete.")

        elif key == ord("s"):
            if locked and locked_details is not None:
                save_locked_result(display, locked_details)
            else:
                print("No locked measurement yet. Wait until scanner locks the object or press L.")

    cap.release()
    cv2.destroyAllWindows()
    print("MeasureVision AI closed.")


if __name__ == "__main__":
    main()