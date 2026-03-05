"""
Run this FIRST to diagnose webcam + face detection issues.
Usage: python debug_camera.py
(No streamlit needed — opens a plain OpenCV window)
"""
import cv2
import sys

print("=== ExamGuard Camera Debugger ===\n")

# ── Step 1: Can we open the camera? ──────────────────────────────────────────
print("Step 1: Opening camera index 0...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ FAILED: Cannot open camera index 0.")
    print("   Try index 1 or 2:")
    for i in [1, 2]:
        c = cv2.VideoCapture(i)
        if c.isOpened():
            print(f"   ✅ Camera found at index {i} — change VideoCapture(0) to VideoCapture({i})")
            c.release()
    sys.exit(1)

print("✅ Camera opened successfully.")

# ── Step 2: Can we read a frame? ─────────────────────────────────────────────
print("\nStep 2: Reading a frame...")
ret, frame = cap.read()
if not ret or frame is None:
    print("❌ FAILED: Camera opened but could not read a frame.")
    print("   This usually means macOS camera permission is denied.")
    print("   → System Settings → Privacy & Security → Camera → enable Terminal")
    cap.release()
    sys.exit(1)

print(f"✅ Frame captured: {frame.shape[1]}x{frame.shape[0]} px")

# ── Step 3: Test face detection ───────────────────────────────────────────────
print("\nStep 3: Testing face detection...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Try progressively more sensitive settings
configs = [
    dict(scaleFactor=1.05, minNeighbors=1, minSize=(20, 20)),
    dict(scaleFactor=1.1,  minNeighbors=2, minSize=(30, 30)),
    dict(scaleFactor=1.15, minNeighbors=3, minSize=(40, 40)),
]

detected = False
for cfg in configs:
    faces = face_cascade.detectMultiScale(gray, **cfg)
    if len(faces) > 0:
        print(f"✅ Face(s) detected: {len(faces)} face(s) with settings {cfg}")
        detected = True
        break
    else:
        print(f"   No face with {cfg}")

if not detected:
    print("\n⚠  No face detected in the still frame.")
    print("   This might be:")
    print("   • Bad lighting — try facing a window or lamp")
    print("   • Camera returning a black/green frame (permission issue)")
    print("   • Face too small or at an angle")
    print("\n   Saving debug frame as 'debug_frame.jpg' — open it to see what the camera sees.")
    cv2.imwrite("debug_frame.jpg", frame)
    cv2.imwrite("debug_frame_gray.jpg", gray)
    print("   Also saved grayscale as 'debug_frame_gray.jpg'")

# ── Step 4: Live preview ──────────────────────────────────────────────────────
print("\nStep 4: Opening LIVE preview window (press Q to quit)...")
print("   Watch the window — if it's black/green, it's a permissions issue.\n")

face_cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Equalise histogram to handle poor lighting
    gray_eq = cv2.equalizeHist(gray)

    faces = face_cascade2.detectMultiScale(gray_eq, scaleFactor=1.2, minNeighbors=8, minSize=(100, 100))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "FACE DETECTED", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    status = f"Faces: {len(faces)}  |  Press Q to quit"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.imshow("ExamGuard Debug", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nDebug session ended.")