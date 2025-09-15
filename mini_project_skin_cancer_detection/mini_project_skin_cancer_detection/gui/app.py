from flask import Flask, render_template, request, session, redirect, flash, url_for, Response
import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import visualize
import os



app = Flask(__name__)
app.secret_key = "skin"
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


detector = None
cap = None
COUNTER, FPS = 0, 0
START_TIME = time.time()
detection_result_list = []






@app.route('/')
def starter():
    return render_template("new.html")



@app.route("/upload_image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        # Check if a file was uploaded
        if "image" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        # Save the file
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Process the image
            result_path = run_image_detection(file_path)

            # Render the result
            return render_template("result.html", result_image=result_path)

    return render_template("upload_image.html")


def run_image_detection(image_path: str) -> str:
    """Run inference on an uploaded image and save the result.

    Args:
        image_path: Path to the uploaded image.

    Returns:
        Path to the saved result image.
    """
    model_path = "finalskin.tflite"  # Path to your TFLite model
    max_results = 5
    score_threshold = 0.5

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        flash(f"ERROR: Unable to read image at {image_path}. Check the file path.")
        return None

    # Resize the image for consistency
    image = cv2.resize(image, (640, 480))

    # Convert the image from BGR to RGB as required by Mediapipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Initialize the object detection model
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        max_results=max_results,
        score_threshold=score_threshold,
    )
    detector = vision.ObjectDetector.create_from_options(options)

    # Run object detection
    detection_result = detector.detect(mp_image)

    # Visualize the results on the image
    result_image = visualize(image, detection_result)

    # Save the result
    result_path = os.path.join(app.config["UPLOAD_FOLDER"], "result.jpg")
    cv2.imwrite(result_path, result_image)

    # Clean up
    detector.close()
    return result_path




def initialize_detector(model_path, max_results=5, score_threshold=0.5):
    """Initialize the Mediapipe object detector."""
    global detector

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        max_results=max_results,
        score_threshold=score_threshold,
        result_callback=save_result,
    )
    detector = vision.ObjectDetector.create_from_options(options)

def save_result(result, unused_output_image, timestamp_ms):
    """Callback to save detection results and calculate FPS."""
    global FPS, COUNTER, START_TIME, detection_result_list

    # Calculate FPS
    if COUNTER % 10 == 0:
        FPS = 10 / (time.time() - START_TIME)
        START_TIME = time.time()

    detection_result_list.append(result)
    COUNTER += 1

def generate_frames():
    """Capture video frames and perform real-time object detection."""
    global cap, detector, detection_result_list

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip and process the frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Perform detection
        detector.detect_async(mp_frame, time.time_ns() // 1_000_000)

        # Display FPS
        cv2.putText(
            frame, f"FPS: {FPS:.1f}", (24, 50),
            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA
        )

        # Draw detections
        if detection_result_list:
            frame = visualize(frame, detection_result_list[0])
            detection_result_list.clear()

        # Encode the frame and yield it for the response
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')




if __name__ == '__main__':
    initialize_detector('finalskin.tflite')
    app.run(host='0.0.0.0', port=5000, debug=True)