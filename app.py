from flask import Flask, render_template, request, Response,send_from_directory
import cv2
import os
from ultralytics import YOLO
from werkzeug.utils import secure_filename


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load YOLO Model
model = YOLO("yolov8n.pt")

# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# ---------------- IMAGE DETECTION ---------------- #

@app.route('/image', methods=['GET', 'POST'])
def image():
    output_image = None

    if request.method == 'POST':

        file = request.files['file']

        if file:
            filename = secure_filename(file.filename)

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            file.save(filepath)

            # Read image
            img = cv2.imread(filepath)

            # Detection
            results = model(img)

            annotated_frame = results[0].plot()

            output_path = os.path.join(
                app.config['OUTPUT_FOLDER'],
                "detected_" + filename
            )

            cv2.imwrite(output_path, annotated_frame)

            output_image = output_path

    return render_template('image.html', output_image=output_image)


# ---------------- VIDEO DETECTION ---------------- #

@app.route('/video', methods=['GET', 'POST'])
def video():

    output_video = None

    if request.method == 'POST':

        file = request.files['file']

        if file:

            filename = secure_filename(file.filename)

            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'],
                filename
            )

            file.save(filepath)

            cap = cv2.VideoCapture(filepath)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps == 0:
                fps = 25

            output_filename = "detected_" + filename

            output_path = os.path.join(
                app.config['OUTPUT_FOLDER'],
                output_filename
            )

            # Chrome Supported Codec
            fourcc = cv2.VideoWriter_fourcc(*'avc1')

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                results = model(frame)

                annotated_frame = results[0].plot()

                out.write(annotated_frame)

            cap.release()
            out.release()

            # Important
            output_video = 'outputs/' + output_filename

    return render_template(
        'video.html',
        output_video=output_video
    )


# ---------------- WEBCAM DETECTION ---------------- #

def generate_frames():

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = model(frame)

        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame +
               b'\r\n')

    cap.release()


@app.route('/webcam')
def webcam():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )



if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.run(debug=True)
