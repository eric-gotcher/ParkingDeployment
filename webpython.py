from flask import Flask, Response, render_template, request
from queue import Queue
import cv2
from ultralytics import YOLO
from threading import Thread

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
queue = Queue()

DEFAULT_VIDEO_PATH = "https://firebasestorage.googleapis.com/v0/b/parking-with-devops.appspot.com/o/istockphoto-1046782266-640-adpp-is_dNpvycW4.mp4?alt=media&token=52ae3e31-e5a5-4beb-8577-2d77f2f71474"
MODEL_PATH = "https://firebasestorage.googleapis.com/v0/b/parking-with-devops.appspot.com/o/best.pt?alt=media&token=c8a71ce0-7731-400d-93eb-b4b36ce8c709"
classNames = ["Empty", "Space Taken"]
process_thread = None  # Initialize process_thread

# Define video processing function
# Define video processing function
def process_video(selected_video_path):
    global process_thread
    try:
        cap = cv2.VideoCapture(selected_video_path)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return
        model = YOLO(MODEL_PATH)

        while True:
            if process_thread and process_thread.stopped():
                break  # Check if the thread should be stopped
            success, frame = cap.read()
            if not success:
                break

            results = model(frame)
            processed_frame = process_frame(frame, results, classNames)
            queue.put(processed_frame)

        cap.release()
    except Exception as e:
        print(f"An error occurred: {e}")


# Define frame processing functions
def process_frame(frame, results, classNames):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            draw_box(frame, box, classNames)
    return frame

def draw_box(frame, box, classNames):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls = int(box.cls[0])
    color = (0,0,255) if classNames[cls] == 'Space Taken' else (0,255,0)
    if box.conf[0] > 0.43 and x2-x1 < 100:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

# Define frame generation function
def generate():
    while True:
        frame = queue.get()
        ret, buffer = cv2.imencode('.jpg', frame)
        frameData = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frameData + b'\r\n')
        queue.task_done()

# Define Flask route for video feed
@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Define Flask route to render the HTML file
@app.route("/")
def parking():
    return render_template('lots.html')  

class StoppableThread(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = False

    def stop(self):
        self._stop_event = True

    def stopped(self):
        return self._stop_event

@app.route("/select_lot", methods=["GET", "POST"])
def select_lot():
    global process_thread
    if request.method == "POST":
        lot_number = request.form.get('lot')
        if lot_number is None:
            return "Invalid lot number", 400  # Return an error response for invalid lot number
        
        # Check if the lot number is valid and select the corresponding video path
        if lot_number == "1":
            selected_video_path = "https://firebasestorage.googleapis.com/v0/b/parking-with-devops.appspot.com/o/gettyimages-1533928757-640_adpp.mp4?alt=media&token=d2fb383f-8504-4d46-a977-2b1e02a6fbfa"
        elif lot_number == "2":
            selected_video_path = "https://firebasestorage.googleapis.com/v0/b/parking-with-devops.appspot.com/o/istockphoto-1046782266-640-adpp-is_dNpvycW4.mp4?alt=media&token=52ae3e31-e5a5-4beb-8577-2d77f2f71474"
        elif lot_number == "3":
            selected_video_path = "https://firebasestorage.googleapis.com/v0/b/parking-with-devops.appspot.com/o/istockphoto-1370353417-640_adpp_is.mp4?alt=media&token=d4cd845b-8b18-4bcf-8be1-b1bc90d26138"
        elif lot_number == "4":
            selected_video_path = "https://firebasestorage.googleapis.com/v0/b/parking-with-devops.appspot.com/o/istockphoto-845199510-640_adpp_is.mp4?alt=media&token=fdbbd0dc-db3a-46e4-b6d3-afa2583cdfba"
        else:
            selected_video_path = DEFAULT_VIDEO_PATH

        print("Selected lot:", lot_number)
        print("Selected video:", selected_video_path)

        # Stop the existing video processing thread if it exists
        if process_thread and process_thread.is_alive():
            process_thread.stop()
            process_thread.join()

        # Start a new video processing thread
        process_thread = StoppableThread(target=process_video, args=(selected_video_path,))
        process_thread.daemon = True
        process_thread.start()

        return "Video selected successfully"
    else:
        return "Method Not Allowed", 405


# Main thread
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
