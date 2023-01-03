from flask import Flask, request
from skimage import io
from webcam_filter import apply_filter

app = Flask(__name__, template_folder="./templates")


@app.route("/")
def stream():
    # TODO: accept stream video
    img = "/Users/wckdouglas/Desktop/test.png"
    frame = io.imread(img)
    return "<br>".join(apply_filter(frame, debug=False))


if __name__ == "__main__":
    app.run(debug=True)
