from time import sleep

import cv2
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
    url_for,
)
from skimage import io
from webcam_filter import apply_filter

app = Flask(__name__, template_folder="./templates")


def stream_template(template_name, **context):
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.enable_buffering(5)
    return rv


def stream_video():
    # define a video capture object
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        yield frame


def stream_emoji():
    for frame in stream_video():
        yield "\n".join(apply_filter(frame, debug=False))


@app.route("/_stream")
def stream():
    # TODO: accept stream video
    # frame = io.imread("/Users/wckdouglas/Desktop/test.png")
    # return render_template("index.html", processed_string="<br>".join(apply_filter(frame, debug=False)))
    img = next(stream_emoji())
    return jsonify(img=img)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
