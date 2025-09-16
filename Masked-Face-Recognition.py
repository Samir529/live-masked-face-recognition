import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer,  RTCConfiguration, WebRtcMode
from PIL import Image
import av
import queue
from typing import List, NamedTuple


faces = ['Abdur Samad', 'Ahsan Ahmed', 'Asef', 'Ashik', 'Azizul Hakim', 'DDS', 'Mayaz', 'Meheraj', 'Nayeem Khan', 'Nayem', 'Risul Islam Fahim', 'Saif', 'Saki', 'Samir', 'Shahtab', 'Shimul Rahman Fahad', 'Shourov', 'Shuvo']

@st.cache_resource
def load_model():
    model = model_from_json(open("vgg16_saved_582.json", "r").read())
    model.load_weights('vgg16_saved_582.h5')
    return model
#app = Flask(__name__)

def dnn_extract_face1(img):
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    # img = cv2.imread('./Data/Samir/IMG20220208204853.jpg')
    (height, width) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[1]):
        confidence = detections[0,0,i,2]
#         print("confidence ",confidence)
        if confidence > 0.5:
            box = detections[0,0,i,3:7] * np.array([width,height,width,height])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(img,(startX,startY),(endX, endY),(0,255,255),2)
            cv2.putText(img, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            face = img[startY:endY,startX:endX]
            return face
        else:
            return None


class Detection(NamedTuple):
        name: str
        prob: float
        

class VideoProcessor(VideoProcessorBase):
    result_queue: "queue.Queue[List[Detection]]"
    def __init__(self) -> None:
        self.result_queue = queue.Queue()
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        result: List[Detection] = []
        frame = frame.to_ndarray(format="bgr24")
        frame = cv2.flip(frame,1)
        face = dnn_extract_face1(frame)
        if type(face) is np.ndarray:
            face = cv2.resize(face, (350, 350))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(face, 'RGB')
            img_array = np.array(im)
            img_array = np.expand_dims(img_array, axis=0)
            pred = model1.predict(img_array)
            predition = np.squeeze(pred)
            predIndex = np.argmax(predition)

            #             name = 'None matching'
            if (predition[predIndex] > 0.95):
                text = "{:.2f}%".format(predition[predIndex] * 100)
                name = str(faces[predIndex]) + ' ' + str(text)
                cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
                result.append(Detection(name=faces[predIndex], prob=float(predition[predIndex])))
            else:
                cv2.putText(frame, '', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
        else:
            cv2.putText(frame, 'No face detected :(', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        #             cv2.putText(frame,'',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        self.result_queue.put(result)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/')
# def index():
#     return render_template('index.html')


#RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.xten.com:3478"]}]}

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css('css/styles.css')
model1 = load_model()
st.markdown('<h2 align="center">Real Time Masked Face Recognition</h2>', unsafe_allow_html=True)
webrtc_ctx = webrtc_streamer(key="example",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=VideoProcessor,
                media_stream_constraints = {"video": True, "audio": False},
                async_processing = True)              

st.markdown("""
            <style>
            table td:nth-child(1) {
                display: none
            }
            table th:nth-child(1) {
                display: none
            }
            </style>
            """, unsafe_allow_html=True)
if st.checkbox("Show the detected face", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break
                    
#if __name__ == '__main__':
#    app.run(debug=True)

