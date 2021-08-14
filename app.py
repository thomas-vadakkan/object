from flask import Flask, render_template, Response
import cv2
import numpy as np

thres = 0.5
nms_threshold = 0.2
# cap = cv2.VideoCapture('https://192.168.1.6:8080/video')
# cap = cv2.VideoCapture('https://my.ivideon.com/cameras/groups/camera/100-t208cRqzt2EIKcU2QSpusB/0')
cap = cv2.imread('image01.jpg')
# cap.set(3, 640)  # width
# cap.set(4, 480)  # height
# cap.set(10, 150)  # brightness
app = Flask(__name__)

className = []

classFile = 'coco.names'
with open(classFile, 'rt') as f:
    className = f.read().rsplit('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# settings
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


# camera = cv2.VideoCapture('rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream')  # use 0 for web camera
# camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture('https://192.168.1.6:8080')

#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        # success, img = cap.read()  # read the camera frame
        # if not success:
        #     break
        # else:
            img = cap
            classIDs, confs, bbox = net.detect(img, confThreshold=thres)
        # print(classIDs,bbox)

            bbox = list(bbox)
            confs = list(np.array(confs).reshape(1, -1)[0])
            confs = list(map(float, confs))
            indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

            count = [0] * len(className)
            if len(classIDs) !=0:
                for i in indices:
                    i = i[0]
                    box = bbox[i]
                    x,y,w,h = box[0],box[1],box[2],box[3]

                    cv2.rectangle(img,(x,y),(x+w,h+y),color=(0,255,0),thickness=2)

                    cv2.putText(img,className[classIDs[i][0]-1].upper(),(box[0]+10,box[1]+30),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confs[i]*100,2))+"%",(box[0]+200,box[1]+30),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

                    for m in classIDs[i]:
                        if classIDs[i][0] == m:
                            count[m] = count[m] + 1
                            cv2.putText(img,className[classIDs[i][0]-1]+": "+str(count[m]),(50,50+m),
                                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                            print(className[classIDs[i][0]-1]+": "+str(count[m]))

            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    # app.run(host='192.168.1.4',port='5000', threaded=True, debug=False, use_reloader=False)
    app.run(debug=False)
    # app.run(host='192.168.1.4', debug=False)
    # app.run(host='103.165.21.211', debug=False)
    # app.run(host='192.168.43.236', debug=False)
    # app.run(host='0.0.0.0', debug=False)

    # app.run(debug=True)
