#!/usr/bin/env python

# import device_patches       # Device specific patches for Jetson Nano (needs to be before importing cv2)

try:
    import cv2
except ImportError:
    print('Missing OpenCV, install via `pip3 install "opencv-python>=4.5.1.48,<5"`')
    exit(1)
import os
import sys, getopt
import signal
import time
from edge_impulse_linux.image import ImageImpulseRunner

runner = None
# if you don't want to see a camera preview, set this to False
show_camera = True
if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False

def now():
    return round(time.time() * 1000)

def get_webcams():
    port_ids = []
    for port in range(5):
        print("Looking for a camera in port %s:" %port)
        camera = cv2.VideoCapture(port)
        if camera.isOpened():
            ret = camera.read()[0]
            if ret:
                backendName =camera.getBackendName()
                # backendName = "OPENCV"
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) found in port %s " %(backendName,h,w, port))
                port_ids.append(port)
            camera.release()
    return port_ids

def sigint_handler(sig, frame):
    print('Interrupted')
    if (runner):
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def help():
    print('python classify.py <path_to_model.eim> <Camera port ID, only required when more than 1 camera is present>')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) == 0:
        help()
        sys.exit(2)

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            # model_info = runner.init(debug=True, timeout=60) # to get debug print out and set longer timeout
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            labels = model_info['model_parameters']['labels']
            if len(args)>= 2:
                videoCaptureDeviceId = int(args[1])
            else:
                port_ids = get_webcams()
                if len(port_ids) == 0:
                    raise Exception('Cannot find any webcams')
                if len(args)<= 1 and len(port_ids)> 1:
                    raise Exception("Multiple cameras found. Add the camera port ID as a second argument to use to this script")
                videoCaptureDeviceId = int(port_ids[0])

            camera = cv2.VideoCapture(videoCaptureDeviceId)
            
            # set resolution here
            # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
            # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            ret = camera.read()[0]
            if ret:
                backendName = camera.getBackendName()
                # backendName = "OPENCV"
                w = camera.get(3)
                h = camera.get(4)
                # w = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                # h = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print("Camera %s (%s x %s) in port %s selected." %(backendName,h,w, videoCaptureDeviceId))
                camera.release()
            else:
                raise Exception("Couldn't initialize selected camera.")

            next_frame = 0 # limit to ~10 fps here

            for res, img in runner.classifier(videoCaptureDeviceId):
                if (next_frame > now()):
                    time.sleep((next_frame - now()) / 1000)

                print('classification runner response', res)

                if "classification" in res["result"].keys():
                    print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                    for label in labels:
                        score = res['result']['classification'][label]
                        print('%s: %.2f\t' % (label, score), end='')
                    print('', flush=True)
                elif "bounding_boxes" in res["result"].keys():
                    print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                    for bb in res["result"]["bounding_boxes"]:
                        # draw box and text
                        # print(res["result"])
                        # get the coords
                        x1 = bb['x']
                        y1 = bb['y']
                        x2 = bb['x'] + bb['width']
                        y2 = bb['y'] + bb['height']
                        conf = bb['value']

                        # define the two classes here
                        if bb['label'] == '0':
                            cls = "Smoke"
                            color = (0, 0, 255)
                        elif bb['label'] == '1':
                            cls = "Fire"
                            color = (255, 0, 0)

                        # draw bounding box
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                        # get the label and the conf. value
                        label = f"{cls}:{int(conf * 100)}%"
                        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                        # draw filled rectangle behind the label
                        cv2.rectangle(img, (x1, y1 - h - 2), (x1 + w, y1), color, 1)
                        # putText for label
                        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                        # if bb['lbel'] == '0':
                        #     print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                        #     img = cv2.rectangle(img, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (0, 0, 255), 1)
                        # elif bb['lbel'] == '1':
                        #     print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % ('Fire', bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                        #     img = cv2.rectangle(img, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)
                elif "freeform" in res['result'].keys():
                    print('Result (%d ms.)' % (res['timing']['dsp'] + res['timing']['classification']))
                    for i in range(0, len(res['result']['freeform'])):
                        print(f'    Freeform output {i}:', ", ".join(f"{x:.4f}" for x in res['result']['freeform'][i]))

                if (show_camera):
                    cv2.imshow('edgeimpulse', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) == ord('q'):
                        break

                next_frame = now() + 100
        finally:
            if (runner):
                runner.stop()

if __name__ == "__main__":
   main(sys.argv[1:])



# Found 2 bounding boxes (129 ms.)
# {'bounding_boxes': [{'height': 269, 'label': '0', 'value': 0.5878275632858276, 'width': 320, 'x': 0, 'y': 23}, {'height': 17, 'label': '1', 'value': 0.5353429913520813, 'width': 34, 'x': 167, 'y': 137}]}
