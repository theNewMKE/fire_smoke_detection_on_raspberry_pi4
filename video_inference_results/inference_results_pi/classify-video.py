#!/usr/bin/env python

# import device_patches       # Device specific patches for Jetson Nano (needs to be before importing cv2)

try:
    import cv2
except ImportError:
    print('Missing OpenCV, install via `pip3 install "opencv-python>=4.5.1.48,<5"`')
    exit(1)
import os
import time
import sys, getopt
from edge_impulse_linux.image import ImageImpulseRunner

runner = None
# if you don't want to see a video preview, set this to False
show_camera = True
if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False


def help():
    print('python classify-video.py <path_to_model.eim> <path_to_video.mp4>')

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

    if len(args) != 2:
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

            vidcap = cv2.VideoCapture(args[1])


            sec = 0
            start_time = time.time()

            def getFrame(sec):
                # don't just read every 1 sec (the timestamp EI gave), instead reading the frames sequentially
                # vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
                hasFrames,image = vidcap.read()
                if hasFrames:
                    return image
                else:
                    print('Failed to load frame', args[1])
                    exit(1)


            img = getFrame(sec)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = None
            output_name = "EI_output.mp4"

            while img.size != 0:

                # imread returns images in BGR format, so we need to convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # copy the raw image as raw
                raw = img.copy()
                H_raw, W_raw, _ = raw.shape

                # get the input clip fps
                input_fps = vidcap.get(cv2.CAP_PROP_FPS)
                # print(input_fps)

                if out is None:
                    out = cv2.VideoWriter(output_name, fourcc, input_fps, (W_raw, H_raw))
                # from the debug image, we know that the original img size if YOLO 320x320, FOMO 48x48

                # check YOLO or FOMO from EI
                # print(modelfile.split('/')[-1] )
                if modelfile.split('/')[-1] == 'fire_detection_yolo_nano.eim' or modelfile.split('/')[-1] == 'fire_detection_yolo_pico.eim':
                    EI_input = 320 # 320, 48
                elif modelfile.split('/')[-1] == 'fire_detection_FOMO2.eim':
                    EI_input = 48 # 320, 48
                # print(EI_input)

                # resize the frame, get the scale first
                scale = min(W_raw / EI_input, H_raw / EI_input)
                scaled_w = W_raw / scale
                scaled_h = H_raw / scale

                offset_x = (scaled_w - EI_input) / 2
                offset_y = (scaled_h - EI_input) / 2

                def map_to_raw(x, y, w, h):
                    x_new = (x + offset_x) * scale
                    y_new = (y + offset_y) * scale
                    w_new = w * scale
                    h_new = h * scale
                    return int(x_new), int(y_new), int(w_new), int(h_new)

                # get_features_from_image also takes a crop direction arguments in case you don't have square images
                features, cropped = runner.get_features_from_image(img)

                # the image will be resized and cropped, save a copy of the picture here
                # so you can see what's being passed into the classifier
                cv2.imwrite('debug.jpg', cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

                res = runner.classify(features)

                if "classification" in res["result"].keys():
                    print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                    for label in labels:
                        score = res['result']['classification'][label]
                        print('%s: %.2f\t' % (label, score), end='')
                    print('', flush=True)

                elif "bounding_boxes" in res["result"].keys():
                    print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                    for bb in res["result"]["bounding_boxes"]:
                        print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))

                        # draw bounding box on 320x320
                        # img = cv2.rectangle(cropped, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)

                        x = bb['x']
                        y = bb['y']
                        w = bb['width']
                        h = bb['height']      
                        conf = bb['value']
                        x2, y2, w2, h2 = map_to_raw(x, y, w, h)

                        # define the two classes here
                        if bb['label'] == '0':
                            cls = "Smoke"
                            color = (0, 0, 255)
                        elif bb['label'] == '1':
                            cls = "Fire"
                            color = (255, 0, 0)              
                        
                        # draw bounding box by using the riszed croods.
                        img = cv2.rectangle(raw, (x2, y2), (x2 + w2, y2 + h2), color, 2)
                        label = f"{cls}:{int(conf * 100)}%"
                        # putText for label
                        cv2.putText(raw, label, (x2, y2 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                        # draw bounding box on raw image
                        # img = cv2.rectangle(raw, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)

                elif "freeform" in res['result'].keys():
                    print('Result (%d ms.)' % (res['timing']['dsp'] + res['timing']['classification']))
                    for i in range(0, len(res['result']['freeform'])):
                        print(f'    Freeform output {i}:', ", ".join(f"{x:.4f}" for x in res['result']['freeform'][i]))

                # Object tracking output
                if "object_tracking" in res["result"].keys():
                    print('Found %d tracked objects' % (len(res["result"]["object_tracking"])))
                    for obj in res["result"]["object_tracking"]:
                        print('\tID=%s, label=%s, value=%.2f, x=%d, y=%d, w=%d, h=%d' % (
                            obj['object_id'], obj['label'], obj['value'], obj['x'], obj['y'], obj['width'], obj['height']))
                        # Draw bounding box
                        x = obj['x']
                        y = obj['y']
                        w = obj['width']
                        h = obj['height']
                        img = cv2.rectangle(cropped, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Draw label and ID
                        label_id_text = f"ID={obj['object_id']}, {obj['label']}"
                        text_x = x
                        text_y = y - 10 if y - 10 > 10 else y + 20
                        img = cv2.putText(img, label_id_text, (text_x, text_y),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if "visual_anomaly_grid" in res["result"].keys():
                    print('Found %d visual anomalies (%d ms.)' % (len(res["result"]["visual_anomaly_grid"]), res['timing']['dsp'] +
                                                                                                                res['timing']['classification'] +
                                                                                                                res['timing']['anomaly']))
                    for grid_cell in res["result"]["visual_anomaly_grid"]:
                        print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (grid_cell['label'], grid_cell['value'], grid_cell['x'], grid_cell['y'], grid_cell['width'], grid_cell['height']))
                        img = cv2.rectangle(cropped, (grid_cell['x'], grid_cell['y']), (grid_cell['x'] + grid_cell['width'], grid_cell['y'] + grid_cell['height']), (255, 125, 0), 1)

                if (show_camera):
                    # cv2.imshow('edgeimpulse', cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
                    cv2.imshow('edgeimpulse', cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) == ord('q'):
                        break

                sec = time.time() - start_time
                sec = round(sec, 2)
                print("Getting frame at: %.2f sec" % sec)
                img = getFrame(sec)

                if out is not None:
                    out.write(cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
        finally:
            if out:
                out.release()

            if (runner):
                runner.stop()

if __name__ == "__main__":
   main(sys.argv[1:])
