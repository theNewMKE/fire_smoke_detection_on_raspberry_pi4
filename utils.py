import os 
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import onnx
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxconverter_common import float16

""" Some functions that can be useful for dealing with coordinates. """

def non_negative(coord):
    # this fucntion can be found here: 
    # https://github.com/gaia-solutions-on-demand/DFireDataset/blob/master/utils/utils.py
    """
        Sets negative coordinates to zero. This fixes bugs in some labeling tools.
        
        Input:
            coord: Int or float
            Any number that represents a coordinate, whether normalized or not.
    """
    
    if coord < 0:
        return 0
    else:
        return coord
    
def pixel2yolo(dim, pixel_coords):
    # this fucntion can be found here: 
    # https://github.com/gaia-solutions-on-demand/DFireDataset/blob/master/utils/utils.py
    """
        Transforms coordinates in YOLO format to coordinates in pixels.
        
        Input:
            dim: Tuple or list
            Image size (width, height).
            pixel_coords: List
            Bounding box coordinates in pixels (xmin, ymin, xmax, ymax).
        Output:
            yolo_coords: List
            Bounding box coordinates in YOLO format (xcenter, ycenter, width, height).
    """
    
    dw = 1/dim[0]
    dh = 1/dim[1]
    xcenter = non_negative(dw*(pixel_coords[0] + pixel_coords[2])/2)
    ycenter = non_negative(dh*(pixel_coords[1] + pixel_coords[3])/2)
    width = non_negative(dw*(pixel_coords[2] - pixel_coords[0]))
    height = non_negative(dh*(pixel_coords[3] - pixel_coords[1]))
    
    yolo_coords = [xcenter, ycenter, width, height]
    
    return yolo_coords

def yolo2pixel(dim, yolo_coords):
    # this fucntion can be found here: 
    # https://github.com/gaia-solutions-on-demand/DFireDataset/blob/master/utils/utils.py
    """
        Transforms coordinates in YOLO format to coordinates in pixels.
        
        Input:
            dim: Tuple or list
            Image size (width, height).
            yolo_coords: List
            Bounding box coordinates in YOLO format (xcenter, ycenter, width, height).
        Output:
            pixel_coords: List
            Bounding box coordinates in pixels (xmin, ymin, xmax, ymax).
    """
    
    xmin = non_negative(round(dim[0] * (yolo_coords[0] - yolo_coords[2]/2)))
    xmax = non_negative(round(dim[0] * (yolo_coords[0] + yolo_coords[2]/2)))
    ymin = non_negative(round(dim[1] * (yolo_coords[1] - yolo_coords[3]/2)))
    ymax = non_negative(round(dim[1] * (yolo_coords[1] + yolo_coords[3]/2)))
    
    pixel_coords = [xmin, ymin, xmax, ymax]
    
    return pixel_coords

def load_labels(label_folder_path: str, i: int):
    """
        Loads the labels for each image
        
        Input:
            label_folder_path: the path of the label folder
            i: the index of each label
        Output:
            parts: YOLO class and bounding box information
    """
    parts = []
    labels = os.listdir(label_folder_path)
    label_path = os.path.join(label_folder_path, labels[i]) # get the path for each label
    with open(label_path, 'r') as f:
        for line in f:
            items = line.strip().split() # split the numbers
            cls = int(items[0]) # class id
            x, y, w, h = map(float, items[1:5]) # get the actual coords
            parts.append((cls, [x, y, w, h])) # arrange parts as class id and yolo box
        # print(parts)
    return parts
# parts = load_labels(train_label_path, 0)
# print(parts)

def show_images_from_folder(image_folder_path: str, label_folder_path: str, title: str, save_image: bool, num_images: int=10):
    """
        Desplaying a certain number of images with bounding box and class name on them
        
        Input:
            image_folder_path: the path of the image folder
            label_folder_path: the path of the label folder
            title: training or testing images
            save_image: boolean value for saving the displayed images or not
            num_images: number of images desplayed
    """
    
    images = os.listdir(image_folder_path)  # get the image names inside a list
    # print(images)

    plt.figure(figsize=(16, 4.5)) # define figure size


    for i in range(num_images):     # plot i images

        img_path = os.path.join(image_folder_path, images[i]) # join the folder and image paths
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        # print(img.shape) # check the image shape

        labels = load_labels(label_folder_path, i) # load image labels
        H, W = img.shape[0:2] # get the H, W of the image

        # get the coords
        for cls, yolo_box in labels:
            # use yolo2pixel function to convert yolobox to pixel coords, the first var is a tuple (width, hight) of the image, but usually we will have H, W from shape function , so swap them
            pixel_coords = yolo2pixel((W, H), yolo_box) 
            if(cls==0):
                # 0->smoke, 1->fire
                # (xmin, ymin), (xmax, ymax)
                cv2.rectangle(img, (pixel_coords[0], pixel_coords[1]), (pixel_coords[2], pixel_coords[3]), (0, 0, 255), 5) # class id 0 -> blue
            else:
                cv2.rectangle(img, (pixel_coords[0], pixel_coords[1]), (pixel_coords[2], pixel_coords[3]), (255, 0, 0), 5) # class id 1 -> red

        plt.subplot(2, int(num_images/2), i+1)
        plt.imshow(img)
        plt.title(f"{title} image {i+1}") 
        plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if save_image:
        plt.savefig("display_img.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()

def train_YOLO(model_name: str, epochs: int=100): 
    """
        Trains YOLO model 
        
        Input:
            model_name: name of the YOLO model
            epochs: training epochs
    """

    model = YOLO(model_name + ".pt")   #yolo11n, yolo12n
    model.to('cuda')

    # https://github.com/gaia-solutions-on-demand/DFireDataset?tab=readme-ov-file # D-Fire dataset
    # https://docs.ultralytics.com/models/yolo11/#usage-examples  # ultralytics examples 
    # https://github.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/yolo_detect.py # Github YOLO for fire detection example
    # https://www.youtube.com/watch?v=r0RspiLG260 # How to Train YOLO Object Detection Models in Google Colab (YOLO11, YOLOv8, YOLOv5)
    # https://www.youtube.com/watch?v=LNwODJXcvt4 # How to Train Ultralytics YOLOv8 models on Your Custom Dataset in Google Colab | Episode 3
    # https://www.youtube.com/watch?v=zza_TDApimg # California wildfires' destruction in Los Angeles area captured in videos
    model.train(
        data="D-Fire/dataset.yml",
        device=0,
        epochs=epochs,  
        name='fire_detection_' + model_name,
        imgsz=640,
        batch=32,  
        patience=10,
    )

def one_image_inference(model_fp32_pt, sample_image_path):
    """
        Does one inference example of an image
        
        Input:
            model_fp32_pt: fp32.pt YOLO model
            sample_image_path: path of example image
    """
    result = model_fp32_pt.predict(sample_image_path)
    # print(result[0])
    result[0].show()

def model_quantization(model_name, model_fp32_path, model_quant_int8D_path, model_quant_fp16_path, 
                       quant_dynamic=True, quant_fp16=True):
    """
        Quantizes models
        
        Input:
            model_name: name of the model
            model_fp32_path: path of the fp32 model
            model_quant_int8D_path: path of the int8 model
            model_quant_fp16_path: path of the fp16 model
            quant_dynamic: boolean value for int8 quntization
            quant_fp16: boolean value for fp16 quntization
    """
    original_model_onnx = onnx.load(f"runs/detect/fire_detection_{model_name}/weights/best.onnx") # get the best model.onnx
    if(quant_dynamic): 
        quantize_dynamic(
            model_input=model_fp32_path,
            model_output=model_quant_int8D_path,
            weight_type=QuantType.QUInt8   # by default the weight type is QInt8, if there is an error, change from QInt8 to QUInt8
            # https://github.com/microsoft/onnxruntime/issues/3130#issuecomment-1105200621
        )

    if(quant_fp16): 
        # float16: https://onnxruntime.ai/docs/performance/model-optimizations/float16.html#float16-tool-arguments
        model_fp16_onnx = float16.convert_float_to_float16(original_model_onnx, min_positive_val=1e-7, max_finite_val=1e4, keep_io_types=False,
                            disable_shape_infer=False, op_block_list=None, node_block_list=None)
        onnx.save(model_fp16_onnx, model_quant_fp16_path)

def model_eval(model, name):
    """
        Evaluates the models
        
        Input:
            model: YOLO model
            name: model name
    """
    metrics = model.val(
        data="D-Fire/dataset.yml",
        split="val",      # use validation set
        verbose=True
    )

    print(f"\nModel name: {name}")
    print("mAP@50-95: ", metrics.box.map)     
    print("mAP@50: ", metrics.box.map50)    
    print("mAP@75: ", metrics.box.map75)    
    print("precision:", metrics.box.mp)
    print("recall: ", metrics.box.mr)
    print(f"inference time: {metrics.speed['inference']:.2f} ms")
    print(f"preprocess: {metrics.speed['preprocess']:.2f} ms")
    print("metrics: ", metrics.results_dict)