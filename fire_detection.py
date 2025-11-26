from utils import show_images_from_folder, train_YOLO, one_image_inference, model_quantization, model_eval
from ultralytics import YOLO
import torch

if __name__=="__main__":

    # check CUDA and GPU device
    print("GPU device: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")

    # display image folder path
    display_image_path = "display/images/"
    display_label_path = "display/labels/"

    # display images with bounding boxes
    show_images_from_folder(display_image_path, display_label_path, 'Display', save_image=False)

    model_name = 'yolo11n' # define model name

    train_YOLO(model_name)  # run this only for training the model

    model_fp32_pt = YOLO(f"runs/detect/{model_name}/weights/best.pt") # get the best model.pt

    # do inferece on an image
    sample_image_path = r'D-Fire/train/images/AoF04038.jpg'
    one_image_inference(model_fp32_pt, sample_image_path)

    # Exporte
    # onnx
    # https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html # quantization details

    model_fp32_pt.export(format='onnx') # export the model as onnx (best.pt->best.onnx)
    model_fp32_path = f"runs/detect/{model_name}/weights/best.onnx"
    calibration_path = f"D-Fire/test/images"
    model_quant_int8D_path = f"runs/detect/{model_name}/weights/best_quant_d.onnx"
    model_quant_fp16_path = f"runs/detect/{model_name}/weights/best_quant_fp16.onnx"

    # ncnn
    # https://github.com/Tencent/ncnn
    model_fp32_pt.export(format="ncnn", name="fp32")
    model_fp32_pt.export(format="ncnn", name="fp16", half=True)

    # quantize model
    model_quantization(model_name, model_fp32_path, model_quant_int8D_path, model_quant_fp16_path, 
                       quant_dynamic=True, quant_fp16=True) # do int8 (dynamic) and fp16 quantization for the original model

    # model evaluation
    model__quant_fp32 = YOLO(f"runs/detect/{model_name}/weights/best_fp32.onnx") 
    model__quant_fp16 = YOLO(f"runs/detect/{model_name}/weights/best_quant_fp16.onnx")
    model_quant_int8D = YOLO(f"runs/detect/{model_name}/weights/best_quant_int8_d.onnx") 
    model__quant_fp32_ncnn = YOLO(f"runs/detect/{model_name}/weights/best_ncnn_model_fp32") 
    model__quant_fp16_ncnn = YOLO(f"runs/detect/{model_name}/weights/best_ncnn_model_fp16")

    # get the val results
    model_eval(model__quant_fp32, f'{model_name}_fp32')
    model_eval(model_quant_int8D, f'{model_name}_int8D')
    model_eval(model__quant_fp16, f'{model_name}_fp16')
    model_eval(model__quant_fp32_ncnn, f'{model_name}_fp32_ncnn')
    model_eval(model__quant_fp16_ncnn, f'{model_name}_fp16_ncnn')


    
