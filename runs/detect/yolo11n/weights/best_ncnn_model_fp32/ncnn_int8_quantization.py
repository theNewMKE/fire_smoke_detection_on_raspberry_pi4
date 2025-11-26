import subprocess
import os

# 输入文件
param_fp32 = "model.ncnn.param"
bin_fp32   = "model.ncnn.bin"

# 输出文件
param_int8 = "model_int8.param"
bin_int8   = "model_int8.bin"

# 校准列表
img_txt = "images.txt"

# ncnn2int8.exe 路径（如果和脚本在同一目录就不用改）
ncnn2int8_path = "ncnn2int8.exe"

cmd = [
    ncnn2int8_path,
    param_fp32,
    bin_fp32,
    param_int8,
    bin_int8,
    img_txt
]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

print("\n🎉 INT8 量化完成！生成文件：")
print(" -", param_int8)
print(" -", bin_int8)





# import ncnn

# net = ncnn.Net()
# ret1 = net.load_param("runs/detect/fire_detection_yolo11n/weights/best_ncnn_model_fp32/model.ncnn.param")
# ret2 = net.load_model("runs/detect/fire_detection_yolo11n/weights/best_ncnn_model_fp32/model.ncnn.bin")
# print("param:", ret1)
# print("bin:", ret2)
# print("inputs:", net.input_names())
# print("outputs:", net.output_names())
