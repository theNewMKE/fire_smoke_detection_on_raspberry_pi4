Folder for best ncnn fp32 model,
the model has 4 files:
metadata.ymal
model.ncnn.bin
model.ncnn.model
model_ncnn.py

The other files are used to int8 quantization,
which is NOT successful!

generate_image_test.py -> generate image list will be used in quantization
images.txt -> generated image list
ncnn2int8.exe -> inside ncnn-20250916-windows-vs2019/x64/bin
ncnn-20250916-windows-vs2019 -> unzipped downloaded folder
Download from: https://github.com/Tencent/ncnn/releases
