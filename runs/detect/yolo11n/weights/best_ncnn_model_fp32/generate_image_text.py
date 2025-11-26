import os

# 设置你的校准图片路径
image_folder = f"D-Fire/test/images"       # ← 修改为你的路径
output_txt = "images.txt"                  # 输出文件

exts = [".jpg", ".jpeg", ".png", ".bmp"]

with open(output_txt, "w") as f:
    for name in os.listdir(image_folder):
        if name.lower().endswith(tuple(exts)):
            full_path = os.path.abspath(os.path.join(image_folder, name))
            f.write(full_path + "\n")

print(f"images.txt 已生成，共 {len(open(output_txt).readlines())} 张图片。")
print(f"输出路径: {os.path.abspath(output_txt)}")
