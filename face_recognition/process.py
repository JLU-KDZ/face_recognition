import os
import shutil


lfw_path = 'lfw'
new_folder_path = 'lfw_train'


if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)


for root, dirs, files in os.walk(lfw_path):
    for dir_name in dirs:
        subfolder_path = os.path.join(root, dir_name)
        # 计算每个子文件夹中的图片数量
        image_count = len([f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])
        # 如果图片数量大于等于20，则复制整个子文件夹到新文件夹
        if image_count >= 20:
            new_subfolder_path = os.path.join(new_folder_path, dir_name)

            if not os.path.exists(new_subfolder_path):
                shutil.copytree(subfolder_path, new_subfolder_path)
            else:
                print(f"跳过复制：{subfolder_path}")

