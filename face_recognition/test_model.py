import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator  #有些tensorflower版本可能会ImageDataGenerator的位置不对
import numpy as np
import os


data_dir = 'lfw_train/'  # 数据集路径
img_size = (64, 64)    # 图像尺寸
batch_size = 32

# 数据加载和预处理
datagen = ImageDataGenerator(
    rescale=1.0 / 255,       # 归一化
    validation_split=0.2     # 80% 用于训练，20% 用于验证
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',     # 稀疏标签
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'
)

num_classes = len(train_generator.class_indices)  # 获取类别数量

# 构建浅层神经网络模型
model = models.Sequential([
    layers.Flatten(input_shape=(64, 64, 3)),   # 输入层，展平图像数据
    layers.Dense(256, activation='relu'),      # 第一个隐藏层
    layers.Dense(128, activation='relu'),      # 第二个隐藏层
    layers.Dense(num_classes, activation='softmax')  # 输出层
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# 加载模型进行预测
def predict_image(img_path, class_labels):
    loaded_model = models.load_model('face_recognition_model.h5')
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # 扩展维度以符合模型输入

    prediction = loaded_model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    return predicted_label

# 批量预测一个文件夹中的所有图片并将结果保存到文件
def predict_folder_images(folder_path, class_labels, output_file):
    with open(output_file, 'w') as f:  # 打开文件准备写入
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # 图片文件是jpg或png格式
                full_path = os.path.join(folder_path, filename)
                predicted_label = predict_image(full_path, class_labels)
                f.write(f"Image: {filename}, Predicted class: {predicted_label}\n")  # 写入文件

# 测试预测函数
if __name__ == "__main__":
    test_images_folder_path = 'lfw_test'  # 替换为测试图像文件夹路径
    output_file_path = 'test_out.txt'  # 输出文件路径
    if os.path.exists(test_images_folder_path):
        # 获取类别标签
        class_labels = list(train_generator.class_indices.keys())
        predict_folder_images(test_images_folder_path, class_labels, output_file_path)
    else:
        print("测试文件夹不存在.")