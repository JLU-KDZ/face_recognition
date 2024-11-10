# train_model.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# 重定向控制台输出到文件
import sys
original_stdout = sys.stdout
output_file = open('train_out.txt', 'w')
sys.stdout = output_file

# 数据集路径和图像参数
data_dir = 'lfw_train/'  # 数据集路径
img_size = (64, 64)      # 图像尺寸
batch_size = 32

# 数据加载和预处理
datagen = ImageDataGenerator(
    rescale=1.0 / 255,    # 归一化
    validation_split=0.2  # 80%用于训练，20%用于验证
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',  # 稀疏标签
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

# 构建模型
model = models.Sequential([
    layers.Flatten(input_shape=(64, 64, 3)),  # 展平图像
    layers.Dense(256, activation='relu'),     # 第一个隐藏层
    layers.Dense(128, activation='relu'),     # 第二个隐藏层
    layers.Dense(num_classes, activation='softmax')  # 输出层
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练参数
epochs = 50  # 训练轮数

# 训练模型并保存训练过程
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# 评估模型
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# 保存模型
model.save('face_recognition_model.h5')
print("Model saved as face_recognition_model.h5")

# 恢复控制台输出
sys.stdout = original_stdout
output_file.close()

# 绘制训练和验证准确率
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('acc.png')
plt.close()

# 绘制训练和验证损失
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('loss.png')
plt.close()
