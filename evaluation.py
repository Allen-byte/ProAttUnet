import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from params import *
from tensorflow_addons.layers import WeightNormalization
import argparse


# 自定义损失函数以忽略填充标签的影响
def custom_categorical_crossentropy(y_true, y_pred):
    # 转换标签为整数
    y_true_int = tf.argmax(y_true, axis=-1)
    # 创建掩码，标记非填充标签的位置
    mask = tf.cast(y_true_int != (num_classes - 1), dtype=tf.float32)
    # 计算普通的categorical crossentropy
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    # 应用掩码
    loss *= mask
    # 返回平均损失，只考虑非填充标签的位置
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", help="select model weights to evaluate")

args = parser.parse_args()

model_name = args.model_name

# 数据路径
base_dir = "./test_sets/"

paths = [base_dir + path for path in os.listdir(base_dir)]
data_list, labels_list = [], []
keys = ["loss", "acc", "mae"]
test_datasets = ["CB513", "TS115", "CASP12", "NEW364"]


custom_objects = {
    'WeightNormalization': WeightNormalization,
    "custom_categorical_crossentropy": custom_categorical_crossentropy
}
# 加载模型
filepath = f"Best Models/{model_name}.hdf5"
model = load_model(filepath, custom_objects=custom_objects)

# 获取测试数据和标签
for path in paths:
    last_name = path.split("/")[-1]
    if last_name.startswith("data"):
        data_list.append(path)
    if last_name.startswith("labels"):
        labels_list.append(path)


test_list = []
for data in data_list:
    for label in labels_list:
        if data.split("/")[-1].split("data")[-1] in label.split("/")[-1]:
            test_list.append((data, label))


name = filepath.split("/")[-1]
res = f"using model: {name}\n---------------------------------------------------------------------------\n"

with open(f"results/{model_name}_{datetime.now()}.log", "a+", encoding="utf-8") as f:
    f.write(res)
    for path in test_list:
        name = path[0].split("/")[-1].split("_")[1].split(".")[0]
        print(f"current test dataset: {name}...")
        test_data = np.load(path[0])
        test_labels = np.load(path[1])

        # 评估模型
        res = model.evaluate(test_data, test_labels, batch_size=16)
        result = dict(zip(keys, res))

        f.write(name + "(recorded at " + str(datetime.now()) + ")\n")
        f.write(str(result) + "\n\n")