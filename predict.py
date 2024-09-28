import torch
import esm
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from tensorflow_addons.layers import WeightNormalization
import argparse
from datetime import datetime
import math


parser = argparse.ArgumentParser()

parser.add_argument("--fasta_file", "-f")
parser.add_argument("--model_name", "-m", help="select model weights to predict")


args = parser.parse_args()


# 自定义损失函数以忽略填充标签的影响
def custom_categorical_crossentropy(y_true, y_pred):
    # 转换标签为整数
    y_true_int = tf.argmax(y_true, axis=-1)
    # 创建掩码，标记非填充标签的位置
    mask = tf.cast(y_true_int != (num_classes-1), dtype=tf.float32)
    # 计算普通的categorical crossentropy
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    # 应用掩码
    loss *= mask
    # 返回平均损失，只考虑非填充标签的位置
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


class CrossAttentionLayer(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(CrossAttentionLayer, self).__init__(**kwargs)
        self.filters = filters
        self.conv_v = layers.Conv1D(filters, kernel_size=1, padding='same')
        self.conv_k = layers.Conv1D(filters, kernel_size=1, padding='same')
        self.conv_q = layers.Conv1D(filters, kernel_size=1, padding='same')
        self.conv_d1 = layers.Conv1D(filters, kernel_size=3, padding='same')
        self.conv_out = layers.Conv1D(filters, kernel_size=1, padding='same')

    def call(self, inputs):
        Q, K, V = inputs
        V_ = self.conv_d1(self.conv_v(V))
        K_ = self.conv_d1(self.conv_k(K))
        Q_ = self.conv_d1(self.conv_q(Q))

        attention_scores = tf.matmul(Q_, K_, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)

        attention_output = tf.matmul(attention_scores, V_)
        output = self.conv_out(attention_output)

        return output

    def get_config(self):
        config = super(CrossAttentionLayer, self).get_config()
        config.update({'filters': self.filters})
        return config


def torch_to_tf(data):
    data_numpy = data.numpy()
    return tf.convert_to_tensor(data_numpy)

# 将序列分为指定长度的子序列，并丢弃不满足阈值的序列（标签同步）
def split_sequences(sequences, window_size):
    all_sub_sequences = []
    indexs = []

    for sequence in sequences:
        sequence = list(sequence)
        if len(sequence) < window_size:
            padding_length = window_size - len(sequence)
            padded_sequence = sequence + ['<pad>'] * padding_length
            all_sub_sequences.append(''.join(padded_sequence))
        else:
            num_sub_sequences = max(2, math.ceil((len(sequence) - window_size) / (window_size // 2)))
            step_size = max(1, (len(sequence) - window_size) // (num_sub_sequences - 1))
            for i in range(num_sub_sequences):
                start_index = i * step_size
                end_index = start_index + window_size
                if end_index > len(sequence):
                    end_index = len(sequence)
                    start_index = end_index - window_size
                sub_sequence = sequence[start_index:end_index]
                all_sub_sequences.append(''.join(sub_sequence))
                indexs.append((start_index, end_index))
    return all_sub_sequences, indexs


def make_esm_list(seqs):
    esm_list = [(f"protein_{i}", seq) for i, seq in enumerate(seqs)]
    return esm_list

# 获取类别标签
def get_labels(preds):
    num_samples, num_instances, num_classes = preds.shape
    labels = np.zeros((num_samples, num_instances))
    for i in range(num_samples):
        for j in range(num_instances):
            labels[i][j] = np.argmax(preds[i][j])

    label_mapping = {0: 'L', 1: 'H', 2: 'B', 3: 'E', 4: 'G', 5: 'I', 6: 'T', 7: 'S'}
    final_labels = np.vectorize(label_mapping.get)(labels)

    return final_labels


def read_fasta_file(file_path):
    protein_data = {}
    with open(file_path, 'r') as file:
        current_protein_name = ""
        current_sequence = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if current_protein_name!= "":
                    protein_data[current_protein_name] = current_sequence
                    current_sequence = ""
                current_protein_name = line[1:]
            else:
                current_sequence += line
        if current_protein_name!= "":
            protein_data[current_protein_name] = current_sequence
    return protein_data


def final_pred(preds, indexs):
    res = [''] * len(seq)
    for (start, end), sub_pred in zip(indexs, preds):
        for i in range(start, end):
            if res[i] == '':
                res[i] = sub_pred[i - start]
    return res
    


custom_objects = {
    'WeightNormalization': WeightNormalization,
    'CrossAttentionLayer': CrossAttentionLayer,
    "custom_categorical_crossentropy": custom_categorical_crossentropy,
}

fasta_path = args.fasta_file
model_name = args.model_name

for item in read_fasta_file(fasta_path).items():
    name = item[0].split("|")[0]                             
    seq = item[1]                                           


model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results


seqs = []
seqs.append(seq)
x, indexs = split_sequences(seqs, 150)
x = make_esm_list(x)
save_path = f"prediction/prediction_{name}_{datetime.now()}.log"

batch_labels, batch_strs, batch_tokens = batch_converter(x)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]
token_representations = token_representations[:, 1: batch_lens[0] - 1]
contacts = results['contacts']
padded_repr = torch.zeros(token_representations.shape[0], 150, token_representations.shape[2])
padded_repr[:, :token_representations.shape[1], :] = token_representations

x = torch.cat((padded_repr, contacts), dim=2)
x = torch_to_tf(x)

new_model = load_model(f'Best Models/{model_name}.hdf5', custom_objects=custom_objects)
preds = new_model.predict(x)
pred_labels = get_labels(preds)
if len(seq) < 150:
    final_res = "".join(pred_labels[0][:len(seq)])
else:
    final_res = "".join(final_pred(pred_labels, indexs))
with open(save_path, "w", encoding="utf-8") as f:
    print(f"The prediction of {name} has been finished! You can check it at {save_path}")
    f.write(f"input sequence: {seq} \n\n")
    f.write(f"prediction result: {final_res}")
# print(new_model.predict(x).shape)
# print(preds)
