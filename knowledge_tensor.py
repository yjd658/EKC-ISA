import numpy as np
import torch
from snownlp import SnowNLP
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence

import torch
from torch.nn.utils.rnn import pad_sequence

import torch
from torch.nn.utils.rnn import pad_sequence

def generate_sentence_tensor(input_file, embedding_dict, max_length=20):
    # 创建一个空的列表用于接纳每一个句子的embedding_tensor
    sentence_tensors_list = []

    # 逐行读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for j,line in enumerate(f):
            # 获取句子部分
            sentence, _ = line.strip().split(' ', 1)

            # 使用Snownlp进行中文分词
            s = SnowNLP(sentence)
            sentence_tokens = s.words

            # 将所有embedding转化为tensor并添加到sentence_tensors_list中
            embedding_size = 300
            sentence_tensor = torch.empty(max_length, embedding_size)

            for i, token in enumerate(sentence_tokens):
                # print(token)
                if i >= max_length:
                    break
                embedding = embedding_dict.get(token, np.zeros(embedding_size))
                embedding_tensor = torch.Tensor(embedding)
                sentence_tensor[i] = embedding_tensor

            # 将当前句子的Tensor添加到sentence_tensors_list中
            print(j)
            sentence_tensors_list.append(sentence_tensor)

    # 使用pad_sequence对不同长度的句子进行填充，将每个句子填充到长度为20
    all_sentence_tensor = pad_sequence(sentence_tensors_list, batch_first=True, padding_value=0)

    # 保存结果到文件
    output_file = 'test_sentences_tensor.pt'
    torch.save(all_sentence_tensor, output_file)

    print("句子信息Tensor已保存到", output_file, "文件。")
    return output_file



f_read = open('dict_file.pkl', 'rb')
dict2 = pickle.load(f_read)
f_read.close()
result_file1 = generate_sentence_tensor('test.txt', dict2)
# result_file2 = generate_sentence_tensor('dev.txt', dict2)
# result_file3 = generate_sentence_tensor('test.txt', dict2)

