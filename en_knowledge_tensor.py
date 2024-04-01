import numpy as np
import torch
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence

def generate_sentence_tensor(input_file, embedding_dict, max_length=50):

    sentence_tensors_list = []


    with open(input_file, 'r', encoding='utf-8') as f:
        for j,line in enumerate(f):

            parts = line.strip().split()



            sentence_tokens = parts[:-1]


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


    all_sentence_tensor = pad_sequence(sentence_tensors_list, batch_first=True, padding_value=0)


    output_file = 'en_dev_sentences_tensor.pt'
    torch.save(all_sentence_tensor, output_file)

    print("句子信息Tensor已保存到", output_file, "文件。")
    return output_file

f_read = open('en_dict_file.pkl', 'rb')
dict2 = pickle.load(f_read)
f_read.close()
result_file1 = generate_sentence_tensor('E_dev.txt', dict2)


