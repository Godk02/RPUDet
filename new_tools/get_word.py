# import torch
# import torchtext.vocab as vocab
# import numpy as np
# import pickle

# # 计算余弦相似度
# def Cos(x, y):
#     cos = torch.matmul(x, y.view((-1,))) / (
#             (torch.sum(x * x) + 1e-9).sqrt() * torch.sum(y * y).sqrt())
#     return cos


# if __name__ == '__main__':
#     # 在类别列表中保留原始名称，使用单个词表示每个类别
#     classes = [
#         'hanging insulator', 
#         'post insulator', 
#         'column insulator', 
#         'wing insulator', 
#         # 'Isolation knife', 
#         # 'Transformer', 
#         # 'Pole'
#     ]
    
#     # 为每个类别分配一个单词表示，便于查找词向量
#     word_map = {
#         'hanging insulator': 'hanging insulator', 
#         'post insulator': 'post insulator', 
#         'column insulator': 'column insulator', 
#         'wing insulator': 'wing insulator', 
#         # 'Isolation knife': 'device',  # 这里将 "Isolation knife" 归为 "device"
#         # 'Transformer': 'transformer', 
#         # 'Pole': 'pole'
#     }
    
#     total = np.array([])
#     glove = vocab.GloVe(name="42B", dim=300)

#     for cls in classes:
#         # 使用单个关键词查找词向量
#         keyword = word_map[cls]  # 获取该类别对应的单词
#         if keyword in glove.stoi:
#             a = glove.vectors[glove.stoi[keyword]]
#             total = np.append(total, a.numpy())
#         else:
#             print(f"'{keyword}' not found in GloVe vocabulary.")

#     total = total.reshape(len(classes), -1)
    
#     # 保存对应类别的word embedding，保持原类别名称不变
#     pickle.dump(total, open('new_wordEmbedding.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

import torch
import clip
import numpy as np
import pickle

# 计算余弦相似度
def Cos(x, y):
    cos = torch.matmul(x, y.view((-1,))) / (
            (torch.sum(x * x) + 1e-9).sqrt() * torch.sum(y * y).sqrt())
    return cos

if __name__ == '__main__':
    # 类别列表，保留原始名称
    classes = [
        # 'hanging insulator',
        # 'post insulator',
        # 'column insulator',
        # 'wing insulator'
    #  'pedestrian',
    # 'people',
    # 'bicycle',
    # 'car',
    # 'van',
    # 'truck',
    # 'tricycle',
    # 'awning-tricycle',
    # 'bus',
    # 'motor', 
    #  'car', 'pickup', 'camping','truck', 'other', 'tractor', 'boat', 'van' 

    # 'plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court', 
    # 'ground track field', 'harbor', 'bridge', 'large vehicle', 'small vehicle', 'helicopter', 'roundabout', 'soccer ball field', 'swimming pool'
    'vehicle','vehicle'


    ]

    # 加载 CLIP 模型和预处理器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 将类别名称转换为 CLIP 支持的文本嵌入
    with torch.no_grad():
        text_inputs = clip.tokenize(classes).to(device)  # 将类别名称转换为 tokens
        text_features = model.encode_text(text_inputs)   # 获取文本嵌入

    # 将文本嵌入转换为 numpy 数组
    total = text_features.cpu().numpy()

    # 保存对应类别的词嵌入，保持原类别名称不变
    with open('usod_clip.pkl', 'wb') as f:
        pickle.dump(total, f, pickle.HIGHEST_PROTOCOL)

    print("Word embeddings have been successfully generated and saved.")

