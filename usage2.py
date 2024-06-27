from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# 定义输入内容
sentences = [
    "天气好热，哪里有卖冰棍的",
    "今天好冷，该多穿两件",
    "夏天",
    "冬天"
]

# 指定预训练模型，在线模型 "thenlper/gte-base-zh"，此处使用本地目录中的模型
model_id = "./thenlper/gte-small"

# 加载 SentenceTransformer 模型
model = SentenceTransformer(model_id)
# 获取向量
embeddings = model.encode(sentences)
# 计算第一个句子（index 0）和第二个句子（index 1）的嵌入向量之间的余弦相似度，并打印结果
print(cos_sim(embeddings[0], embeddings[1]))
