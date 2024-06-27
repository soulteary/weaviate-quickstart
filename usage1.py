import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# 定义输入内容
input_texts = [
    "天气好热，哪里有卖冰棍的",
    "今天好冷，该多穿两件",
    "夏天",
    "冬天"
]

# 指定预训练模型，在线模型 "thenlper/gte-base-zh"，此处使用本地目录中的模型
model_id = "./thenlper/gte-small"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

# 对输入内容进行分词、编码
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

# 获取 Embeddings
outputs = model(**batch_dict)
embeddings = outputs.last_hidden_state[:, 0]
 
# 标准化 embeddings，使用 L2 归一化，使其长度为 1
embeddings = F.normalize(embeddings, p=2, dim=1)
# 计算相似度，选择第一个元素，和除了第一个元素进行比较
scores = (embeddings[:1] @ embeddings[1:].T) * 100
# 打印结果分数
print(scores.tolist())
