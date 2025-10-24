import os
import sys
import pandas as pd
from safetensors.torch import load_file
from src.model import BertMultilabelModel
from transformers import BertTokenizerFast
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------  CONFIGURATION  -------------------------------- #
model_dir = f"./models/BERT-model/"
text_column = 'text'  # 输入文件的中文列名
batch_size = 32
num_labels = 3

# --------------------------------------------------------------------------------- #

def load_train_data(file_path):
    """加载train.csv数据并清洗，确保每条文本都被处理"""
    global text_column

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        df = pd.read_csv(f)

    if text_column not in df.columns:
        raise ValueError(f"train.csv必须包含文本列 '{text_column}'")
    
    texts = []
    for text in df[text_column]:
        if not isinstance(text, str):
            text = str(text)
        texts.append(text.strip())
    
    assert len(texts) == len(df), f"处理后文本数量({len(texts)})与原始数据({len(df)})不匹配"
    return texts



def dataset_vectorize(train_data_path, output_vector_file):
    global model_dir
    model_filename = "model.safetensors"
    model_weight_path = os.path.join(model_dir, model_filename)
    global batch_size, num_labels


    # 检查模型文件是否存在
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型文件夹不存在: {model_dir}")

    if not os.path.exists(model_weight_path):
        files_in_dir = os.listdir(model_dir)
        raise FileNotFoundError(
            f"模型权重文件不存在: {model_weight_path}\n"
            f"模型文件夹内的文件有: {files_in_dir}"
        )

    # 加载分词器
    print(f"加载分词器...")
    tokenizer = BertTokenizerFast.from_pretrained(
        model_dir,
        local_files_only=True
    )

    print(tokenizer)
    # 加载模型（关键修改：加载具体的权重文件）
    print(f"加载模型: {model_weight_path}")
    model = BertMultilabelModel(
        model_name=model_dir,  # 模型配置从文件夹加载
        tokenizer=tokenizer,
        num_labels=num_labels
    )

    # 加载safetensors格式的权重
    state_dict = load_file(model_weight_path, device='cpu')
    model.load_state_dict(state_dict)

    # 加载train.csv所有文本
    print(f"加载训练数据: {train_data_path}")
    train_texts = load_train_data(train_data_path)
    print(f"共加载 {len(train_texts)} 条训练文本")

    # 生成所有文本的向量
    print("开始生成训练文本向量...")
    train_vectors = model.generate_text_vector_group(
        texts=train_texts,
        batch_size=batch_size,
        save_path=output_vector_file
    )

    # 验证向量数量与文本数量一致
    assert len(train_vectors) == len(train_texts), \
        f"生成的向量数量({len(train_vectors)})与文本数量({len(train_texts)})不匹配"

    print(f"训练文本向量生成完成！")
    print(f"向量形状: {train_vectors.shape}（行数={len(train_vectors)}，维度={train_vectors.shape[1]}）")
    print(f"向量已保存至: {output_vector_file}")






if __name__ == "__main__":
    # ------------------- 配置参数（关键修改：指定具体权重文件） -------------------
    train_data_path = "../data/raw/all_data.csv"
    output_vector_file = "../../data/processed/train_vectors.npy"

    dataset_vectorize(train_data_path, output_vector_file)
