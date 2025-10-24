import pandas as pd
import jieba
import re
import matplotlib.pyplot as plt
from gensim import corpora, models
from gensim.models import CoherenceModel
import warnings
import seaborn as sns
import numpy as np
warnings.filterwarnings('ignore')

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 中文停止词列表
CHINESE_STOPWORDS = []
with open("./cn_stopwords.txt", encoding='utf-8') as f:
    for line in f:
        CHINESE_STOPWORDS.append(line.strip())

print(CHINESE_STOPWORDS)

jieba.load_userdict("./SexHateLex.txt")


def preprocess_text(text):
    """预处理文本：去除特殊字符和数字"""
    # 去除特殊字符和数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    """使用jieba分词并去除停止词"""
    # 分词
    tokens = jieba.lcut(text)
    # 过滤停止词和长度为1的词
    filtered_tokens = [
        token for token in tokens 
        if token not in CHINESE_STOPWORDS
    ]
    return filtered_tokens

def load_and_preprocess_data(csv_path):
    """加载数据并进行预处理和分词"""
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        if 'text' not in df.columns:
            raise ValueError("CSV文件中必须包含'text'列")
        
        # 预处理文本
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        
        # 分词
        df['tokens'] = df['cleaned_text'].apply(tokenize_text)

        
        print(f"加载并预处理完成，有效文本数量: {len(df)}")
        return df['tokens'].tolist()
    
    except Exception as e:
        print(f"数据处理错误: {str(e)}")
        return None

def set_model(corpus, dictionary, num_topics=5):
    model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    return model, num_topics

def find_optimal_topic_number(corpus, dictionary, texts, start=2, limit=5, step=1):
    """
    寻找最佳主题数
    通过计算困惑度和一致性分数来确定
    """
    perplexity_values = []
    coherence_values = []
    model_list = []
    
    for num_topics in range(start, limit + 1, step):
        # 训练LDA模型
        model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=100,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        model_list.append(model)
        
        # 计算困惑度
        perplexity = model.log_perplexity(corpus)
        perplexity_values.append(perplexity)
        
        # 计算一致性分数
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()
        coherence_values.append(coherence)
        
        print(f"主题数: {num_topics}, 困惑度: {perplexity:.4f}, 一致性分数: {coherence:.4f}")

    # 设置seaborn风格
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

    # 假设以下是你的数据（实际使用时替换为你的真实数据）
    topics_range = range(start, limit + 1, step)

    # 创建图表
    plt.figure(figsize=(12, 7))
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # 困惑度图表（值越低越好）
    color = sns.color_palette("deep")[0]
    ax1.set_xlabel('Number of Topics', fontsize=14)
    ax1.set_ylabel('Perplexity', color=color, fontsize=14)
    sns.lineplot(x=list(topics_range), y=perplexity_values, ax=ax1,
                 color=color, marker='o', linewidth=2.5, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)

    # 一致性分数图表（值越高越好）
    ax2 = ax1.twinx()
    color = sns.color_palette("deep")[1]
    ax2.set_ylabel('Coherence Score', color=color, fontsize=14)
    sns.lineplot(x=list(topics_range), y=coherence_values, ax=ax2,
                 color=color, marker='s', linewidth=2.5, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)

    # 添加网格线使读数更容易
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 添加标题和优化布局
    plt.title('LDA Model Performance by Number of Topics', fontsize=16, pad=20)
    fig.tight_layout()

    # 保存和显示图表
    plt.savefig('lda_topic_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 找到一致性分数最高的模型
    optimal_index = coherence_values.index(max(coherence_values))
    optimal_num_topics = range(start, limit + 1, step)[optimal_index]
    
    print(f"最佳主题数: {optimal_num_topics}")
    
    return model_list[optimal_index], optimal_num_topics

def print_topics(model, num_words=40):
    """打印每个主题的关键词"""
    topics = model.print_topics(num_words=num_words)
    for idx, topic in topics:
        print(f"主题 {idx + 1}: {topic}")
    return topics

def main(csv_path, output_topics_path=None):
    """主函数：执行完整的LDA主题建模流程"""
    # 加载和预处理数据
    texts = load_and_preprocess_data(csv_path)
    if not texts:
        return
    
    # 创建词典和语料库
    dictionary = corpora.Dictionary(texts)
    # 过滤极端词（出现太少或太多的词）
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    print(f"词典大小: {len(dictionary)}")
    print(f"语料库大小: {len(corpus)}")
    
    # 寻找最佳主题数
    optimal_model, optimal_num_topics = find_optimal_topic_number(
        corpus=corpus,
        dictionary=dictionary,
        texts=texts,
        start=2,
        limit=5,
        step=1
    )

    # optimal_model, optimal_num_topics = set_model(corpus, dictionary, num_topics=5)
    
    # 打印最佳主题
    print("\n最佳主题模型的主题关键词:")
    topics = print_topics(optimal_model, num_words=50)
    
    # 保存主题结果
    if output_topics_path:
        with open(output_topics_path, 'w', encoding='utf-8') as f:
            for idx, topic in topics:
                f.write(f"主题 {idx + 1}: {topic}\n")
        print(f"主题结果已保存到 {output_topics_path}")
    
    return optimal_model, dictionary, corpus

if __name__ == "__main__":
    # 示例用法
    input_csv = "../data/lda_data/all_data.csv"  # 输入CSV文件路径
    output_topics = "../data/lda_data/lda_topics_all_data.txt"  # 输出主题结果路径
    main(input_csv, output_topics)
    