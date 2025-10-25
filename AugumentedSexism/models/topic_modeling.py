import multiprocessing

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
plt.rcParams["axes.unicode_minus"] = False

class LDAModel:
    CHINESE_STOPWORDS = []

    def __init__(self, dataset_file, output_file, stopwords_file='../models/LDA-model/stopwords.txt', lexicon_file='../models/LDA-model/SexHateLex.txt'):
        self.corpus = None
        self.dictionary = None
        self.model = None

        self.dataset = pd.read_csv(dataset_file)
        if 'text' not in self.dataset.columns:
            raise ValueError("数据集中必须包含'text'列")

        self.output_file = output_file
        with open(stopwords_file, encoding='utf-8') as f:
            for line in f:
                self.CHINESE_STOPWORDS.append(line.strip())

        if lexicon_file != '':
            jieba.load_userdict(lexicon_file)

    @staticmethod
    def preprocess_text(text):
        """预处理文本：去除特殊字符和数字"""
        # 去除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text


    def tokenize_text(self, text):
        """使用jieba分词并去除停止词"""
        # 分词
        tokens = jieba.lcut(text)
        # 过滤停止词和长度为1的词
        filtered_tokens = [
            token for token in tokens
            if token not in self.CHINESE_STOPWORDS
        ]
        return filtered_tokens

    def load_and_preprocess_data(self):
        """加载数据并进行预处理和分词"""

        # 预处理文本
        self.dataset['cleaned_text'] = self.dataset['text'].apply(self.preprocess_text)

        # 分词
        self.dataset['tokens'] = self.dataset['cleaned_text'].apply(self.tokenize_text)
        print(f"预处理完成，有效文本数量: {len(self.dataset)}")
        return self.dataset['tokens'].tolist()



    def find_optimal_topic_number(self, texts, start=2, limit=5, step=1):
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
                corpus=self.corpus,
                id2word=self.dictionary,
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
            perplexity = model.log_perplexity(self.corpus)
            perplexity_values.append(perplexity)

            # 计算一致性分数
            coherence_model = CoherenceModel(model=model, texts=texts, dictionary=self.dictionary, coherence='c_v')
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
        plt.savefig('../data/processed/lda_topic_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 找到一致性分数最高的模型
        optimal_index = coherence_values.index(max(coherence_values))
        optimal_num_topics = range(start, limit + 1, step)[optimal_index]

        print(f"最佳主题数: {optimal_num_topics}")
        self.model = model_list[optimal_index]
        return optimal_num_topics

    def print_topics(self, num_words=40):
        """打印每个主题的关键词"""
        topics = self.model.print_topics(num_words=num_words)
        for idx, topic in topics:
            print(f"主题 {idx + 1}: {topic}")
        return topics

    def modeling(self):
        """主函数：执行完整的LDA主题建模流程"""
        # 加载和预处理数据
        texts = self.load_and_preprocess_data()

        # 创建词典和语料库
        self.dictionary = corpora.Dictionary(texts)
        self.dictionary.filter_extremes(no_below=5, no_above=0.5)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]

        print(f"词典大小: {len(self.dictionary)}")
        print(f"语料库大小: {len(self.corpus)}")

        # 寻找最佳主题数
        self.find_optimal_topic_number(
            texts=texts
        )

        # 打印最佳主题
        print("\n最佳主题模型的主题关键词:")
        topics = self.print_topics(num_words=50)

        # 保存主题结果
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for idx, topic in topics:
                f.write(f"主题 {idx + 1}: {topic}\n")


def run_lda():
    model = LDAModel("../data/processed/small_data.csv", "../data/processed/topic.txt")
    model.modeling()

if __name__ == '__main__':
    run_lda()