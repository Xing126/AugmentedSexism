from typing import List, Tuple
import requests
import pandas as pd
import numpy as np
import time


class APIClient:
    """豆包API的客户端封装"""

    def __init__(self, api_key, api_url, model):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model

    def call(self, prompt: str) -> str:
        """调用大语言模型API获取结果"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 构建请求体（符合豆包API格式）
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,  # 确定性输出
            "max_tokens": 10  # 限制输出长度
        }
        for _ in range(5):
            try:
                response = requests.post(self.api_url, headers=headers, json=data)
                response.raise_for_status()  # 检查HTTP错误
                response_data = response.json()

                # 解析返回结果（根据API实际响应格式调整）

                response = response_data["choices"][0]["message"]["content"].strip()
                if len(response) <= 2:
                    return response

                else:
                    print(response)
                    time.sleep(5)

            except Exception as e:
                print(f"豆包API调用失败: {str(e)}")
                if 'response' in locals():
                    print(f"API响应: {response.text}")
                time.sleep(5)

        return ''


class TextAnnotator:
    """文本标注的核心类，整合示例抽取和API调用"""

    def __init__(self, api_client : APIClient, train_dataset_path, test_dataset_path, output_folder, indices_path=None):
        self.api_client = api_client
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.output_folder = output_folder

        if indices_path:
            self.indices = np.load(indices_path)
        else:
            raise FileNotFoundError("没有发现名字为indices的示例参考文件")



    def extract_indices(self, index, task) -> List[Tuple[str, str]]:
        """
        提取示例的方法
        :param index: 文本编号/索引
        :param task: 任务类型
        :return: 提取的示例
        """

        def find_keywords(num, task):
            for key,value in TaskRules.RULES[task].items():
                if num == value[0]:
                    return key

        # 最佳的n个示例
        dataset_path = self.train_dataset_path
        train_data = pd.read_csv(dataset_path)
        result_5 = train_data.iloc[self.indices[index]]

        # 任务的编号对应意思
        labeled_dataset = []
        for index, row in result_5.iterrows():
            data = (row["text"], find_keywords(row[task], task))
            labeled_dataset.append(data)

        return labeled_dataset



    def annotate(self, index, input_text, task):
        """
        标注文本的主方法
        :param index: 文本编号/索引
        :param input_text: 待标注文本
        :param task: 任务类型
        :return: 标注结果字典
        """
        # 校验任务合法性
        if task not in TaskRules.TASK_DESCRIPTIONS:
            raise ValueError(f"无效任务类型，可选任务: {list(TaskRules.TASK_DESCRIPTIONS.keys())}")

        # 提取示例
        # examples = self.extract_indices(index, task)
        examples = None

        # 构建提示词
        prompt = self._build_prompt(task, input_text, examples=examples)
        print(prompt)

        # 调用API获取结果
        result = self.api_client.call(prompt)

        # 返回完整结果
        return {
            "task": task,
            "result": result,
        }


    def _build_prompt(self, task, input_text, examples=None):
        """构建带示例的提示词"""
        prompt_parts = [TaskRules.TASK_DESCRIPTIONS[task]]

        # 添加示例
        if examples:
            prompt_parts.append("\n参考以下示例：")
            for i, (text, label) in enumerate(examples, 1):
                prompt_parts.append(f"示例{i}：")
                prompt_parts.append(f"句子：{text}")
                prompt_parts.append(f"判断：{label}")

        else:
            pass

        # 添加待标注文本
        prompt_parts.append("\n请判断：")
        prompt_parts.append(f"句子：{input_text}")
        prompt_parts.append("判断：")

        return "\n".join(prompt_parts)




class TaskRules:

    # 任务描述映射
    TASK_DESCRIPTIONS = {
        'sexism': """性别歧视，满足以下任意一点就为性别歧视：
                   1.明确使用性别歧视语言攻击或侮辱特定性别群体或个人。
                   2.煽动基于性别的暴力或宣扬性别仇恨，但未直接使用性侮辱语言。
                   3.辱骂那些攻击或对某一性别群体持负面态度者。
                   4.对性侵、性取向及性骚扰等恶性事件或意图表示支持。
                   5.通过物化外貌、过度简化形象或宣扬男性优越论来强化对性别群体的负面刻板印象。
                   6.以讽刺或隐晦方式表达潜在的性别偏见。
                   我给定一句话，你需要判断这句话是否包含性别歧视，如果这句话包含性别歧视，你需要回答‘是’，否则回答'否'。""",

        'gender': "请判断以下句子评价对象的性别，仅回答'男'或'女'；如果两个性别都涉及了请仅回复‘都’。",
        'hostile': "你需要判断其表述方式，若语句具有攻击性、使用贬损性性别词汇或包含威胁内容，就回答“敌意”，反之则标记为“温和”",

        'misogyny': "厌女，指的是针对女性的憎恨、厌恶及偏见。请判断以下句子是否包含厌女内容，仅回答'是'或'否'。",
        'misandry': "厌男，是指对男性的仇恨、贬抑、歧视。请判断以下句子是否包含厌男内容，仅回答'是'或'否'。",

    }

    # 规则格式：{当前任务: {当前结果: [后续任务列表]}}
    RULES = {

        # 若性别歧视判断为"是"，则进一步判断敌意程度和具体歧视类型
        'sexism': {
            '是': (1, 'hostile'),
            '否': (0, 'hostile')
        },

        # 若判断为敌意表述，进一步判断是否有性别歧视
        'hostile': {
            '敌意': (1, 'gender'),
            '温和': (0, 'gender')
        },

        # 判断针对的性别， 进一步判断是否有敌意
        'gender': {
            '男': (1, ''),
            '女': (0, ''),
            '都': (2, ''),
        },

        # 判断厌女
        'misogyny': {
            '是': (1, 'misandry'),
            '否': (0, 'misandry')
        },


        # 判断厌男
        'misandry': {
            '是': (1, ''),
            '否': (0, '')
        },
    }

    @classmethod
    def get_next_task(cls, current_task: str, current_result: str) -> tuple[int, str]:
        """根据当前任务和结果，获取后续任务列表"""
        return cls.RULES.get(current_task, {}).get(current_result, [])



class TaskPublisher:

    def __init__(self, annotator: TextAnnotator, save_path):
        self.annotator = annotator
        self.unsuccessful_index = []
        self.save_path = save_path
        try:
            self._dataset = pd.read_csv(self.save_path)
            self.length = len(self._dataset)
        except FileNotFoundError or AttributeError:
            self._dataset = pd.DataFrame(columns=["ID", "text", "sexism", "gender", "hostile", "misogyny", "misandry"])
            self.length = 0



    def task_series_publish(self, text:str, index, start_task="sexism"):
        row = pd.DataFrame([[-1]*7], columns=["ID","text","sexism","gender","hostile","misogyny","misandry"])
        row["text"] = text
        row["ID"] = index

        def _publish_next_tasks(task, index, row):

            # 提取AI判断结果
            result_dict = self.annotator.annotate(index, text, task)
            _result = result_dict["result"]
            print(f"{task}->{_result}")

            if _result:

                # 根据规则获取后续任务
                num_result, next_task = TaskRules.get_next_task(task, _result or '')
                row[task] = num_result
                if next_task:
                    return _publish_next_tasks(next_task, index, row)
                else:
                    print(row)
                    print(f"{text[:4]} 已完成所有处理")
                    return row

            else:
                return pd.DataFrame()


        row = _publish_next_tasks(start_task, index, row)

        if not row.empty:
            self._dataset = pd.concat([self._dataset, row], ignore_index=True).fillna(-1)
        else:
            self.unsuccessful_index.append(index)



    def text_series_publish(self, text:str, index, task):
        row = pd.DataFrame([[-1]*2], columns=["ID",task])
        row["text"] = text
        row["ID"] = index

        # 提取AI判断结果
        result_dict = self.annotator.annotate(index, text, task)
        _result = result_dict["result"]
        print(f"{task}->{_result}")

        if _result:
            num_result, next_task = TaskRules.get_next_task(task, _result or '')
            row[task] = num_result

            self._dataset = pd.concat([self._dataset, row], ignore_index=True).fillna(-1)
        else:
            self.unsuccessful_index.append(index)



    def batch_publish(self, test_path, method, task=None):
        try:
            texts = pd.read_csv(test_path)
            texts = texts[self.length:]
            count = 0
            for index, row in texts.iterrows():
                if method == "task":
                    self.task_series_publish(row["text"], index, task)
                elif method == "text" and task:
                    self.text_series_publish(row["text"], index, task)

                count += 1
                print(count)


        except KeyboardInterrupt:
            pass

        # except KeyboardInterrupt as e:
        #     self._dataset.to_csv(self.save_path, index=False)
        #     print(e)
        #     print(self._dataset)
        #     print(self.unsuccessful_index)

        self._dataset.to_csv(self.save_path, index=False)
        print(self._dataset)
        print(self.unsuccessful_index)


    def get_prompt(self, target, task, texts : pd.DataFrame):
        for index, row in texts.iterrows():
            if target == row["text"]:
                try:
                    self.task_series_publish(row["text"], index, start_task=task)
                except TypeError:
                    pass

    def get_dataset(self):
        return self._dataset


if __name__ == "__main__":

    # 初始化标注器
    api_client = APIClient(
        api_url="https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        api_key="fb9f6828-c8d7-4f87-a038-6145b078a48b",
        model="doubao-1-5-lite-32k-250115")


    annotator = TextAnnotator(api_client,
                              train_dataset_path="../data/compairison/train.csv",
                              test_dataset_path="../data/compairison/test.csv",
                              output_folder="../data/compairison/",
                              )

    publisher = TaskPublisher(annotator,
                              save_path="../data/compairison/test_extend_no_examples.csv")

    publisher.batch_publish(test_path="../data/compairison/test.csv",
                            method="task",
                            task="sexism")