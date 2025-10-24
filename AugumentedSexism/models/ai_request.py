import requests
import pandas as pd
import time


class APIClient:
    def __init__(self, api_key, api_url, model):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model

    TASK_DESCRIPTIONS = {
        'gender': "请判断以下句子评价对象的性别，仅回答'男'或'女'；如果两个性别都涉及了请仅回复‘都’。",
        'sexism': """性别歧视，满足以下任意一点就为性别歧视：
                       1.明确使用性别歧视语言攻击或侮辱特定性别群体或个人。
                       2.煽动基于性别的暴力或宣扬性别仇恨，但未直接使用性侮辱语言。
                       3.辱骂那些攻击或对某一性别群体持负面态度者。
                       4.对性侵、性取向及性骚扰等恶性事件或意图表示支持。
                       5.通过物化外貌、过度简化形象或宣扬男性优越论来强化对性别群体的负面刻板印象。
                       6.以讽刺或隐晦方式表达潜在的性别偏见。
                       我给定一句话，你需要判断这句话是否包含性别歧视，如果这句话包含性别歧视，你需要回答‘是’，否则回答'否'。""",
        'hostile': "你需要判断其表述方式，若语句具有攻击性、使用贬损性性别词汇或包含威胁内容，就回答“敌意”，反之则标记为“温和”",
        'misogyny': "厌女，指的是针对女性的憎恨、厌恶及偏见。请判断以下句子是否包含厌女内容，仅回答'是'或'否'。",
        'misandry': "厌男，是指对男性的仇恨、贬抑、歧视。请判断以下句子是否包含厌男内容，仅回答'是'或'否'。",

    }

    RULES = {
        'gender': {'男': 1, '女': 0, '都': 2},
        'hostile': {'敌意': 1, '温和': 0},
        'sexism': {'是': 1, '否': 0},
        'misogyny': {'是': 1, '否': 0},
        'misandry': {'是': 1, '否': 0},
    }

    RULES_INVERTED = {
        'gender': {1: '男', 0: '女', 2: '都'},
        'hostile': {1: '敌意', 0: '温和'},
        'sexism': {1: '是', 0: '否'},
        'misogyny': {1: '是', 0: '否'},
        'misandry': {1: '是', 0: '否'}
    }

    def _call(self, prompt: str) -> str:
        """get response from API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,  # Output certainly
            "max_tokens": 10  # maximum length
        }
        for _ in range(5):
            try:
                response = requests.post(self.api_url, headers=headers, json=data)
                response.raise_for_status()  # check HTTP ERROR
                response_data = response.json()

                # return ai response text
                response = response_data["choices"][0]["message"]["content"].strip()
                if len(response) <= 2:
                    return response

                else:
                    print(response)
                    time.sleep(3)

            except Exception as e:
                print(f"False to access AI API: {str(e)}")
                if 'response' in locals():
                    print(f"API response: {response.text}")
                time.sleep(3)

        return ''


    def _handle(self, rule, response, unlabeled_dict) -> dict:
        unlabeled_dict[rule] = self.RULES[rule][response]
        return unlabeled_dict


    def _build_prompt(self, rule, test_text, examples=None):
        """构建带示例的提示词"""
        prompt_parts = [self.TASK_DESCRIPTIONS[rule]]

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
        prompt_parts.append(f"句子：{test_text}")
        prompt_parts.append("判断：")

        return "\n".join(prompt_parts)

    def api_quest(self, rule, test_text, examples, unlabeled_dict) -> dict:
        _prompt = self._build_prompt(rule, test_text, examples)
        _response = self._call(_prompt)
        if _response:
            labeled_dict = self._handle(rule, _response, unlabeled_dict)
            return labeled_dict
        return {}



class ExampleProducer:
    def __init__(self, indices, train_path):
        self.indices = indices
        self.train_dataset = pd.read_csv(train_path)

    def produce_examples(self, rule, index):
        result_5 = self.train_dataset.iloc[self.indices[index]]

        # 任务的编号对应意思
        examples = []
        for _, row in result_5.iterrows():
            data = (row["text"], APIClient.RULES_INVERTED[rule][row[rule]])
            examples.append(data)
        return examples


class TextProducer:
    def __init__(self, test_path):
        self.test_dataset = pd.read_csv(test_path)
        self.length = self.test_dataset.shape[0]

    def produce_texts(self, index):
        return self.test_dataset.iloc[index]['text']


class TextHandler:
    def __init__(self, output_path):
        self.text = ''
        try:
            self._dataset = pd.read_csv(output_path)
            self.length = len(self._dataset)
        except FileNotFoundError or AttributeError:
            self._dataset = pd.DataFrame(columns=["ID", "text", "sexism", "gender", "hostile", "misogyny", "misandry"])
            self.length = 0

    LABELED_DICT = {
    "ID": 0,
    "text": "",
    "sexism": -1,
    "gender": -1,
    "hostile": -1,
    "misogyny": -1,
    "misandry": -1
}

    def concat_text(self, labeled_dict, index, text):
        labeled_dict["ID"] = index
        labeled_dict["text"] = text
        self._dataset = self._dataset.append(labeled_dict)

    def get_dataset(self):
        return self._dataset


class AugumentedSexismModel:
    def __init__(self,
                 example_producer: ExampleProducer,
                 text_producer: TextProducer,
                 api_client: APIClient,
                 text_handler: TextHandler,
                 rules:list, start_index=0):

        self.index = start_index
        self.rules = rules
        self.rules_count = 0
        self.unsuccessful_indexes = []
        self.example_producer = example_producer
        self.text_producer = text_producer
        self.api_client = api_client
        self.text_handler = text_handler

    def yield_rules(self):
        if self.rules_count == len(self.rules):
            self.rules_count = 0
            return ''

        rule = self.rules[self.rules_count]
        self.rules_count += 1
        return rule

    def main(self):
        total_texts = self.text_producer.length
        for index in range(total_texts):
            test_text = self.text_producer.produce_texts(index)

            labeled_dict = self.text_handler.LABELED_DICT
            while True:
                rule = self.yield_rules()
                if rule == '':
                    break
                examples = self.example_producer.produce_examples(rule, index)
                labeled_dict = self.api_client.api_quest(rule, test_text, examples)
                if not labeled_dict:
                    self.unsuccessful_indexes.append(index)
                    break

            self.text_handler.concat_text(labeled_dict, index, test_text)

        return self.text_handler.get_dataset()