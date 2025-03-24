import base64
import os
import shutil
import weakref
from abc import abstractmethod
from pathlib import Path

from openai import OpenAI
from typing import List, Dict, Type, final


from logger import MyLogger
from const import *
from prompt import *

LOGGER = MyLogger()
API_KEY = QWEN_API["API_KEY"]
BASE_URL = QWEN_API["BASE_URL"]
MODEL_QWEN = QWEN_API["MODEL"]
CRITIC_MODEL = QWEN_API["CRITIC_MODEL"]
MODEL_QWEN_72B = QWEN_API["MODEL-72B"]
STUPID_MODEL1 = QWEN_API["STUPID_MODEL-1"]
STUPID_MODEL2 = QWEN_API["STUPID_MODEL-2"]

API_KEY_deepseek = DeepSeek_API["API_KEY"]
BASE_URL_deepseek = DeepSeek_API["BASE_URL"]
MODEL_deepseek_v3 = DeepSeek_API["MODEL-V3"]
MODEL_deepseek_r1 = DeepSeek_API["MODEL-R1"]
MODEL_deepseek_qwq = DeepSeek_API["MODEL-QwQ"]


class FlyweightMeta(type):

    def __new__(mcs, name, parents, dct):
        dct['pool'] = weakref.WeakValueDictionary()
        return super().__new__(mcs, name, parents, dct)

    @staticmethod
    def _serialize_params(cls, *args, **kwargs):
        args_list = list(map(str, args))
        args_list.extend([str(kwargs), cls.__name__])
        key = ''.join(args_list)
        return key

    def __call__(cls, *args, **kwargs):
        key = FlyweightMeta._serialize_params(cls, *args, **kwargs)
        pool = getattr(cls, 'pool', {})

        instance = pool.get(key)
        if instance is None:
            instance = super().__call__(*args, **kwargs)
            pool[key] = instance
        return instance

class Bot:
    # __metaclass__ = ABCMeta

    def __init__(self,
                 api_key,
                 base_url,
                 max_history: int = 8) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            # base_url="https://api.deepseek.com/beta"
        )
        # self.tokenizer = qw_model.tokenizer
        # self.model = qw_model.model
        self.user_histories: Dict[str, List[Dict[str, str]]] = {}
        self.max_history = max_history

    @abstractmethod
    def generate_response(self, user_id: str, new_messages: List[Dict[str, str]], max_length: int,model:str) -> str:
        pass

    @abstractmethod
    def reset_history(self, user_id: str) -> None:
        pass

    @final
    def _prepare_history(self, user_id: str, new_messages: List[Dict[str, str]], system_prompt: Dict[str, str]) -> List[
        Dict[str, str]]:
        if user_id not in self.user_histories:
            self.user_histories[user_id] = [system_prompt]

        history = self.user_histories[user_id]
        history.extend(new_messages)
        history = [system_prompt] + history[1:][-self.max_history:]
        #print(history)
        return history

    @final
    def _generate_response(self, history: List[Dict[str, str]], max_length: int = 3000, model: str = MODEL_deepseek_v3) -> str:
        # LOGGER.info(f'history:{history}')
        # LOGGER.info(f'MODEL:{model}')
        if model == "deepseek-reasoner":
            # 过滤掉assistant角色的消息
            system_prompt = {
                "role": "user",
                "content": history[0]['content']
            }
            history = [system_prompt] + history[1:][-self.max_history:]
            history = [msg for msg in history if (msg['role'] != 'assistant' and msg['role'] != 'system')]
            message = {
                "role": "user",
                "content": ''
            }
            for m in history:
                message["content"] += m["content"]
            history = [message]
            print('reasoner_history', history)
        completion = self.client.chat.completions.create(
            model=model,
            messages=history,
            # top_p=0.97, # default 0.8
            # temperature=0.7,  # default 0.7 for qwen max
            # presence_penalty=1.1,
            max_tokens=max_length,
            # seed=1,
            # extra_body={"enable_search": True}
        )
        response = completion.choices[0].message.content
        #print(response)
        return response

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @final
    def _generate_ocr_response(self, image_path: str, model: str = MODEL_QWEN_72B) -> str:
        """增强的OCR响应生成方法"""
        LOGGER.info(f"Processing image: {image_path}")

        try:
            # 验证文件存在
            if not os.path.exists(image_path):
                raise FileNotFoundError("Image file not found")

            # 读取并验证文件内容
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                if not image_data:
                    raise ValueError("Empty image file")
                if len(image_data) > 20 * 1024 * 1024:  # 20MB限制
                    raise ValueError("Image file too large")

            # Base64编码
            base64_image = base64.b64encode(image_data).decode('utf-8')
            if not base64_image:
                raise ValueError("Base64 encoding failed")

            LOGGER.debug(f"Image encoded successfully, size: {len(base64_image)} bytes")

        except Exception as e:
            LOGGER.error(f"Image processing error: {str(e)}")
            return "图片处理失败，请检查图片格式（支持PNG/JPG/JPEG）"

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "请解析图中的数学题目"},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }]
        #处理另外两种格式图片，jpg可看作jpeg处理
        if image_path.lower().endswith('.png'):
            messages[0]['content'][1]['image_url'] = f"data:image/png;base64,{base64_image}"
        elif image_path.lower().endswith('.jpg'):
            messages[0]['content'][1]['image_url'] = f"data:image/jpeg;base64,{base64_image}"
        #删除临时保存在本地的图片, user_images/USER_ID/CHAT_ID/FILE_NAME
        try:
            shutil.rmtree(Path(image_path).parent.parent.parent)
        except Exception as e:
            LOGGER.error(f"Image deleting error: {image_path}")

        try:
            # OCR 用qwen的 ocr模型暂时
            client = OpenAI(
                api_key=API_KEY,
                base_url=BASE_URL,
            )
            # 带超时的API调用
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                # max_tokens=3000,
                timeout=30  # 30秒超时
            )
            print("*****************Response*****************")
            response = response.choices[0].message.content
            print(response)
            return response

        except TimeoutError:
            LOGGER.error("API request timed out")
            return "请求超时，请稍后再试"
        except Exception as e:
            LOGGER.error(f"API request failed: {str(e)}")
            return "服务暂时不可用，请稍后再试"

class Problem_Solver(Bot, metaclass=FlyweightMeta):

    def __init__(self, api_key: str = API_KEY, base_url: str = BASE_URL, max_history: int = 8) -> None:
        super().__init__(api_key, base_url, max_history)
        self.__repr__ = self.__str__
        self.system_prompt = {
            "role": "system",
            "content": PROBLEM_SOLVER_PROMPT
        }

    def __str__(self) -> str:
        return 'A mathematician who focuses on solving mathematical problems.'

    def generate_response(self, user_id: str, history: List[Dict[str, str]], max_length: int, model=MODEL_deepseek_r1) -> str:
        #history = self._prepare_history(user_id, new_messages, self.system_prompt)
        response: str = self._generate_response(history, max_length, model=model)
        response: str = response.replace('```markdown', '').replace('```', '')
        history.append({'role': 'assistant', 'content': response})
        self.user_histories[user_id] = history

        return response

    def reset_history(self, user_id):
        if user_id in self.user_histories:
            self.user_histories[user_id] = [self.system_prompt]

class Stupid_Student(Bot, metaclass=FlyweightMeta):

    def __init__(self, api_key: str = API_KEY_deepseek, base_url: str = BASE_URL_deepseek, max_history: int = 8) -> None:
        super().__init__(api_key, base_url, max_history)
        self.__repr__ = self.__str__
        self.system_prompt = {
            "role": "system",
            "content": STUPID_MODEL_PROMPT
        }

    def generate_response(self, user_id: str, history: List[Dict[str, str]], max_length: int,model=MODEL_deepseek_v3) -> str:
        #history = self._prepare_history(user_id, new_messages, self.system_prompt)
        response: str = self._generate_response(history, max_length, model=model)
        response: str = response.replace('```markdown', '').replace('```', '')
        #history.append({'role': 'user', 'content': response})
        self.user_histories[user_id] = history

        return response

    def reset_history(self, user_id):
        if user_id in self.user_histories:
            self.user_histories[user_id] = [self.system_prompt]

class Teaching_Assistant(Bot, metaclass=FlyweightMeta):

    def __init__(self, api_key: str = API_KEY, base_url: str = BASE_URL, max_history: int = 8) -> None:
        super().__init__(api_key, base_url, max_history)
        self.__repr__ = self.__str__
        self.system_prompt = {
            "role": "system",
            "content": TEACHING_MODEL_PROMPT
        }

    def __str__(self) -> str:
        return 'An mathematics AI teacher who helps students with mathematics learning.'

    def get_answer(self, user_id: str, questions: List[Dict[str, str]], max_length: int, PS_bot: object, model=MODEL_QWEN) -> str:
        history = PS_bot._prepare_history(user_id, questions, PS_bot.system_prompt)
        # response: str = MPS_bot._generate_response(history, max_length, model=model)
        response: str = self._generate_response(history, max_length, model=model)
        response: str = response.replace('```markdown', '').replace('```', '')
        history.append({'role': 'assistant', 'content': response})
        self.user_histories[user_id] = history
        return response

    def generate_response(self, user_id: str, history: List[Dict[str, str]], max_length: int, model=MODEL_QWEN) -> str:
        #history = self._prepare_history(user_id, new_messages, self.system_prompt)
        response: str = self._generate_response(history, max_length, model)
        response: str = response.replace('```markdown', '').replace('```', '')
        history.append({'role': 'assistant', 'content': response})
        self.user_histories[user_id] = history

        return response

    def reset_history(self, user_id):
        if user_id in self.user_histories:
            self.user_histories[user_id] = [self.system_prompt]

class Education_Expert(Bot, metaclass=FlyweightMeta):
    def __init__(self, api_key: str = API_KEY, base_url: str = BASE_URL, max_history: int = 8) -> None:
        super().__init__(api_key, base_url, max_history)
        # self.user_histories: Dict[str, List[Dict[str, str]]] = {}
        self.__repr__ = self.__str__
        self.system_prompt_error_locating = {
            "role": "system",
            "content": Education_Expert_Error_Locating_PROMPT
        }
        self.system_prompt_suggestions = {
            "role": "system",
            "content": Education_Expert_Suggestions_PROMPT
        }

    def get_critic_error_locating(self, user_id: str, new_messages: List[Dict[str, str]], max_length: int, MTA_bot: object, model=CRITIC_MODEL) -> str:
        # history = MTA_bot._prepare_history(user_id, new_messages, self.system_prompt_error_locating)
        # print("*****************Critic History*****************")
        # print(history)
        history = [self.system_prompt_error_locating] + new_messages
        response: str = self._generate_response(history, max_length, model)
        history.append({'role': 'assistant', 'content': response})
        self.user_histories[user_id] = history
        return response

    def get_critic_suggestions(self, user_id: str, new_messages: List[Dict[str, str]], max_length: int, MTA_bot: object, model=CRITIC_MODEL) -> str:
        # history = MTA_bot._prepare_history(user_id, new_messages, self.system_prompt_suggestions)
        # print("*****************Critic History*****************")
        # print(history)
        history = [self.system_prompt_suggestions] + new_messages
        response: str = self._generate_response(history, max_length, model)
        history.append({'role': 'assistant', 'content': response})
        self.user_histories[user_id] = history
        return response

    def reset_history(self, user_id):
        if user_id in self.user_histories:
            self.user_histories[user_id] = [self.system_prompt_suggestions]

class VL_OCR_Bot():
    def __init__(self):
        self.api_key = API_KEY
        self.base_url = BASE_URL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_response_with_VL(self, image_path):
        base64_image = self.encode_image(image_path)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": VL_Model_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                ],
            }
        ]
        if image_path.lower().endswith('.png'):
            messages[1]['content'][0]['image_url'] = f"data:image/png;base64,{base64_image}"
        elif image_path.lower().endswith('.jpg'):
            messages[1]['content'][0]['image_url'] = f"data:image/jpeg;base64,{base64_image}"
        completion = self.client.chat.completions.create(
            #model="qwen-vl-max-latest",
            model='qwen-vl-max-2025-01-25',
            messages=messages
        )
        return completion.choices[0].message.content.split("---")


class BotShop(object):

    def __init__(self, bot_cls: Type[Bot]) -> None:
        self.bot_cls = bot_cls

    def buy_bot(self, api_key: str = API_KEY, base_url: str = BASE_URL, max_history: int = 8) -> Bot:
        bot = self.bot_cls(api_key, base_url, max_history)
        return bot