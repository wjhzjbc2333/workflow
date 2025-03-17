import time

from openai import OpenAI
import os
from pathlib import Path
import base64
import numpy as np
from prompt import *
from chatbot_api import *



# #  读取本地文件，并编码为 BASE64 格式
# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")
#
#
# base64_image = encode_image("test.png")
# client = OpenAI(
#     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
#     api_key="sk-30b325ff84cd484991a8e7a7777b63eb",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# completion = client.chat.completions.create(
#     model="qwen-vl-ocr",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image_url",
#                     # 需要注意，传入BASE64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
#                     # PNG图像：  f"data:image/png;base64,{base64_image}"
#                     # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
#                     # WEBP图像： f"data:image/webp;base64,{base64_image}"
#                     "image_url": {"url": f"data:image/png;base64,{base64_image}"},
#                     "min_pixels": 28 * 28 * 4,
#                     "max_pixels": 28 * 28 * 1280
#                 },
#                 # 为保证识别效果，目前模型内部会统一使用"Read all the text in the image."进行识别，用户输入的文本不会生效。
#                 {"type": "text", "text": "Read all the text in the image."},
#             ],
#         }
#     ],
# )
# print(completion.choices[0].message.content)


'''Qwen-long输出文档内容'''
'''纯文本或json文件可能可用，直接识别pdf或word文档由于数学公式和占位符效果较差'''

# client = OpenAI(
#     api_key="sk-30b325ff84cd484991a8e7a7777b63eb",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务base_url
# )
#
# file_object = client.files.create(file=Path("test.pdf"), purpose="file-extract")
# print(file_object.id)
#
#
# completion = client.chat.completions.create(
#     model="qwen-long",
#     messages=[
#         {'role': 'system', 'content': '你是识别文档内容的人工智能助手，请将文档的全部内容按原本的排列方式输出。'},
#         # 请将 'file-fe-xxx'替换为您实际对话场景所使用的 file-id。
#         {'role': 'system', 'content': f'fileid://{file_object.id}'},
#         {'role': 'user', 'content': ''}
#     ],
#     stream=True,
#     stream_options={"include_usage": True}
# )
#
# full_content = ""
# for chunk in completion:
#     if chunk.choices and chunk.choices[0].delta.content:
#         # 拼接输出内容
#         full_content += chunk.choices[0].delta.content
#         print(chunk.model_dump())
#
# print({full_content})

'''各种Qwen-VL模型'''
'''可用，效果不错，能帮忙分割题目'''
# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")
#
# # 将xxxx/test.png替换为你本地图像的绝对路径
# base64_image = encode_image("test1.png")
# client = OpenAI(
#     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
#     api_key="sk-30b325ff84cd484991a8e7a7777b63eb",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# completion = client.chat.completions.create(
#     model="qwen-vl-max-latest",
#     messages=[
#     	{
#     	    "role": "system",
#             "content": [{"type":"text","text": "你是用于识别图像内容的人工智能助手。\n请识别图中的数学题目和公式，并按照相同的格式返回文本内容。\n输出时清删去公式中的转义符和占位符。\n在有多道题目时，请在题目之间加入“---”作为分割线。"}]},
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image_url",
#                     # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
#                     # PNG图像：  f"data:image/png;base64,{base64_image}"
#                     # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
#                     # WEBP图像： f"data:image/webp;base64,{base64_image}"
#                     "image_url": {"url": f"data:image/png;base64,{base64_image}"},
#                 },
#                 {"type": "text", "text": ""},
#             ],
#         }
#     ],
# )
# print(completion.choices[0].message.content)

'''阿里云-文字识别下教育场景识别'''
'''整页试卷识别-勉强能用 试卷切题识别-不好用 题目识别-也很一般'''
'''使用体验完全不如Qwen-VL'''


'''deepseek-v3'''
# client = OpenAI(
#     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
#     api_key="sk-30b325ff84cd484991a8e7a7777b63eb",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# )
#
# completion = client.chat.completions.create(
#     model="deepseek-v3",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
#     messages=[
#         {'role': 'user', 'content': '1. (2024四川广安) 代数式 -3x 的意义可以是（）\nA. -3 与 x 的和\nB. -3 与 x 的差\nC. -3 与 x 的积\nD. -3 与 x 的商'}
#     ]
# )
#
# # 通过reasoning_content字段打印思考过程
# print("思考过程：")
# print(completion.choices[0].message.reasoning_content)
#
# # 通过content字段打印最终答案
# print("最终答案：")
# print(completion.choices[0].message.content)

'''硅基流动-deepseek r1'''
# from openai import OpenAI
#
# client = OpenAI(
#     base_url='https://api.siliconflow.cn/v1',
#     api_key='sk-ytlhumlrgvmwkisiirugbjnsvwztwmdqbwqszghjfumyzrbw'
# )
#
# # 发送带有流式输出的请求
# response = client.chat.completions.create(
#     model="Qwen/QwQ-32B",
#     messages=[{'content': STUPID_MODEL_PROMPT, 'role': 'system'},
#               {'content': '计算并化简(2x + 3y)^2  - (2x - 3y)(2x + 3y) - (3y - 2x)^2', 'role': 'system'},
#               {'content': '难度：中等\n我们先从第一个部分开始，计算并展开表达式中的每一项。\n你先尝试展开第一个部分：$(2x + 3y)^2$。\n请写出它的展开式。', 'role': 'user'},
#               {'content': '答：$(2x + 3y)^2 = 4x^2 + 12xy + 9y^2$', 'role': 'assistant'},
#               {'content': '很好！接下来，我们来看第二个部分：$(2x - 3y)(2x + 3y)$。\n这是一个**平方差公式**的形式，你可以尝试直接写出它的结果。\n平方差公式是：$(a - b)(a + b) = a^2 - b^2$。\n请写出它的结果。', 'role': 'user'}
#               ]
#     #stream=True  # 启用流式输出
# )
#
# print(response.choices[0].message.content)

'''识别文档内容-硅基流动不行，qwen-long效果一般'''

# with open("./test.pdf", "rb") as file:
#     file_content = file.read()
#
# client = OpenAI(
#     base_url='https://api.siliconflow.cn/v1',
#     api_key='sk-ytlhumlrgvmwkisiirugbjnsvwztwmdqbwqszghjfumyzrbw'
# )
#
# # 发送带有流式输出的请求
# response = client.chat.completions.create(
#     model="deepseek-ai/DeepSeek-V3",
#     messages=[{'content': '这是一份中考数学试卷，请提取其中的每道题目并按题号输出。要求同时给出题目类型：如选择题、解答题或几何题、计算题等。也要给出题目难度。', 'role': 'system'},
#               {'content': file_content, 'role': 'user'},
#               ],
#     stream=True  # 启用流式输出
# )
# for chunk in response:
#     chunk_message = chunk.choices[0].delta.content
#     print(chunk_message, end='', flush=True)

# client = OpenAI(
#     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
#     api_key="sk-30b325ff84cd484991a8e7a7777b63eb",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# file_object = client.files.create(file=Path("test2.pdf"), purpose="file-extract")
# completion = client.chat.completions.create(
#     model="qwen-long",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
#     messages=[
#         {'role': 'system', 'content': '这是一份中考数学试卷，请提取其中的每道题目并按题号输出。请尽量仔细地识别数学公式内容，保证与原文相同。'},
#         {'role': 'system', 'content': f'fileid://{file_object.id}'},
#         {'role': 'user', 'content': '这是一份中考数学试卷，请提取其中的每道题目并按题号输出。请尽量仔细地识别数学公式内容，保证与原文相同。'}
#     ]
# )
# print(completion.choices[0].message.content)
'''使用Adobe Acrobat合并pdf并导出到图片'''
'''解题并输出各种属性'''
# question = '计算并化简(2x + 3y)^2  - (2x - 3y)(2x + 3y) - (3y - 2x)^2'
# print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
# PS_shop: BotShop = BotShop(Problem_Solver)
# PS_bot: Problem_Solver = PS_shop.buy_bot(api_key=API_KEY_deepseek,base_url=BASE_URL_deepseek,max_history=8)
# messages=[
#     {'role': 'system', 'content': PROBLEM_SOLVER_PROMPT},
#     {'role': 'user', 'content': question},
# ]
# resp = PS_bot.generate_response('111', messages, 3000)
# line = question.replace(',', '，') + ',' + resp.replace('\n', ' ') + '\n'
# print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
# with open('calculate_problems/answers.txt', 'a+', encoding='utf-8') as f:
#     f.write(line)

from openai import OpenAI

client = OpenAI(
    base_url='https://api.siliconflow.cn/v1',
    api_key='sk-ytlhumlrgvmwkisiirugbjnsvwztwmdqbwqszghjfumyzrbw'
)

# 发送带有流式输出的请求
completion = client.chat.completions.create(
    model="Qwen/QwQ-32B",
    messages=[{'content': PROBLEM_SOLVER_PROMPT, 'role': 'system'},
              {'content': '计算并化简(2x + 3y)^2  - (2x - 3y)(2x + 3y) - (3y - 2x)^2', 'role': 'user'},
              ]
)
print(completion.choices[0].message.content)