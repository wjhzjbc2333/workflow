import time
import traceback

from chatbot_api import *
from const import *

'''
工作流程：
1.调用OCR模型切题/直接读取文档获得题目，对题目进行多标签分类
2.使用笨模型（学生）尝试解题，强模型（老师）判断是否出错并指点
3.使用评判模型评价交互过程
4.保存交互和评判记录，做下一道题。
'''

'''
问题：
1.deepseek思考太慢，经常超时
2.笨模型全都不听话，一股脑全输出了
3.交互过程长度很容易超过其中某个模型的上限要求
4.几何类题目会做的模型很少 AlphaGeometry？
'''
API_KEY = QWEN_API["API_KEY"]
BASE_URL = QWEN_API["BASE_URL"]
API_KEY_deepseek = DeepSeek_API["API_KEY"]
BASE_URL_deepseek = DeepSeek_API["BASE_URL"]


PS_shop: BotShop = BotShop(Problem_Solver)
PS_bot: Problem_Solver = PS_shop.buy_bot(api_key=API_KEY_deepseek,base_url=BASE_URL_deepseek,max_history=8)

SS_shop: BotShop = BotShop(Stupid_Student)
SS_bot: Stupid_Student = SS_shop.buy_bot(api_key=API_KEY_deepseek, base_url=BASE_URL_deepseek, max_history=8)

TA_shop: BotShop = BotShop(Teaching_Assistant)
TA_bot: Teaching_Assistant = TA_shop.buy_bot(api_key=API_KEY, base_url=BASE_URL, max_history=8)

EE_shop: BotShop = BotShop(Education_Expert)
EE_bot: Education_Expert = EE_shop.buy_bot(api_key=API_KEY, base_url=BASE_URL, max_history=8)

VL_bot = VL_OCR_Bot()


USER_ID = "111"

def reset_history():
    PS_bot.reset_history(USER_ID)
    SS_bot.reset_history(USER_ID)
    TA_bot.reset_history(USER_ID)
    EE_bot.reset_history(USER_ID)

def get_questions_by_qwenVL():
    #Folder = 'calculate_problems/pics/'
    Folder = '中考题目/2019-2023计算+概率/pics/'
    file_names = os.listdir(Folder)
    file = open('中考题目/2019-2023计算+概率/problems.txt', 'a+', encoding='utf-8')
    for file_name in file_names:
        image_path = Folder + file_name
        try:
            questions = VL_bot.generate_response_with_VL(image_path)
        except:
            questions = []
        for question in questions:
            file.write(question.strip().replace('\n', '').replace('\r', '') + '\n')
    file.close()

def get_questions_by_qwenLong(path):
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-30b325ff84cd484991a8e7a7777b63eb",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    file_object = client.files.create(file=Path(path), purpose="file-extract")
    completion = client.chat.completions.create(
        model="qwen-long",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {'role': 'system',
             'content': '这是一份中考数学试卷，请提取其中的每道题目并按题号输出。遇到数学公式时请尽量以latex格式给出。'},
            {'role': 'system', 'content': f'fileid://{file_object.id}'},
            {'role': 'user',
             'content': '这是一份中考数学试卷，请提取其中的每道题目并按题号输出。遇到数学公式时请尽量以latex格式给出。'}
        ]
    )
    completion.choices[0].message.content

def get_and_store_answers_and_attrs():
    number = 0
    with open('calculate_problems/problems.txt', 'r', encoding='utf-8') as f:
        questions = f.readlines()
    for question in questions:
        number += 1
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print(f"问题{number}")

        messages = [
            {'role': 'system', 'content': PROBLEM_SOLVER_PROMPT},
            {'role': 'user', 'content': question},
        ]
        try:
            resp = PS_bot.generate_response('111', messages, 3000)
            line = question.replace(',', '，').replace('\n', '').replace('\r', '') + ',' + resp.replace('\n', ' ').replace('\r', ' ') + '\n'
        except:
            line = question.replace(',', '，').replace('\n', '').replace('\r', '') + ',\n'
        with open('calculate_problems/answers_r1.txt', 'a+', encoding='utf-8') as f:
            f.write(line)

def read_attrs_and_interact():
    with open('calculate_problems/answers.txt', 'r', encoding='utf-8') as f:
        attrs = f.readlines()
    number = 894
    for attr in attrs:
        number += 1
        question = attr.split(',')[0]
        #是否需要answer？
        answer = attr.split(',')[-1]
        student_history = [
            {'role': 'system', 'content': STUPID_MODEL_PROMPT},
            {'role': 'system', 'content': question}
        ]
        teacher_history = [
            {'role': 'system', 'content': TEACHING_MODEL_PROMPT},
            {'role': 'user', 'content': question}
        ]

        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print(f"问题{number}")

        file = open(f'calculate_problems/interactions/问题{number}.txt', 'a+', encoding='utf-8')
        file.write(f'学生：{question}\n')
        teacher_response = ''
        while teacher_response.find("问答结束") == -1:
            #学生与老师交互
            try:
                teacher_response = TA_bot.generate_response(USER_ID, teacher_history, 500)
            except:
                teacher_response = '问答结束'
            teacher_history.append({'role': 'assistant', 'content': teacher_response})
            student_history.append({'role': 'user', 'content': teacher_response})
            teacher_response = teacher_response.replace('\n', '').replace('\r', '') + '\n'
            file.write(f'老师：{teacher_response}')

            try:
                student_response = SS_bot.generate_response(USER_ID, student_history, 500)
            except:
                student_response = ''
            student_history.append({'role': 'assistant', 'content': student_response})
            teacher_history.append({'role': 'user', 'content': student_response})
            student_response = student_response.replace('\n', '').replace('\r', '') + '\n'
            file.write(f'学生：{student_response}')

        '''评判模型-错误定位'''
        try:
            response = EE_bot.get_critic_error_locating(USER_ID, teacher_history[1:], 3000, TA_bot).replace('\n', '').replace('\r', '') + '\n'
        except:
            response = '\n'
        file.write(f'错误定位：{response}')
        '''评判模型-给出建议'''
        try:
            response = EE_bot.get_critic_suggestions(USER_ID, teacher_history[1:], 3000, TA_bot).replace('\n', '').replace('\r', '') + '\n'
        except:
            response = '\n'
        file.write(f'给出建议：{response}')

        reset_history()
        file.close()

if __name__ == '__main__':
    number = 0
    '''视觉模型-识别题目'''
    '''题目已提取到problems.txt'''
    get_questions_by_qwenVL()
    '''解题模型，给出题目属性和答案'''
    '''属性已提取到answers_r1/v3.txt'''
    #get_and_store_answers_and_attrs()
    '''老师学生模型交互'''
    '''结果储存至interactions下'''
    #read_attrs_and_interact()
