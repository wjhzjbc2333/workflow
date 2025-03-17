#qwen-math最大输入输出都是3072，上下文长度4096
#qwen-vl-ocr最大输入30000，最大输出4096，上下文长度34096
#qwen-vl-max最大输入30720，最大输出2048，上下文长度32768
QWEN_API = {
    "API_KEY" : "sk-30b325ff84cd484991a8e7a7777b63eb",
    "BASE_URL" : "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "MODEL": "qwen-math-plus-latest",
    "MODEL-72B" : "qwen2.5-math-72b-instruct",
    "MODEL-7B" : "qwen2.5-math-7b-instruct",
    "CRITIC_MODEL": "qwen-max-latest",
    "STUPID_MODEL-1" : "qwen-1.8b-chat",
    "STUPID_MODEL-2" : "qwen-math-plus",
}
#QWEN版deepseek
# DeepSeek_API = {
#     "API_KEY" : "sk-30b325ff84cd484991a8e7a7777b63eb",
#     "BASE_URL" : "https://dashscope.aliyuncs.com/compatible-mode/v1",
#     "MODEL-V3" : "deepseek-v3",
#     "MODEL-R1" : "deepseek-r1",
# }

#硅基流动版deepseek
DeepSeek_API = {
    "API_KEY" : "sk-ytlhumlrgvmwkisiirugbjnsvwztwmdqbwqszghjfumyzrbw",
    "BASE_URL" : "https://api.siliconflow.cn/v1",
    "MODEL-V3" : "deepseek-ai/DeepSeek-V3",
    "MODEL-R1" : "deepseek-ai/DeepSeek-R1",
    "MODEL-QwQ" : "Qwen/QwQ-32B"
}

