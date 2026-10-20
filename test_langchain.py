# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# import os
# from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

api_key = "sk-0c7707325f5941e296f105479592a743"
api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model_emb = "text-embedding-v4"
model_chat = "qwen-plus"

# print(f"Testing LangChain OpenAIEmbeddings with base={api_base}...")

# try:
#     # 模拟 utils.py 中的初始化方式
#     llm = ChatOpenAI(
#         api_key=api_key,
#         openai_api_base=api_base,
#         model=model_chat,
#         temperature=0.0,
#         extra_body={"enable_thinking": False},
#         #callbacks=[StreamingStdOutCallbackHandler()]
#         n=3,
#             # 使流式返回的最后一个数据包包含Token消耗信息
#     )
#     res=llm.invoke("Hello!你是什么模型，你是qwen哪个版本，控制每个回答在200token以内")
#     print("Chat Success! ",res)
#     #print("Chat Success!", res.content)

#     # embed_model=OpenAIEmbeddings( openai_api_key=api_key,
#     #         openai_api_base=api_base,
#     #         model=model_emb,
#     #         check_embedding_ctx_length=False,
#     #         chunk_size=1000,
#     #         timeout=300.0)
#     # out_embedding=embed_model.embed_query("test")
#     # print(f"query的嵌入结果：{out_embedding}")

# except Exception as e:
#     print("Chat Failed!")
#     print(e)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json

api_key = "sk-0c7707325f5941e296f105479592a743"
api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model_name = "qwen3.5-plus"

# 1. 初始化模型
# 注意：extra_body 直接作为参数传递给 ChatOpenAI
llm = ChatOpenAI(
    api_key=api_key,
    base_url=api_base,  # LangChain 较新版本使用 base_url，旧版本用 openai_api_base
    model=model_name,
    temperature=0.7,    # 建议设置温度，否则 n>1 时结果可能完全一样
    extra_body={
        "enable_thinking": False
    },
    # 重要：这里不直接设 n=3 在构造函数里，因为 invoke 会忽略它。
    # 我们将在调用时通过 config 传递 n，或者使用 generate 方法。
    n=3
)

# 2. 构建消息
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="你是谁？")
]

print(f"正在请求 {model_name} (n=3)...")

try:
    # 【关键步骤】使用 .generate() 方法而不是 .invoke()
    # .invoke() 只会返回第一个结果 (choices[0])
    # .generate() 会返回完整的 LLMResult，包含所有 n 个结果
    response = llm.generate([messages])
    
    # 3. 解析结果
    # response.generations 是一个列表，外层对应输入列表（这里只有1组消息），内层对应 n 个结果
    if response.generations and len(response.generations) > 0:
        choices = response.generations[0] # 获取第一组消息对应的所有生成结果
        
        print(f"成功获取 {len(choices)} 个回复:\n")
        
        results_list = []
        for i, generation in enumerate(choices):
            content = generation.text
            results_list.append({
                "index": i,
                "content": content,
                # 可以获取更多元数据，如 finish_reason, usage 等
                "metadata": generation.generation_info
            })
            print(f"--- 回复 {i+1} ---")
            print(content)
            print("-" * 30)
            
        # 如果需要模拟 model_dump_json()
        print("\n=== JSON 格式输出 ===")
        print(json.dumps(results_list, ensure_ascii=False, indent=2))
    else:
        print("未生成任何结果。")

except Exception as e:
    print("请求失败:")
    print(e)
    # 打印详细错误以便调试
    import traceback
    traceback.print_exc()