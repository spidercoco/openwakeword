import os
import argparse
import json
import re
from openai import OpenAI

# 建议在运行前配置环境变量 DASHSCOPE_API_KEY
# export DASHSCOPE_API_KEY="your-api-key"

def get_similar_words(wakeword):
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    prompt = f"""你是一个语音算法辅助助手。请为唤醒词 "{wakeword}" 生成 10-15 个在中文发音上非常相近、容易引起误触发的词语或短语。
    要求：
    1. 包含声母相同、韵母相同或声调相近的词。
    2. 返回格式必须是纯 JSON 字符串数组，例如：["词1", "词2", "词3"]。
    3. 不要包含任何解释性文字。
    """

    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that only outputs JSON arrays."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"} if False else None # 某些模型支持，这里手动解析更稳
        )
        
        content = completion.choices[0].message.content
        # 尝试从返回内容中提取 JSON 数组
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            words = json.loads(match.group(0))
            return words
        else:
            return [wakeword + "类似"] # 保底
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wakeword", type=str, required=True)
    args = parser.parse_args()
    
    similar_words = get_similar_words(args.wakeword)
    if similar_words:
        print(f"WORDS:{','.join(similar_words)}")
    else:
        print("WORDS:")
