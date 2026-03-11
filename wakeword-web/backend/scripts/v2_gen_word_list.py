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

    prompt = f"""你是一个中文语音对抗样本生成器。

输入：一个中文词语

目标：生成大量读音非常接近但绝对不能发音完全一样的词语。

要求：
1. 拼音相似度 >= 80%
2. 可以改变：
   - 声母
   - 韵母
   - 声调
3. 可以生成无意义词语
4. 保持相同字数
5. 不要解释
用json的列表返回。

**请在返回前再次根据声母，韵母，声调分别确认，绝对不能和原词语完全一样。**

输入词语："{wakeword}"
"""

    try:
        completion = client.chat.completions.create(
            model="qwen3.5-plus",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that only outputs JSON arrays."},
                {"role": "user", "content": prompt},
            ],
            extra_body={"enable_thinking": True}
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
