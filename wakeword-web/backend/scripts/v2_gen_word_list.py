import os
import argparse
import json
import re
import dashscope
from dashscope import Generation

# 建议在运行前配置环境变量 DASHSCOPE_API_KEY
# export DASHSCOPE_API_KEY="your-api-key"

def get_similar_words(wakeword):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    model_name = "qwen-max"

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

    messages = [
        {"role": "system", "content": "You are a helpful assistant that only outputs JSON arrays."},
        {"role": "user", "content": prompt}
    ]

    print(f"--- LLM Call (DashScope) ---")
    print(f"Model: {model_name}")
    print(f"Prompt: {wakeword}")
    print(f"----------------------------")

    try:
        response = Generation.call(
            api_key=api_key,
            model=model_name,
            messages=messages,
            result_format="message",
            #enable_thinking=True
        )

        if response.status_code == 200:
            msg = response.output.choices[0].message
            
            # 打印思考过程（如果存在）

            answer_content = msg.content
            print("--- 完整回复 ---")
            print(answer_content)
            print("----------------")

            # 尝试从返回内容中提取 JSON 数组
            match = re.search(r"\[.*\]", answer_content, re.DOTALL)
            if match:
                words = json.loads(match.group(0))
                return words
            else:
                return [wakeword + "类似"]
        else:
            print(f"API Error: {response.code} - {response.message}")
            return []
            
    except Exception as e:
        print(f"Error calling DashScope: {e}")
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
