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
    model_name = "qwen3-max"

    prompt = f"""你是一个中文语音对抗样本生成器。
    输入：一个中文词语
    任务：生成与输入词语读音高度相似但绝不完全相同的词语，并返回结构化结果。
    要求：
    - 输出词语必须与输入字数一致；
    - 必须有音节的声母、韵母或声调变化；
    - 优先选择真实存在或语义合理的词语；

    输出格式：
    返回一个 JSON 对象，包含以下两个字段：
    "original_pinyin": 原输入词的标准带调拼音（字符串）
    "samples": 一个列表，每个元素为对象，包含：
    - "text": 对抗样本词语（字符串）
    - "pinyin": 该词语的标准带调拼音（字符串）
    不要包含任何额外说明、注释或 Markdown。 输入是：{wakeword}
    """

    messages = [{"role": "user", "content": prompt}]

    try:
        response = Generation.call(
            api_key=api_key,
            model=model_name,
            messages=messages,
            result_format="message",
            #enable_thinking=True
        )

        if response.status_code == 200:
            content = response.output.choices[0].get('message', {}).get('content', "")
            # 提取并解析 JSON
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                original_pinyin = data.get("original_pinyin")
                if "samples" in data:
                    seen_pinyin = {original_pinyin} if original_pinyin else set()
                    result = []
                    for s in data.get("samples", []):
                        text = s.get("text")
                        pinyin = s.get("pinyin")
                        if text and pinyin and pinyin not in seen_pinyin:
                            result.append(text)
                            seen_pinyin.add(pinyin)
                    return result
            return []
        else:
            return []
    except:
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
