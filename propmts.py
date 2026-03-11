import itertools
import json
import csv
import random

# -------------------------
# 声音属性
# -------------------------

gender = ["男性","女性"]

age = [
"儿童",
"青年",
"中年",
"中老年",
"老年"
]

pitch = [
"低沉",
"中等",
"高亢"
]

speed = [
"偏慢",
"正常",
"稍快"
]

volume = [
"轻声",
"中等",
"洪亮"
]

timbre = [
"浑厚",
"磁性",
"明亮",
"沙哑",
"温润"
]

emotion = [
"平静",
"温和",
"严肃",
"威严",
"热情"
]

tone = [
"陈述式",
"命令式",
"讲述式"
]

style = [
"播报式",
"演讲式",
"聊天式",
"讲故事式"
]

clarity = [
"清晰",
"非常清晰"
]


# -------------------------
# 过滤规则
# -------------------------

def is_valid(c):

    g,a,p,s,v,t,e,to,st,cl = c

    # 年龄 vs 音高
    if a=="儿童" and p=="低沉":
        return False

    if a=="老年" and p=="高亢":
        return False

    # 儿童声音限制
    if a=="儿童" and t in ["浑厚","沙哑"]:
        return False

    # 轻声不能威严
    if v=="轻声" and e in ["威严"]:
        return False

    # 平静不能命令
    if e=="平静" and to=="命令式":
        return False

    # 女性极低沉概率低
    if g=="女性" and p=="低沉" and t=="浑厚":
        return False

    return True


# -------------------------
# 生成
# -------------------------

results=[]

for combo in itertools.product(
        gender,
        age,
        pitch,
        speed,
        volume,
        timbre,
        emotion,
        tone,
        style,
        clarity
):

    if is_valid(combo):
        results.append(combo)


print("合理组合数:",len(results))


# -------------------------
# 随机抽取
# -------------------------

MAX=10000

if len(results) > MAX:
    results=random.sample(results,MAX)


# -------------------------
# 转换为结构
# -------------------------

voices=[]

for r in results:

    d={
        "性别":r[0],
        "年龄":r[1],
        "音高":r[2],
        "语速":r[3],
        "音量":r[4],
        "音色":r[5],
        "情绪":r[6],
        "语调":r[7],
        "风格":r[8],
        "清晰度":r[9],
        "prompt":
        f"{r[1]}{r[0]}，{r[5]}嗓音，音高{r[2]}，语速{r[3]}，音量{r[4]}，语气{r[6]}，{r[7]}语调，{r[8]}风格，发音{r[9]}"
    }

    voices.append(d)


# -------------------------
# JSON
# -------------------------

with open("voices.json","w",encoding="utf8") as f:
    json.dump(voices,f,ensure_ascii=False,indent=2)


# -------------------------
# CSV
# -------------------------

with open("voices.csv","w",encoding="utf8",newline="") as f:

    writer=csv.writer(f)

    writer.writerow([
        "性别","年龄","音高","语速","音量",
        "音色","情绪","语调","风格","清晰度"
    ])

    for v in voices:

        writer.writerow([
            v["性别"],
            v["年龄"],
            v["音高"],
            v["语速"],
            v["音量"],
            v["音色"],
            v["情绪"],
            v["语调"],
            v["风格"],
            v["清晰度"]
        ])

print("生成voice数量:",len(voices))
