import os
import uuid
import torch
import random
import argparse
import soundfile as sf
import numpy as np
import json
import yaml
import re
import sys
import subprocess
import asyncio
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import jwt

# 尝试导入推理库
try:
    import onnxruntime as ort
    import openwakeword
    from openwakeword.utils import AudioFeatures
    HAS_INFERENCE_LIBS = True
except ImportError:
    HAS_INFERENCE_LIBS = False

# --- 配置 ---
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
ALGORITHM = "HS256"
BASE_PREFIX = "/site"
DATABASE_URL = "sqlite:///./wakeword.db"
TRAIN_CONDA_ENV = "oww_train"

# --- 数据库模型 ---
Base = declarative_base()
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    github_id = Column(String, unique=True, index=True)
    username = Column(String); avatar_url = Column(String)

class Task(Base):
    __tablename__ = "tasks"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    wakeword = Column(String); status = Column(String); sub_status = Column(String)
    current_step = Column(Integer, default=1); progress = Column(Integer, default=0)
    params = Column(JSON); created_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

async def get_current_user(request: Request, db: Session = Depends(get_db)):
    auth_header = request.headers.get("Authorization")
    user = None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            if token == 'local-dev-mode': user = db.query(User).filter(User.username == "LocalDev").first()
            else:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                user = db.query(User).filter(User.id == payload.get("user_id")).first()
        except: pass
    if not user:
        user = db.query(User).filter(User.username == "LocalDev").first()
        if not user:
            user = User(github_id="local", username="LocalDev", avatar_url="https://github.com/identicons/local.png")
            db.add(user); db.commit(); db.refresh(user)
    return user

class TrainRequest(BaseModel):
    wakeword: str; similar_words: List[str]; num_samples: int; steps: int; layer_size: int; aug_rounds: int
class PreviewRequest(BaseModel): wakeword: str
class SimilarWordsRequest(BaseModel): wakeword: str

app = FastAPI(root_path=BASE_PREFIX)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

os.makedirs("static/previews", exist_ok=True)
os.makedirs("static/datasets", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 物理任务找回逻辑 ---
def recover_physical_tasks():
    db = SessionLocal()
    user = db.query(User).filter(User.username == "LocalDev").first()
    if not user: return
    
    datasets_dir = "static/datasets"
    if not os.path.exists(datasets_dir): return
    
    for task_id in os.listdir(datasets_dir):
        task_path = os.path.join(datasets_dir, task_id)
        if not os.path.isdir(task_path): continue
        
        # 检查是否已完成 (有配置文件和模型文件)
        config_file = os.path.join(task_path, "config.yaml")
        model_file = os.path.join(task_path, "beary_custom.onnx")
        
        if os.path.exists(config_file) and os.path.exists(model_file):
            # 检查数据库记录
            existing = db.query(Task).filter(Task.id == task_id).first()
            if not existing:
                print(f"Recovering completed task: {task_id}")
                try:
                    with open(config_file, 'r', encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)
                    
                    wakeword = cfg.get("target_phrase", task_id)
                    if isinstance(wakeword, list): wakeword = wakeword[0]
                    
                    new_task = Task(
                        id=task_id, user_id=user.id, wakeword=wakeword,
                        status="Completed", sub_status="物理找回已完成任务",
                        current_step=5, progress=100,
                        params={"num_samples": cfg.get("n_samples"), "steps": cfg.get("steps"), "layer_size": cfg.get("layer_size")},
                        created_at=datetime.fromtimestamp(os.path.getctime(model_file))
                    )
                    db.add(new_task)
                except Exception as e:
                    print(f"Failed to recover {task_id}: {e}")
    db.commit()
    db.close()

# --- 显存检测逻辑 ---
def get_free_vram_gb():
    try:
        if torch.cuda.is_available():
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0); a = torch.cuda.memory_allocated(0)
            return (t - (r + a)) / 1024**3
        if torch.backends.mps.is_available(): return 16.0 
    except: pass
    return 0.0

async def wait_for_vram(task_id, min_gb=6.0):
    while True:
        f = get_free_vram_gb()
        if f >= min_gb: return f
        db = SessionLocal(); t = db.query(Task).filter(Task.id == task_id).first()
        if t: t.sub_status = f"显存不足(剩余 {f:.1f}G), 正在等待..."; db.commit()
        db.close(); await asyncio.sleep(5)

# --- 业务逻辑 ---
def get_dynamic_config(n_samples, steps, layer_size, aug_rounds):
    val_samples = max(20, int(n_samples * 0.2)); pos_batch = 50 if n_samples > 100 else 16
    neg_weight = 1500 if layer_size >= 64 else 800; acc = 0.6 if steps >= 1000 else 0.5
    return {"n_samples": n_samples, "n_samples_val": val_samples, "steps": steps, "layer_size": layer_size, "augmentation_rounds": aug_rounds, "batch_n_per_class": {"positive": pos_batch, "adversarial_negative": 50, "ACAV100M_sample": 128}, "max_negative_weight": neg_weight, "target_accuracy": acc}

def generate_task_yaml(task_dir, wakeword, similar_words, n_samples, steps, layer_size, aug_rounds):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    template_path = os.path.join(root_dir, "my_model.yaml")
    config_data = {}
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding="utf-8") as f: config_data = yaml.safe_load(f)
    dynamic = get_dynamic_config(n_samples, steps, layer_size, aug_rounds)
    config_data.update({"target_phrase": wakeword, "similar_phrases": similar_words, "model_name": "beary_custom", "output_dir": task_dir})
    config_data.update(dynamic)
    with open(os.path.join(task_dir, "config.yaml"), "w", encoding="utf-8") as f: yaml.dump(config_data, f, allow_unicode=True)

async def run_cmd_v2(cmd, task_id, step_num, total_steps, sub_status_msg, cwd=None, env=None, track_dir=None, target_total=0, concurrent_num=1):
    db = SessionLocal(); t = db.query(Task).filter(Task.id == task_id).first()
    if t: t.sub_status, t.current_step, t.progress = sub_status_msg, step_num, 0; db.commit()
    db.close()
    
    if track_dir and target_total > 0 and os.path.exists(track_dir):
        count = len([f for f in os.listdir(track_dir) if f.endswith(".wav")])
        if count >= target_total:
            db_u = SessionLocal(); t_u = db_u.query(Task).filter(Task.id == task_id).first()
            if t_u: t_u.progress, t_u.sub_status = 100, f"{sub_status_msg} ({count}/{target_total})"; db_u.commit()
            db_u.close(); return
print(f"🚀 [TASK {task_id}] Step {step_num}: {sub_status_msg} (Parallel: {concurrent_num})")
processes = []
for i in range(concurrent_num):
    # 暂时放开第一个进程的 stdout 用于调试
    stdout_dest = subprocess.PIPE if i == 0 else subprocess.DEVNULL
    p = subprocess.Popen(cmd, stdout=stdout_dest, stderr=subprocess.PIPE, text=True, bufsize=1, cwd=cwd, env=env)
    processes.append(p)

# 异步读取调试日志
async def log_output(proc, idx, stream_name):
    stream = proc.stdout if stream_name == 'stdout' else proc.stderr
    while True:
        line = await asyncio.to_thread(stream.readline)
        if not line: break
        print(f"[{task_id}][P{idx} {stream_name}] {line.strip()}")

for i, p in enumerate(processes):
    if i == 0: asyncio.create_task(log_output(p, i, 'stdout'))
    asyncio.create_task(log_output(p, i, 'stderr'))
try:
    while any(p.poll() is None for p in processes):
        current_count = 0
        if track_dir and os.path.exists(track_dir): current_count = len([f for f in os.listdir(track_dir) if f.endswith(".wav")])

        if target_total > 0:
            step_percent = min(100, int((current_count / target_total) * 100))
            db_u = SessionLocal(); t_u = db_u.query(Task).filter(Task.id == task_id).first()
            if t_u:
                t_u.progress = step_percent
                # 如果数量为 0，给出一个人性化的提示
                display_count = f"{current_count}/{target_total}" if current_count > 0 else "正在初始化模型..."
                t_u.sub_status = f"{sub_status_msg} ({display_count})"
                db_u.commit()
            db_u.close()
                        try: p.terminate()
                        except: pass
                    break
            await asyncio.sleep(1)
    except Exception as e: print(f"Monitor error: {e}")
    for p in processes: p.wait()

async def run_v2_pipeline(task_id, resume_from_step=1):
    db = SessionLocal(); t = db.query(Task).filter(Task.id == task_id).first()
    if not t: return
    backend_dir = os.path.dirname(os.path.abspath(__file__)); scripts_dir = os.path.join(backend_dir, "scripts")
    task_dir = os.path.join(backend_dir, "static/datasets", task_id); config_path = os.path.join(task_dir, "config.yaml")
    root_dir = os.path.dirname(os.path.dirname(backend_dir))
    pos_train_dir = os.path.join(task_dir, "positive_train_tts"); neg_train_dir = os.path.join(task_dir, "negative_train_tts")
    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    n_pos = config.get("n_samples", 500); n_neg = config.get("n_samples", 500)
    env = os.environ.copy(); env["PYTHONPATH"] = root_dir + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    total_steps = 5; t.status = "Running"; db.commit(); db.close()
    try:
        for step_idx in [1, 2]:
            if resume_from_step <= step_idx:
                free_vram = await wait_for_vram(task_id, min_gb=6.0)
                concurrent = 3 if free_vram >= 18.0 else 1
                if step_idx == 1: await run_cmd_v2(["python", "v2_generate_positives.py", "--config", config_path], task_id, 1, total_steps, "生成正样本", scripts_dir, env, track_dir=pos_train_dir, target_total=n_pos, concurrent_num=concurrent)
                else: await run_cmd_v2(["python", "v2_generate_similars.py", "--config", config_path], task_id, 2, total_steps, "生成近似词样本", scripts_dir, env, track_dir=neg_train_dir, target_total=n_neg, concurrent_num=concurrent)
        if resume_from_step <= 3: await run_cmd_v2(["python", "v2_resample.py", "--config", config_path], task_id, 3, total_steps, "重采样音频", scripts_dir, env)
        if resume_from_step <= 4: await run_cmd_v2(["conda", "run", "-n", TRAIN_CONDA_ENV, "--no-capture-output", "python", "v2_augment.py", "--config", config_path], task_id, 4, total_steps, "样本增强与特征提取", scripts_dir, env)
        if resume_from_step <= 5: await run_cmd_v2(["conda", "run", "-n", TRAIN_CONDA_ENV, "--no-capture-output", "python", "v2_train.py", "--config", config_path], task_id, 5, total_steps, "训练模型", scripts_dir, env)
        db_f = SessionLocal(); t_f = db_f.query(Task).filter(Task.id == task_id).first()
        if t_f: t_f.status, t_f.sub_status, t_f.progress = "Completed", "训练完成", 100; db_f.commit(); db_f.close()
    except Exception as e:
        db_e = SessionLocal(); t_e = db_e.query(Task).filter(Task.id == task_id).first()
        if t_e: t_e.status, t_e.sub_status = "Failed", str(e); db_e.commit(); db_e.close()

@app.websocket("/api/ws/test-model/{task_id}")
async def websocket_test_model(websocket: WebSocket, task_id: str):
    await websocket.accept()
    backend_dir = os.path.dirname(os.path.abspath(__file__)); root_dir = os.path.dirname(os.path.dirname(backend_dir))
    model_path = os.path.join(root_dir, "beary.onnx") if task_id == "built-in-beary" else os.path.join(backend_dir, f"static/datasets/{task_id}/beary_custom.onnx")
    if not os.path.exists(model_path) or not HAS_INFERENCE_LIBS:
        await websocket.send_json({"error": "模型文件未找到或推理库未加载"}); await websocket.close(); return
    try:
        session = ort.InferenceSession(model_path); input_name = session.get_inputs()[0].name; model_window_size = session.get_inputs()[0].shape[1]
        F = AudioFeatures(); audio_buffer = np.array([], dtype=np.int16)
        while True:
            data = await websocket.receive_bytes(); audio_buffer = np.append(audio_buffer, np.frombuffer(data, dtype=np.int16))
            if len(audio_buffer) >= 20480:
                features = F.embed_clips(audio_buffer.reshape(1, -1))
                if features.shape[1] >= model_window_size:
                    score = float(session.run(None, {input_name: features[:, -model_window_size:, :]})[0][0][0])
                    await websocket.send_json({"score": score})
                    audio_buffer = np.array([], dtype=np.int16) if score > 0.5 else audio_buffer[-48000:]
    except WebSocketDisconnect: pass
    except Exception as e:
        try: await websocket.send_json({"error": str(e)})
        except: pass

@app.get("/")
async def read_root(): return RedirectResponse(url="/site/static/frontend/index.html")
@app.get("/api/me")
async def get_me(u=Depends(get_current_user)): return u
@app.get("/api/my-tasks")
async def get_my_tasks(u=Depends(get_current_user), db: Session = Depends(get_db)): return db.query(Task).filter(Task.user_id == u.id).order_by(Task.created_at.desc()).all()
@app.post("/api/preview")
async def generate_preview(req: PreviewRequest, u=Depends(get_current_user)):
    p_id = str(uuid.uuid4())[:8]; backend_dir = os.path.dirname(os.path.abspath(__file__)); out_dir = os.path.join(backend_dir, "static/previews", p_id); os.makedirs(out_dir, exist_ok=True)
    scripts_dir = os.path.join(backend_dir, "scripts"); cmd = [sys.executable, "generate_samples.py", "--wakeword", req.wakeword, "--output_dir", out_dir, "--num_samples", "3"]
    env = os.environ.copy(); env["PYTHONPATH"] = os.path.dirname(os.path.dirname(backend_dir)) + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=scripts_dir, env=env); return {"urls": [f"/site/static/previews/{p_id}/{f}" for f in os.listdir(out_dir) if f.endswith(".wav")]}
@app.post("/api/generate-similar-words")
def generate_similar_words(req: SimilarWordsRequest):
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"); cmd = [sys.executable, "v2_gen_word_list.py", "--wakeword", req.wakeword]
    try:
        res = subprocess.check_output(cmd, cwd=scripts_dir, text=True, timeout=120, stderr=subprocess.STDOUT, env=os.environ.copy())
        match = re.search(r"WORDS:(.*)", res)
        if match: return {"similar_words": [w.strip() for w in match.group(1).split(",") if w.strip()]}
        return {"similar_words": []}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/train")
async def start_training(req: TrainRequest, bt: BackgroundTasks, u=Depends(get_current_user), db: Session = Depends(get_db)):
    task_id = str(uuid.uuid4())[:8]; backend_dir = os.path.dirname(os.path.abspath(__file__)); task_dir = os.path.join(backend_dir, "static/datasets", task_id); os.makedirs(task_dir, exist_ok=True); generate_task_yaml(task_dir, req.wakeword, req.similar_words, req.num_samples, req.steps, req.layer_size, req.aug_rounds)
    new_task = Task(id=task_id, user_id=u.id, wakeword=req.wakeword, status="Pending", sub_status="等待中", params=req.dict(), current_step=1); db.add(new_task); db.commit(); bt.add_task(run_v2_pipeline, task_id, 1); return {"task_id": task_id}
@app.post("/api/retry/{task_id}")
async def retry_task(task_id: str, bt: BackgroundTasks, step: int = None, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task: raise HTTPException(status_code=404, detail="Task not found")
    start_from = step if step is not None else task.current_step
    task.status = "Pending"; task.sub_status = f"准备从第 {start_from} 步重试..."; db.commit(); bt.add_task(run_v2_pipeline, task_id, start_from); return {"message": "Retry started"}
@app.get("/api/models")
async def list_models(user=Depends(get_current_user), db: Session = Depends(get_db)):
    tasks = db.query(Task).filter(Task.user_id == user.id, Task.status == "Completed").all()
    user_models = [{"id":t.id, "wakeword":t.wakeword, "params":t.params, "download_url":f"/site/static/datasets/{t.id}/beary_custom.onnx", "is_built_in": False} for t in tasks]
    built_in = [{"id": "built-in-beary", "wakeword": "小熊 (内置)", "params": {"num_samples": "N/A", "steps": "N/A", "layer_size": "Standard"}, "download_url": "/site/static/beary.onnx", "is_built_in": True}]
    return built_in + user_models

if __name__ == "__main__":
    import uvicorn
    # 启动前清理僵尸任务
    db = SessionLocal(); stale_tasks = db.query(Task).filter(Task.status == "Running").all()
    for t in stale_tasks: t.status, t.sub_status = "Failed", "服务重启，任务已中断"
    db.commit()
    # 启动前找回物理任务
    recover_physical_tasks()
    db.close(); uvicorn.run(app, host="0.0.0.0", port=8000)
