from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from authlib.integrations.starlette_client import OAuth
import jwt
import uuid
import os
import time
import subprocess
import re
from datetime import datetime, timedelta
from typing import List, Optional

# --- 配置 ---
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
ALGORITHM = "HS256"
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "mock-id")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "mock-secret")
CONDA_ENV_TTS = os.getenv("CONDA_ENV_NAME", "qwen3-tts")
CONDA_ENV_TRAIN = "python310"

DATABASE_URL = "sqlite:///./wakeword.db"

# --- 数据库模型 ---
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    github_id = Column(String, unique=True, index=True)
    username = Column(String)
    avatar_url = Column(String)

class Task(Base):
    __tablename__ = "tasks"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    wakeword = Column(String)
    status = Column(String) # Pending, Running, Completed, Failed
    sub_status = Column(String) 
    current_step = Column(Integer, default=1)
    progress = Column(Integer, default=0)
    params = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# --- Pydantic 模型 ---
class PreviewRequest(BaseModel):
    wakeword: str
    speaker: str

class TrainRequest(BaseModel):
    wakeword: str
    num_samples: int
    epochs: int

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

os.makedirs("static/previews", exist_ok=True)
os.makedirs("static/datasets", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 依赖与鉴权 ---
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
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user = db.query(User).filter(User.id == payload.get("user_id")).first()
        except: pass
    if not user:
        user = db.query(User).filter(User.username == "LocalDev").first()
        if not user:
            user = User(github_id="local", username="LocalDev", avatar_url="https://github.com/identicons/local.png")
            db.add(user)
            db.commit()
            db.refresh(user)
    return user

# --- 执行工具 ---
def run_command_with_progress(cmd: List[str], task_id: str, step_num: int, start_progress: int, end_progress: int, sub_status_msg: str, cwd: str = None):
    db = SessionLocal()
    t = db.query(Task).filter(Task.id == task_id).first()
    if t:
        t.sub_status = f"{step_num}/5: {sub_status_msg}"
        t.current_step = step_num
        t.progress = start_progress
        db.commit()
    db.close()

    env = os.environ.copy()
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = root_dir + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

    print(f"Executing Step {step_num}: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=cwd, env=env)
    
    progress_re = re.compile(r"PROGRESS:(\d+)/(\d+)")
    for line in process.stdout:
        match = progress_re.search(line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            percent = int(start_progress + (end_progress - start_progress) * (current / total))
            db_u = SessionLocal()
            t_u = db_u.query(Task).filter(Task.id == task_id).first()
            if t_u:
                t_u.progress = percent
                t_u.sub_status = f"{step_num}/5: {sub_status_msg} ({current}/{total})"
                db_u.commit()
            db_u.close()
    
    process.wait()
    if process.returncode != 0:
        raise Exception(f"Failed at stage {step_num}/5: {sub_status_msg}")

# --- 训练全流程流水线 ---
def training_pipeline(task_id: str, resume_from_step: int = 1):
    db = SessionLocal()
    t = db.query(Task).filter(Task.id == task_id).first()
    if not t:
        db.close()
        return
    wakeword = t.wakeword
    num_samples = t.params.get('num_samples', 100)
    epochs = t.params.get('epochs', 50)
    t.status = "Running"
    db.commit()
    db.close()

    try:
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        scripts_dir = os.path.join(backend_dir, "scripts")
        dataset_dir = os.path.join(backend_dir, "static/datasets", task_id)
        wavs_raw_dir = os.path.join(dataset_dir, "wavs_raw")
        wavs_resampled_dir = os.path.join(dataset_dir, "wavs_resampled")
        
        os.makedirs(wavs_raw_dir, exist_ok=True)
        os.makedirs(wavs_resampled_dir, exist_ok=True)
        
        neg_feat = os.path.join(dataset_dir, "negative_features.npy")
        pos_feat = os.path.join(dataset_dir, "positive_features.npy")
        model_out = os.path.join(dataset_dir, "beary_custom.onnx")

        if resume_from_step <= 1:
            run_command_with_progress(
                ["conda", "run", "-n", CONDA_ENV_TTS, "--no-capture-output", "python", "generate_samples.py", 
                 "--wakeword", wakeword, "--output_dir", wavs_raw_dir, "--num_samples", str(num_samples)],
                task_id, 1, 0, 40, "生成声音样本", cwd=scripts_dir
            )

        if resume_from_step <= 2:
            run_command_with_progress(
                ["conda", "run", "-n", CONDA_ENV_TRAIN, "--no-capture-output", "python", "resample_web.py",
                 "--input_dir", wavs_raw_dir, "--output_dir", wavs_resampled_dir],
                task_id, 2, 40, 50, "重采样正样本", cwd=scripts_dir
            )

        if resume_from_step <= 3:
            run_command_with_progress(
                ["conda", "run", "-n", CONDA_ENV_TRAIN, "--no-capture-output", "python", "negative_web.py",
                 "--output_file", neg_feat],
                task_id, 3, 50, 70, "提取负样本特征", cwd=scripts_dir
            )

        if resume_from_step <= 4:
            run_command_with_progress(
                ["conda", "run", "-n", CONDA_ENV_TRAIN, "--no-capture-output", "python", "positive_web.py",
                 "--positive_input_dir", wavs_resampled_dir, "--output_file", pos_feat],
                task_id, 4, 70, 85, "提取正样本特征", cwd=scripts_dir
            )

        if resume_from_step <= 5:
            run_command_with_progress(
                ["conda", "run", "-n", CONDA_ENV_TRAIN, "--no-capture-output", "python", "train_web.py",
                 "--positive_features", pos_feat, "--negative_features", neg_feat, "--output_model", model_out, "--epochs", str(epochs)],
                task_id, 5, 85, 100, "模型训练中", cwd=scripts_dir
            )

        db_f = SessionLocal()
        t_f = db_f.query(Task).filter(Task.id == task_id).first()
        if t_f:
            t_f.status = "Completed"
            t_f.sub_status = "训练完成"
            t_f.progress = 100
            db_f.commit()
        db_f.close()

    except Exception as e:
        db_e = SessionLocal()
        t_e = db_e.query(Task).filter(Task.id == task_id).first()
        if t_e:
            t_e.status = "Failed"
            t_e.sub_status = str(e)
            db_e.commit()
        db_e.close()

# --- 路由 ---
@app.get("/")
async def read_root():
    # 改为相对路径
    return RedirectResponse(url="static/frontend/index.html")

@app.get("/api/me")
async def get_me(user: User = Depends(get_current_user)):
    return user

@app.get("/api/my-tasks")
async def get_my_tasks(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(Task).filter(Task.user_id == user.id).order_by(Task.created_at.desc()).all()

@app.get("/api/models")
async def list_models(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    completed_tasks = db.query(Task).filter(Task.user_id == user.id, Task.status == "Completed").order_by(Task.created_at.desc()).all()
    models = []
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    import json
    for t in completed_tasks:
        task_dir = os.path.join(backend_dir, "static/datasets", t.id)
        model_path = os.path.join(task_dir, "beary_custom.onnx")
        metrics_path = os.path.join(task_dir, "metrics.json")
        if os.path.exists(model_path):
            # 移除开头的 / 改为相对路径
            m_data = {
                "id": t.id, 
                "wakeword": t.wakeword, 
                "created_at": t.created_at, 
                "params": t.params,
                "download_url": f"static/datasets/{t.id}/beary_custom.onnx"
            }
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                        m_data["recall"] = metrics.get("final_recall", 0)
                except: pass
            plot_path = os.path.join(task_dir, "training_plot.png")
            if os.path.exists(plot_path):
                m_data["plot_url"] = f"static/datasets/{t.id}/training_plot.png"
            models.append(m_data)
    return models

@app.post("/api/preview")
async def generate_preview(req: PreviewRequest, user: User = Depends(get_current_user)):
    preview_id = str(uuid.uuid4())[:8]
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(backend_dir, "static/previews", preview_id)
    os.makedirs(output_dir, exist_ok=True)
    scripts_dir = os.path.join(backend_dir, "scripts")
    cmd = ["conda", "run", "-n", CONDA_ENV_TTS, "--no-capture-output", "python", "generate_samples.py", 
           "--wakeword", req.wakeword, "--output_dir", output_dir, "--num_samples", "3"]
    env = os.environ.copy()
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = root_dir + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=scripts_dir, env=env)
    files = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
    # 移除开头的 / 改为相对路径
    urls = [f"static/previews/{preview_id}/{f}" for f in files]
    return {"urls": urls}

@app.post("/api/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks, 
                         user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    task_id = str(uuid.uuid4())[:8]
    new_task = Task(id=task_id, user_id=user.id, wakeword=req.wakeword, status="Pending", sub_status="等待中", params=req.dict(), current_step=1)
    db.add(new_task)
    db.commit()
    background_tasks.add_task(training_pipeline, task_id, 1)
    return {"task_id": task_id}

@app.post("/api/retry/{task_id}")
async def retry_task(task_id: str, background_tasks: BackgroundTasks, step: int = None, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    start_from = step if step is not None else task.current_step
    task.status = "Pending"
    task.sub_status = f"准备从第 {start_from} 步重试..."
    db.commit()
    background_tasks.add_task(training_pipeline, task_id, start_from)
    return {"message": f"Retry started from step {start_from}"}

@app.get("/api/status/{task_id}")
async def get_status(task_id: str, db: Session = Depends(get_db)):
    return db.query(Task).filter(Task.id == task_id).first()

@app.post("/api/test/{task_id}")
async def test_model(task_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    import librosa
    import onnxruntime as ort
    import openwakeword.utils
    import numpy as np
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(backend_dir, "static/datasets", task_id, "beary_custom.onnx")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    audio, _ = librosa.load(file.file, sr=16000)
    min_length = 3 * 16000
    if len(audio) < min_length:
        audio = np.pad(audio, (min_length - len(audio), 0))
    else:
        audio = np.pad(audio, (min_length, 0))
    audio_int16 = (audio * 32767).astype(np.int16).reshape(1, -1)
    F = openwakeword.utils.AudioFeatures()
    features = F.embed_clips(audio_int16)
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    model_window_size = session.get_inputs()[0].shape[1]
    n_embeddings = features.shape[1]
    scores = []
    for i in range(0, n_embeddings - model_window_size + 1):
        window = features[:, i : i + model_window_size, :]
        outputs = session.run(None, {input_name: window})
        scores.append(float(outputs[0][0][0]))
    max_score = max(scores) if scores else 0
    return {"score": max_score, "detected": max_score > 0.5}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
