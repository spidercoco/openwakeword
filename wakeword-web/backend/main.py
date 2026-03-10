from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import jwt
import uuid
import os
import yaml
import time
import subprocess
import re
from datetime import datetime, timedelta
from typing import List, Optional

# --- 配置 ---
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
ALGORITHM = "HS256"
BASE_PREFIX = "/site"
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
    similar_words = Column(JSON) 
    status = Column(String) 
    sub_status = Column(String) 
    current_step = Column(Integer, default=1)
    progress = Column(Integer, default=0)
    params = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

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

# --- Pydantic 模型 ---
class PreviewRequest(BaseModel):
    wakeword: str
    speaker: Optional[str] = "vivian"

class SimilarWordsRequest(BaseModel):
    wakeword: str

class TrainRequest(BaseModel):
    wakeword: str
    similar_words: List[str]
    num_samples: int
    epochs: int

app = FastAPI(root_path=BASE_PREFIX)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

os.makedirs("static/previews", exist_ok=True)
os.makedirs("static/datasets", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 业务逻辑：生成任务 YAML ---
def generate_task_yaml(task_dir: str, wakeword: str, similar_words: List[str], num_samples: int, epochs: int):
    config_data = {
        "target_phrase": wakeword,
        "similar_phrases": similar_words,
        "n_samples": num_samples,
        "n_samples_val": int(num_samples * 0.2),
        "steps": epochs * 20, 
        "epochs": epochs,
        "model_name": "custom_model",
        "output_dir": "./" 
    }
    with open(os.path.join(task_dir, "config.yaml"), "w") as f:
        yaml.dump(config_data, f)

# --- 核心流水线重构 ---
def run_v2_pipeline(task_id: str, resume_from_step: int = 1):
    db = SessionLocal()
    t = db.query(Task).filter(Task.id == task_id).first()
    if not t: return
    
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(backend_dir, "scripts")
    task_dir = os.path.join(backend_dir, "static/datasets", task_id)
    config_path = os.path.join(task_dir, "config.yaml")
    
    env = os.environ.copy()
    root_dir = os.path.dirname(os.path.dirname(backend_dir))
    env["PYTHONPATH"] = root_dir + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

    total_steps = 5
    t.status = "Running"
    db.commit()
    db.close()

    try:
        if resume_from_step <= 1:
            run_cmd_v2(["python", "v2_generate_positives.py", "--config", config_path], task_id, 1, total_steps, 0, 30, "生成正样本", scripts_dir, env)
        if resume_from_step <= 2:
            run_cmd_v2(["python", "v2_generate_similars.py", "--config", config_path], task_id, 2, total_steps, 30, 60, "生成近似词样本", scripts_dir, env)
        if resume_from_step <= 3:
            run_cmd_v2(["python", "v2_resample.py", "--config", config_path], task_id, 3, total_steps, 60, 70, "重采样音频", scripts_dir, env)
        if resume_from_step <= 4:
            run_cmd_v2(["python", "v2_augment.py", "--config", config_path], task_id, 4, total_steps, 70, 90, "样本增强与特征提取", scripts_dir, env)
        if resume_from_step <= 5:
            run_cmd_v2(["python", "v2_train.py", "--config", config_path], task_id, 5, total_steps, 90, 100, "训练模型", scripts_dir, env)

        db_f = SessionLocal()
        t_f = db_f.query(Task).filter(Task.id == task_id).first()
        t_f.status, t_f.sub_status, t_f.progress = "Completed", "训练完成", 100
        db_f.commit()
        db_f.close()
    except Exception as e:
        db_e = SessionLocal()
        t_e = db_e.query(Task).filter(Task.id == task_id).first()
        t_e.status, t_e.sub_status = "Failed", str(e)
        db_e.commit()
        db_e.close()

def run_cmd_v2(cmd, task_id, step, total, start_p, end_p, msg, cwd, env):
    db = SessionLocal()
    t = db.query(Task).filter(Task.id == task_id).first()
    t.sub_status, t.current_step, t.progress = f"{step}/{total}: {msg}", step, start_p
    db.commit()
    db.close()

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=cwd, env=env)
    progress_re = re.compile(r"PROGRESS:(\d+)/(\d+)")
    for line in process.stdout:
        match = progress_re.search(line)
        if match:
            curr, tot = int(match.group(1)), int(match.group(2))
            percent = int(start_p + (end_p - start_p) * (curr / tot))
            db_u = SessionLocal()
            t_u = db_u.query(Task).filter(Task.id == task_id).first()
            if t_u:
                t_u.progress = percent
                t_u.sub_status = f"{step}/{total}: {msg} ({curr}/{tot})"
                db_u.commit()
            db_u.close()
    process.wait()
    if process.returncode != 0: raise Exception(f"Step {step} failed: {msg}")

# --- 路由 ---
@app.get("/")
async def read_root(): return RedirectResponse(url="/site/static/frontend/index.html")

@app.get("/api/me")
async def get_me(u=Depends(get_current_user)): return u

@app.get("/api/my-tasks")
async def get_my_tasks(u=Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(Task).filter(Task.user_id == u.id).order_by(Task.created_at.desc()).all()

@app.post("/api/preview")
async def generate_preview(req: PreviewRequest, u=Depends(get_current_user)):
    p_id = str(uuid.uuid4())[:8]
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(backend_dir, "static/previews", p_id)
    os.makedirs(out_dir, exist_ok=True)
    scripts_dir = os.path.join(backend_dir, "scripts")
    cmd = ["python", "generate_samples.py", "--wakeword", req.wakeword, "--output_dir", out_dir, "--num_samples", "3"]
    subprocess.run(cmd, check=True, cwd=scripts_dir)
    return {"urls": [f"/site/static/previews/{p_id}/{f}" for f in os.listdir(out_dir) if f.endswith(".wav")]}

@app.post("/api/generate-similar-words")
async def generate_similar_words(req: SimilarWordsRequest):
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
    cmd = ["python", "v2_gen_word_list.py", "--wakeword", req.wakeword]
    try:
        res = subprocess.check_output(cmd, cwd=scripts_dir, text=True)
        match = re.search(r"WORDS:(.*)", res)
        if match:
            words = [w.strip() for w in match.group(1).split(",") if w.strip()]
            return {"similar_words": words}
        return {"similar_words": []}
    except: return {"similar_words": [req.wakeword + "测试", "近似音1", "近似音2"]}

@app.post("/api/train")
async def start_training(req: TrainRequest, bt: BackgroundTasks, u=Depends(get_current_user), db: Session = Depends(get_db)):
    task_id = str(uuid.uuid4())[:8]
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    task_dir = os.path.join(backend_dir, "static/datasets", task_id)
    os.makedirs(task_dir, exist_ok=True)
    generate_task_yaml(task_dir, req.wakeword, req.similar_words, req.num_samples, req.epochs)
    new_task = Task(id=task_id, user_id=u.id, wakeword=req.wakeword, similar_words=req.similar_words,
                    status="Pending", sub_status="等待中", params=req.dict(), current_step=1)
    db.add(new_task)
    db.commit()
    bt.add_task(run_v2_pipeline, task_id, 1)
    return {"task_id": task_id}

@app.post("/api/retry/{task_id}")
async def retry_task(task_id: str, bt: BackgroundTasks, step: int = None, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    start_from = step if step is not None else task.current_step
    task.status = "Pending"
    db.commit()
    bt.add_task(run_v2_pipeline, task_id, start_from)
    return {"message": "Retry started"}

@app.get("/api/models")
async def list_models(user=Depends(get_current_user), db: Session = Depends(get_db)):
    tasks = db.query(Task).filter(Task.user_id == user.id, Task.status == "Completed").all()
    return [{"id":t.id, "wakeword":t.wakeword, "params":t.params, "download_url":f"/site/static/datasets/{t.id}/beary_custom.onnx"} for t in tasks]

@app.get("/api/status/{task_id}")
async def get_status(task_id: str, db: Session = Depends(get_db)):
    return db.query(Task).filter(Task.id == task_id).first()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
