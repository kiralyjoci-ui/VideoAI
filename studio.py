"""VideoAI Studio web API. Run: uvicorn studio:app --host 0.0.0.0 --port 7860"""
import json
import base64
import secrets
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path('/workspace/video_ai_studio')
INPUTS = ROOT / 'inputs'
OUTPUTS = ROOT / 'outputs'
TEMP = ROOT / 'temp'
EXISTING = Path('/workspace/video_tests')
for folder in (INPUTS, OUTPUTS, TEMP):
    folder.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent
PYTHON = sys.executable
app = FastAPI(title='VideoAI Studio')

@app.middleware('http')
async def password_protection(request: Request, call_next):
    password = os.environ.get('VIDEOAI_PASSWORD')
    if not password:
        return Response('Set VIDEOAI_PASSWORD before starting the server', status_code=503)
    authorization = request.headers.get('authorization', '')
    try:
        scheme, token = authorization.split(' ', 1)
        username, supplied = base64.b64decode(token).decode().split(':', 1)
        valid = scheme.lower() == 'basic' and secrets.compare_digest(username, 'videoai') and secrets.compare_digest(supplied, password)
    except (ValueError, UnicodeError, Exception):
        valid = False
    if not valid:
        return Response('Authentication required', status_code=401, headers={'WWW-Authenticate':'Basic realm="VideoAI Studio"'})
    return await call_next(request)

jobs = {}
lock = threading.Lock()
active = None
EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv', '.webm', '.mpg', '.mpeg'}

class Settings(BaseModel):
    filename: str
    ffmpeg: bool = True
    gamma: float = Field(1.9, ge=0.1, le=5)
    brightness: float = Field(0.06, ge=-1, le=1)
    contrast: float = Field(1.12, ge=0.1, le=5)
    ffmpeg_filter: str = 'basic'
    denoise_luma_spatial: float = Field(6, ge=0, le=30)
    denoise_chroma_spatial: float = Field(4, ge=0, le=30)
    denoise_luma_temporal: float = Field(9, ge=0, le=30)
    denoise_chroma_temporal: float = Field(6, ge=0, le=30)
    sharpen: float = Field(0.6, ge=-2, le=2)
    scunet: bool = True
    scunet_model: str = 'scunet_color_real_psnr'
    realesrgan: bool = True
    scale: int = Field(2)
    preview_second: float = Field(1, ge=0)
    test_seconds: float = Field(0, ge=0, le=60)

def allowed_path(name):
    if name.startswith('tests/'):
        basename = name[6:]
        folder = EXISTING
    else:
        basename = name
        folder = INPUTS
    if Path(basename).name != basename or Path(basename).suffix.lower() not in EXTENSIONS:
        raise HTTPException(400, 'Invalid video filename')
    path = folder / basename
    if not path.is_file():
        raise HTTPException(404, 'Video not found')
    return path

def probe(path):
    result = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,duration,nb_frames',
        '-show_entries', 'format=duration', '-of', 'json', str(path)
    ], text=True)
    obj = json.loads(result)
    stream = obj['streams'][0]
    rate = stream.get('avg_frame_rate', '30/1')
    a, b = (int(v) for v in rate.split('/'))
    duration = float(stream.get('duration') or obj.get('format', {}).get('duration') or 0)
    return {'width': stream['width'], 'height': stream['height'],
            'fps': a/b if b else 30, 'duration': duration}

def validate(s):
    allowed = {'scunet_color_real_psnr', 'scunet_color_real_gan',
               'scunet_color_15', 'scunet_color_25', 'scunet_color_50'}
    if s.scunet_model not in allowed or s.scale not in (1, 2, 4) or s.ffmpeg_filter not in ('basic', 'legacy_aggressive'):
        raise HTTPException(400, 'Invalid model or scale')
    if s.scunet and not Path('/workspace/SCUNet/model_zoo', s.scunet_model + '.pth').is_file():
        raise HTTPException(400, 'SCUNet model missing')
    if s.realesrgan and not Path('/workspace/ComfyUI/models/upscale_models/RealESRGAN_x4plus.pth').is_file():
        raise HTTPException(400, 'RealESRGAN model missing')

@app.get('/')
def home():
    return FileResponse(HERE / 'studio.html')

@app.get('/api/files')
def files():
    result = []
    for folder, prefix in ((INPUTS, ''), (EXISTING, 'tests/')):
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in EXTENSIONS:
                try:
                    result.append({'name': prefix+p.name, 'size': p.stat().st_size, **probe(p)})
                except Exception:
                    result.append({'name': prefix+p.name, 'size': p.stat().st_size})
    return result

@app.post('/api/upload')
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or '').name
    if not name or Path(name).suffix.lower() not in EXTENSIONS:
        raise HTTPException(400, 'Unsupported video format')
    safe = uuid.uuid4().hex[:8] + '_' + re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
    target = INPUTS / safe
    try:
        with target.open('wb') as out:
            while chunk := await file.read(4 * 1024 * 1024):
                out.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return {'name': safe}

@app.get('/api/video/{name}')
def video(name: str):
    return FileResponse(allowed_path(name), media_type='video/mp4')

@app.get('/api/output/{name}')
def output(name: str):
    if Path(name).name != name or not name.endswith('.mp4'):
        raise HTTPException(400)
    path = OUTPUTS / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type='video/mp4', filename=name)

def execute(job, command, stage, expected=0):
    job['stage'] = stage
    job['stage_frame'] = 0
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True)
    job['process'] = proc
    try:
        for line in proc.stdout:
            line = line.strip()
            if line:
                job['log'] = line[-350:]
            m = re.search(r'Frame\s+(\d+)/', line)
            if m:
                job['stage_frame'] = int(m.group(1))
                if expected:
                    job['percent'] = round(100 * min(int(m.group(1)), expected) / expected, 1)
            if job['cancel']:
                proc.terminate()
                break
        try:
            code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait()
        if job['cancel']:
            raise RuntimeError('Cancelled')
        if code:
            raise RuntimeError(stage + ': ' + job['log'])
    finally:
        job['process'] = None

def pipeline(job, settings, preview=False):
    global active
    s = settings
    jid = job['id']
    scratch = TEMP / jid
    scratch.mkdir(parents=True, exist_ok=True)
    src = allowed_path(s.filename)
    current = src
    original = src
    try:
        duration = probe(src)['duration']
        seconds = min(duration, s.test_seconds) if s.test_seconds else duration
        if preview:
            # Feed real consecutive frames to temporal filters (hqdn3d).
            # The selected frame is the final frame of this short segment.
            fps = probe(src)['fps']
            target = max(0.0, min(s.preview_second, max(0.0, duration - 1.0 / fps)))
            clip_start = max(0.0, target - 1.5)
            clip_length = min(duration - clip_start, target - clip_start + 1.0 / fps)
            clip = scratch / 'preview_source.mp4'
            execute(job, ['ffmpeg','-y','-v','error','-ss',str(clip_start),
                          '-i',str(src),'-t',str(clip_length),
                          '-map','0:v:0','-an','-c:v','libx264','-crf','17',
                          '-pix_fmt','yuv420p',str(clip)], 'Preparing temporal preview')
            current = clip
            seconds = clip_length
        elif s.test_seconds:
            trimmed = scratch / 'trim.mp4'
            execute(job, ['ffmpeg','-y','-v','error','-i',str(src),'-t',str(seconds),
                          '-map','0:v:0','-an',
                          '-c:v','libx264','-crf','17',str(trimmed)], 'Trimming test')
            current = trimmed
        if s.ffmpeg:
            adjusted = scratch / 'corrected.mp4'
            vf = f'eq=gamma={s.gamma}:brightness={s.brightness}:contrast={s.contrast}'
            if s.ffmpeg_filter == 'legacy_aggressive':
                vf += (f',hqdn3d={s.denoise_luma_spatial}:{s.denoise_chroma_spatial}:'
                       f'{s.denoise_luma_temporal}:{s.denoise_chroma_temporal},'
                       f'unsharp=5:5:{s.sharpen}')
            execute(job, ['ffmpeg','-y','-v','error','-i',str(current),'-vf',vf,
                          '-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p',
                          str(adjusted)], 'FFmpeg correction')
            current = adjusted
        info = probe(current)
        expected = max(1, round(info['duration'] * info['fps']))
        if s.scunet:
            denoised = scratch / 'scunet.mp4'
            execute(job, [PYTHON,str(HERE/'scunet_video.py'),
                          '--input',str(current),'--output',str(denoised),
                          '--model',f'/workspace/SCUNet/model_zoo/{s.scunet_model}.pth',
                          '--frames','0'], 'SCUNet', expected)
            current = denoised
        if s.realesrgan:
            enlarged = scratch / 'upscaled.mp4'
            execute(job, [PYTHON,str(HERE/'realesrgan_video.py'),
                          '--input',str(current),'--output',str(enlarged),
                          '--frames','0','--scale',str(s.scale)], 'RealESRGAN', expected)
            current = enlarged
        if preview:
            final = OUTPUTS / (jid + '.png')
            # Reverse only the short preview segment to take its final frame.
            execute(job, ['ffmpeg','-y','-v','error','-i',str(current),
                          '-vf','reverse','-frames:v','1',str(final)], 'Saving preview')
        else:
            final = OUTPUTS / (jid + '.mp4')
            # Retain source audio if present, and preserve processed video timestamps.
            execute(job, ['ffmpeg','-y','-v','error','-i',str(current),
                          '-i',str(original),'-map','0:v:0','-map','1:a?',
                          '-c:v','copy','-c:a','aac','-shortest',str(final)], 'Finalizing audio')
        job['result'] = '/api/preview/' + jid if preview else '/api/output/' + final.name
        job['percent'] = 100
        job['stage'] = 'Done'
        job['status'] = 'done'
    except Exception as exc:
        job['status'] = 'cancelled' if job['cancel'] else 'error'
        job['log'] = str(exc)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        with lock:
            if active == jid:
                active = None

@app.post('/api/start')
def start(s: Settings, preview: bool = False):
    global active
    allowed_path(s.filename)
    validate(s)
    with lock:
        if active is not None:
            raise HTTPException(409, 'Another job is running')
        jid = uuid.uuid4().hex[:12]
        job = {'id': jid, 'status': 'running', 'stage': 'Starting',
               'stage_frame': 0, 'percent': 0, 'log': '', 'result': None,
               'cancel': False, 'process': None, 'started': time.time()}
        jobs[jid] = job
        active = jid
    threading.Thread(target=pipeline, args=(job, s, preview), daemon=True).start()
    return {'id': jid}

@app.get('/api/jobs/{jid}')
def job_status(jid: str):
    if jid not in jobs:
        raise HTTPException(404)
    j = jobs[jid]
    return {k:v for k,v in j.items() if k != 'process'}

@app.post('/api/jobs/{jid}/cancel')
def cancel(jid: str):
    if jid not in jobs:
        raise HTTPException(404)
    j = jobs[jid]
    j['cancel'] = True
    if j['process'] and j['process'].poll() is None:
        j['process'].terminate()
    return {'ok': True}

@app.get('/api/preview/{jid}')
def preview_result(jid: str):
    path = OUTPUTS / (jid + '.png')
    if jid not in jobs or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type='image/png')
