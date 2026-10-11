# VideoAI Studio – Windows + RunPod

A Windows gépen fut a webes kezelőfelület, a RunPod RTX 4090-en a videófeldolgozás.

## RunPod (egy terminál)
```bash
cd /workspace/VideoAI
export VIDEOAI_PASSWORD='VALASSZ_EROS_JELSZOT'
/workspace/comfy-venv/bin/python -m uvicorn studio:app --host 127.0.0.1 --port 7860
```

## Windows – SSH alagút (másik terminál)
A RunPod SSH Connect paneljén megadott SSH felhasználóval/kulccsal:
```powershell
ssh -N -L 17860:127.0.0.1:7860 root@213.173.110.223 -p 15400 -i "C:\UTVONAL\runpod_key"
```
A kulcs és a port Podonként változhat. A parancsablakot hagyd nyitva. Az SSH-szervernek engedélyeznie kell a TCP port forwardingot.

## Windows – Webes felület (harmadik terminál)
A `studio.html` és `windows_studio.py` fájl legyen ugyanabban a mappában.
```powershell
py -m pip install fastapi uvicorn httpx python-multipart
$env:VIDEOAI_PASSWORD="VALASSZ_EROS_JELSZOT"
py windows_studio.py
```
Böngésző: http://127.0.0.1:8765

A jelszónak mindkét gépen egyeznie kell. Ne tedd bele a GitHub-repóba.
