"""Windows-local VideoAI Studio. Run: python windows_studio.py
Requires: pip install fastapi uvicorn httpx python-multipart
SSH tunnel must forward localhost:17860 to RunPod localhost:7860.
"""
import os
from pathlib import Path
import httpx
import base64
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, Response
from contextlib import asynccontextmanager

HERE = Path(__file__).resolve().parent
REMOTE = "http://127.0.0.1:17860"
USERNAME = "videoai"

@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=5, read=3600), follow_redirects=False) as client:
        app.state.client = client
        yield

app = FastAPI(title="VideoAI Studio Windows", lifespan=lifespan)

@app.get("/")
async def home():
    return FileResponse(HERE / "studio.html")

@app.get("/api/local-status")
async def local_status():
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(REMOTE + "/api/files", auth=(USERNAME, os.getenv("VIDEOAI_PASSWORD", "")))
        return {"connected": r.status_code == 200, "http_status": r.status_code}
    except httpx.HTTPError:
        return {"connected": False, "http_status": None}

@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(path: str, request: Request):
    password = os.getenv("VIDEOAI_PASSWORD")
    if not password:
        raise HTTPException(503, "Set VIDEOAI_PASSWORD on Windows first")
    # Stream large uploads instead of buffering video in memory.
    headers = {}
    if request.headers.get("content-type"):
        headers["content-type"] = request.headers["content-type"]
    async def incoming():
        async for chunk in request.stream():
            yield chunk
    headers["authorization"] = "Basic " + base64.b64encode((USERNAME + ":" + password).encode()).decode()
    client = request.app.state.client
    try:
        req = client.build_request(
            request.method, REMOTE + "/api/" + path,
            params=request.query_params, headers=headers,
            content=incoming() if request.method in ("POST", "PUT") else None,
        )
        response = await client.send(req, stream=True)
    except httpx.HTTPError as exc:
        raise HTTPException(502, "RunPod SSH tunnel unavailable: " + str(exc))
    passthrough = {}
    for key in ("content-type", "content-disposition", "content-length", "accept-ranges", "content-range"):
        if key in response.headers:
            passthrough[key] = response.headers[key]
    async def chunks():
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await response.aclose()
    return StreamingResponse(chunks(), status_code=response.status_code, headers=passthrough)

if __name__ == "__main__":
    import uvicorn
    print("VideoAI Studio: http://127.0.0.1:8765")
    uvicorn.run(app, host="127.0.0.1", port=8765)
