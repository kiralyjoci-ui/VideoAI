import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

def video_info(path):
    result = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'json', str(path)
    ], text=True)
    s = json.loads(result)['streams'][0]
    return int(s['width']), int(s['height']), s['r_frame_rate']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='/workspace/video_tests/SCUNet_test.mp4')
    p.add_argument('--output', default='/workspace/video_tests/RealESRGAN_2x_test.mp4')
    p.add_argument('--model', default='/workspace/ComfyUI/models/upscale_models/RealESRGAN_x4plus.pth')
    p.add_argument('--frames', type=int, default=10)
    p.add_argument('--tile', type=int, default=256)
    args = p.parse_args()
    if args.tile < 64 or args.frames < 0:
        p.error('tile must be >=64 and frames >=0')
    source, dest, weights = Path(args.input), Path(args.output), Path(args.model)
    if not source.is_file() or not weights.is_file():
        p.error('Input video or model not found')
    if source.resolve() == dest.resolve():
        p.error('Output must differ from input')
    dest.parent.mkdir(parents=True, exist_ok=True)

    # ComfyUI includes the ESRGAN RRDB architecture and uses this checkpoint already.
    sys.path.insert(0, '/workspace/ComfyUI')
    from comfy.utils import load_torch_file, tiled_scale
    from comfy_extras.chainner_models import model_loading

    state = load_torch_file(str(weights), safe_load=True)
    if 'params_ema' in state:
        state = state['params_ema']
    elif 'params' in state:
        state = state['params']
    model = model_loading.load_state_dict(state).eval()
    device = torch.device('cuda')
    model.to(device)

    w, h, fps = video_info(source)
    scale = 2
    reader = subprocess.Popen([
        'ffmpeg', '-nostdin', '-v', 'error', '-i', str(source),
        '-map', '0:v:0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    writer = subprocess.Popen([
        'ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{w*scale}x{h*scale}', '-r', fps,
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
        '-crf', '17', '-pix_fmt', 'yuv420p', str(dest)
    ], stdin=subprocess.PIPE)
    print(f'Upscaling {w}x{h} -> {w*scale}x{h*scale}, {fps} fps', flush=True)
    count = 0
    start = time.monotonic()
    try:
        with torch.inference_mode():
            while args.frames == 0 or count < args.frames:
                raw = reader.stdout.read(w*h*3)
                if len(raw) != w*h*3:
                    break
                frame = np.frombuffer(raw, np.uint8).reshape(h,w,3).copy()
                x = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).float().to(device)/255
                # ComfyUI tiled upscale, avoiding full-resolution GPU inference.
                upscaled = tiled_scale(
                    x, lambda tile: model(tile), tile_x=args.tile,
                    tile_y=args.tile, overlap=32, upscale_amount=model.scale,
                    out_channels=3, output_device=device,
                    pbar=None
                )
                # RealESRGAN x4 model: downsample x4 result to requested x2.
                out = torch.nn.functional.interpolate(
                    upscaled, size=(h*scale,w*scale), mode='bicubic', align_corners=False
                ).clamp(0,1)
                rgb = (out[0].permute(1,2,0)*255).round().byte().cpu().numpy()
                writer.stdin.write(rgb.tobytes())
                count += 1
                print(f'Frame {count}/{args.frames or "all"} | {time.monotonic()-start:.1f}s', flush=True)
    finally:
        reader.stdout.close()
        reader.terminate()
        writer.stdin.close()
        code = writer.wait()
        reader.wait()
    print(f'Finished: {dest} | frames={count} | ffmpeg={code}', flush=True)
    if code != 0 or count == 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
