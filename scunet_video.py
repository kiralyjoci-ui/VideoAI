import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCUNET_DIR = Path('/workspace/SCUNet')
sys.path.insert(0, str(SCUNET_DIR / 'models'))
from network_scunet import SCUNet


def probe_video(path):
    raw = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-of', 'json', str(path)
    ], text=True)
    stream = json.loads(raw)['streams'][0]
    return int(stream['width']), int(stream['height']), stream['r_frame_rate']


def main():
    parser = argparse.ArgumentParser(description='SCUNet GPU video denoiser, frame-by-frame')
    parser.add_argument('--input', default='/workspace/video_tests/C_aggressive.mp4')
    parser.add_argument('--output', default='/workspace/video_tests/SCUNet_test.mp4')
    parser.add_argument('--model', default='/workspace/SCUNet/model_zoo/scunet_color_real_psnr.pth')
    parser.add_argument('--frames', type=int, default=1, help='Number of frames to process; 0 for all')
    parser.add_argument('--tile', type=int, default=256, help='Tile size, divisible by 64')
    parser.add_argument('--overlap', type=int, default=32, help='Tile overlap')
    args = parser.parse_args()
    if args.tile % 64 or args.tile <= args.overlap or args.overlap < 0:
        parser.error('Tile must be a multiple of 64 and larger than overlap')
    src, dst, weights = map(Path, (args.input, args.output, args.model))
    if not src.is_file() or not weights.is_file() or weights.stat().st_size < 1000000:
        parser.error('Missing input video or valid SCUNet model')
    if src.resolve() == dst.resolve():
        parser.error('Input and output must differ')
    dst.parent.mkdir(parents=True, exist_ok=True)
    w, h, fps = probe_video(src)
    device = torch.device('cuda')
    model = SCUNet(in_nc=3, config=[2]*7, dim=64).eval()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    model.to(device)
    print(f'SCUNet loaded | {w}x{h} | {fps} fps | tile {args.tile}', flush=True)
    reader = subprocess.Popen([
        'ffmpeg', '-nostdin', '-v', 'error', '-i', str(src),
        '-map', '0:v:0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    writer = subprocess.Popen([
        'ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', fps,
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
        '-crf', '17', '-pix_fmt', 'yuv420p', str(dst)
    ], stdin=subprocess.PIPE)
    n = 0
    start = time.monotonic()
    try:
        with torch.inference_mode():
            while args.frames == 0 or n < args.frames:
                raw = reader.stdout.read(w*h*3)
                if len(raw) != w*h*3:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
                x = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
                pad_h, pad_w = (-h) % 64, (-w) % 64
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate') if pad_h or pad_w else x
                _, _, hp, wp = x.shape
                acc = torch.zeros_like(x)
                norm = torch.zeros((1, 1, hp, wp), device=device)
                step = args.tile - args.overlap
                ys = list(range(0, max(1, hp-args.tile+1), step))
                xs = list(range(0, max(1, wp-args.tile+1), step))
                ys.append(max(0, hp-args.tile))
                xs.append(max(0, wp-args.tile))
                for y in sorted(set(ys)):
                    for xx in sorted(set(xs)):
                        patch = x[:, :, y:y+args.tile, xx:xx+args.tile]
                        py, px = (-patch.shape[2]) % 64, (-patch.shape[3]) % 64
                        if py or px:
                            patch = F.pad(patch, (0, px, 0, py), mode='replicate')
                        pred = model(patch).clamp_(0, 1)
                        actual_h = min(args.tile, hp-y)
                        actual_w = min(args.tile, wp-xx)
                        acc[:, :, y:y+actual_h, xx:xx+actual_w] += pred[:, :, :actual_h, :actual_w]
                        norm[:, :, y:y+actual_h, xx:xx+actual_w] += 1
                y = (acc / norm)[:, :, :h, :w]
                result = (y[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy())
                writer.stdin.write(result.tobytes())
                n += 1
                print(f'Frame {n}/{args.frames or "all"} | {time.monotonic()-start:.1f}s', flush=True)
    finally:
        reader.stdout.close()
        reader.terminate()
        writer.stdin.close()
        code = writer.wait()
        reader.wait()
    print(f'Finished: {dst} | {n} frames | ffmpeg exit={code}', flush=True)
    if code != 0 or n == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
