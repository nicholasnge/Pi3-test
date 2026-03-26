#!/usr/bin/env python3
"""
pi3_to_colmap.py

Replace COLMAP initialization with Pi3X for 3DGS training.
Runs Pi3X on a raw image directory and writes COLMAP text-format files
(cameras.txt, images.txt, points3D.txt) that 3DGS reads directly.

Usage:
    python pi3_to_colmap.py \
        --images ~/3DGSDATASETS/bonsai/images_2 \
        --output ~/3DGSDATASETS/bonsai/sparse_pi3/0

Then point your 3DGS trainer at:
    --source_path ~/3DGSDATASETS/bonsai \
    --model_path ~/3DGSDATASETS/bonsai/output_pi3 \
    with the sparse dir containing cameras.txt / images.txt / points3D.txt
"""

import argparse
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

from pi3.models.pi3x import Pi3X
from pi3.pipe.pi3x_vo import Pi3XVO
from pi3.utils.geometry import recover_intrinsic_from_rays_d


# ─────────────────────────────────────────────────────────────────────────────
# Image loading (mirrors Pi3's load_images_as_tensor but also returns filenames
# and original dimensions for COLMAP output)
# ─────────────────────────────────────────────────────────────────────────────

def load_images(image_dir, interval=1, pixel_limit=255000):
    """
    Returns:
        tensor   : (N, 3, H, W) float32 in [0,1]  (Pi3-resized)
        filenames: list of N original filenames (sorted)
        orig_hw  : (H_orig, W_orig) of the first image (assumes all same size)
        model_hw : (H_model, W_model) after Pi3 resizing
    """
    exts = ('.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.JPEG')
    all_files = sorted([f for f in os.listdir(image_dir) if f.endswith(exts)])
    selected  = all_files[::interval]

    first_img  = Image.open(os.path.join(image_dir, selected[0])).convert('RGB')
    W_orig, H_orig = first_img.size

    scale = math.sqrt(pixel_limit / (W_orig * H_orig))
    k = round(W_orig * scale / 14)
    m = round(H_orig * scale / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_orig / H_orig:
            k -= 1
        else:
            m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    print(f"Pi3 input size : {TARGET_W}x{TARGET_H}  (original: {W_orig}x{H_orig})")

    to_tensor = transforms.ToTensor()
    tensors = []
    for fname in selected:
        img = Image.open(os.path.join(image_dir, fname)).convert('RGB')
        img = img.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        tensors.append(to_tensor(img))

    return torch.stack(tensors), selected, (H_orig, W_orig), (TARGET_H, TARGET_W)


# ─────────────────────────────────────────────────────────────────────────────
# Rotation matrix → quaternion (qw, qx, qy, qz)
# ─────────────────────────────────────────────────────────────────────────────

def rotmat_to_quat(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s  = 0.5 / math.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (R[2,1] - R[1,2]) * s
        qy = (R[0,2] - R[2,0]) * s
        qz = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s  = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        qw = (R[2,1] - R[1,2]) / s
        qx = 0.25 * s
        qy = (R[0,1] + R[1,0]) / s
        qz = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s  = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        qw = (R[0,2] - R[2,0]) / s
        qx = (R[0,1] + R[1,0]) / s
        qy = 0.25 * s
        qz = (R[1,2] + R[2,1]) / s
    else:
        s  = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        qw = (R[1,0] - R[0,1]) / s
        qx = (R[0,2] + R[2,0]) / s
        qy = (R[1,2] + R[2,1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz])


# ─────────────────────────────────────────────────────────────────────────────
# COLMAP text writers
# ─────────────────────────────────────────────────────────────────────────────

def write_cameras_txt(out_dir, width, height, fx, fy, cx, cy):
    """Single shared PINHOLE camera."""
    path = os.path.join(out_dir, 'cameras.txt')
    with open(path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
    print(f"Wrote {path}")


def write_images_txt(out_dir, filenames, poses_c2w):
    """
    poses_c2w : (N, 4, 4) numpy, camera-to-world, OpenCV convention.
    COLMAP wants world-to-camera: R_w2c = R_c2w^T, t_w2c = -R_c2w^T @ t_c2w
    """
    path = os.path.join(out_dir, 'images.txt')
    with open(path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(filenames)}\n")
        for i, (fname, pose) in enumerate(zip(filenames, poses_c2w)):
            R_c2w = pose[:3, :3]
            t_c2w = pose[:3,  3]
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            qw, qx, qy, qz = rotmat_to_quat(R_w2c)
            tx, ty, tz = t_w2c
            f.write(f"{i+1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{tx:.9f} {ty:.9f} {tz:.9f} 1 {fname}\n")
            f.write("\n")   # empty 2D-3D correspondences (not needed by 3DGS)
    print(f"Wrote {path}")


def write_points3d_txt(out_dir, xyz, rgb):
    """
    xyz : (M, 3) float32
    rgb : (M, 3) uint8
    """
    path = os.path.join(out_dir, 'points3D.txt')
    with open(path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(xyz)}, mean track length: 0\n")
        for i, (pt, col) in enumerate(zip(xyz, rgb)):
            f.write(f"{i+1} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                    f"{int(col[0])} {int(col[1])} {int(col[2])} 0.0\n")
    print(f"Wrote {path}  ({len(xyz):,} points)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images',   required=True,
                        help='Directory of input images (e.g. scene/images_2)')
    parser.add_argument('--output',   required=True,
                        help='Output dir for COLMAP files (e.g. scene/sparse_pi3/0)')
    parser.add_argument('--ckpt',     default=None,
                        help='Optional path to Pi3X checkpoint (.safetensors or .pth)')
    parser.add_argument('--interval', type=int, default=1,
                        help='Frame sampling interval (default: 1 = use all)')
    parser.add_argument('--max_pts',  type=int, default=200_000,
                        help='Max points to write to points3D.txt (default: 200000)')
    parser.add_argument('--chunk',    type=int, default=32,
                        help='VO chunk size (default: 32; use 64 on A100-80)')
    parser.add_argument('--overlap',  type=int, default=8,
                        help='VO overlap (default: 8; use 12 on A100-80)')
    parser.add_argument('--conf_thr', type=float, default=0.1,
                        help='Confidence threshold for point filtering (default: 0.1)')
    parser.add_argument('--device',   default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    # ── 1. Load images ────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading images from {args.images}")
    imgs_tensor, filenames, (H_orig, W_orig), (H_model, W_model) = \
        load_images(args.images, interval=args.interval)
    N = len(filenames)
    print(f"      {N} images  |  original {W_orig}x{H_orig}  |  model {W_model}x{H_model}")

    imgs_batch = imgs_tensor.unsqueeze(0).to(device)   # (1, N, 3, H, W)

    # ── 2. Load model ─────────────────────────────────────────────────────────
    print(f"\n[2/5] Loading Pi3X model...")
    # Keep multimodal enabled — ray_embed is needed for intrinsic conditioning.
    if args.ckpt is not None:
        model = Pi3X().eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            model.load_state_dict(load_file(args.ckpt), strict=False)
        else:
            model.load_state_dict(
                torch.load(args.ckpt, map_location=device, weights_only=False), strict=False)
    else:
        model = Pi3X.from_pretrained('yyfz233/Pi3X').eval()
    model = model.to(device)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # ── 3. Estimate intrinsics on first chunk ─────────────────────────────────
    print(f"\n[3/5] Estimating intrinsics from first chunk...")
    probe_n = min(args.chunk, N)
    probe_imgs = imgs_batch[:, :probe_n]

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        probe_out = model(probe_imgs, with_prior=False)

    # Use the official least-squares intrinsic recovery from predicted ray directions.
    rays = probe_out['rays'][0].float()   # (probe_n, H, W, 3)
    K = recover_intrinsic_from_rays_d(rays, force_center_principal_point=True)  # (probe_n, 3, 3)
    fx_model = K[:, 0, 0].mean().item()
    fy_model = K[:, 1, 1].mean().item()
    cx_model = K[:, 0, 2].mean().item()
    cy_model = K[:, 1, 2].mean().item()
    print(f"      fx={fx_model:.1f}  fy={fy_model:.1f}  at model res {W_model}x{H_model}")

    # Rescale to original image resolution
    scale_x = W_orig / W_model
    scale_y = H_orig / H_model
    fx_orig = fx_model * scale_x
    fy_orig = fy_model * scale_y
    cx_orig = cx_model * scale_x
    cy_orig = cy_model * scale_y
    print(f"      fx={fx_orig:.1f}  fy={fy_orig:.1f}  cx={cx_orig:.1f}  cy={cy_orig:.1f}  at original res {W_orig}x{H_orig}")

    del probe_out, rays
    torch.cuda.empty_cache()

    # ── 4. Run Pi3XVO for poses + global point cloud ──────────────────────────
    print(f"\n[4/5] Running Pi3XVO (chunk={args.chunk}, overlap={args.overlap})...")
    pipe = Pi3XVO(model)

    vo_out = pipe(
        imgs_batch,
        chunk_size=args.chunk,
        overlap=args.overlap,
        conf_thre=args.conf_thr,
        dtype=dtype,
        intrinsics=(fx_model, fy_model, cx_model, cy_model),
    )
    # shapes: (1, N, H, W, 3),  (1, N, 4, 4),  (1, N, H, W)
    all_points = vo_out['points'][0].float().cpu()          # (N, H, W, 3)
    all_poses  = vo_out['camera_poses'][0].float().cpu()    # (N, 4, 4)
    all_conf   = vo_out['conf'][0].float().cpu()            # (N, H, W)

    del vo_out
    torch.cuda.empty_cache()

    # ── 5. Write COLMAP files ─────────────────────────────────────────────────
    print(f"\n[5/5] Writing COLMAP files to {args.output}")

    # cameras.txt
    write_cameras_txt(args.output, W_orig, H_orig, fx_orig, fy_orig, cx_orig, cy_orig)

    # images.txt
    write_images_txt(args.output, filenames, all_poses.numpy())

    # points3D.txt — confidence-filtered + random subsample.
    # Depth-edge filtering is already applied inside Pi3XVO (confidence zeroed at edges),
    # so the confidence mask alone is sufficient here.
    valid_mask = all_conf > args.conf_thr                   # (N, H, W)

    pts_all  = all_points[valid_mask].numpy()               # (M, 3)
    # get colours from original (model-res) images
    imgs_np  = imgs_tensor.permute(0, 2, 3, 1).numpy()     # (N, H, W, 3)
    rgb_all  = (imgs_np[valid_mask.numpy()] * 255).astype(np.uint8)  # (M, 3)

    M = len(pts_all)
    print(f"      Valid points after filtering: {M:,}")
    if M > args.max_pts:
        idx = np.random.choice(M, args.max_pts, replace=False)
        pts_all = pts_all[idx]
        rgb_all = rgb_all[idx]

    write_points3d_txt(args.output, pts_all, rgb_all)

    print("\nDone!")
    print(f"  cameras.txt  → {args.output}/cameras.txt")
    print(f"  images.txt   → {args.output}/images.txt")
    print(f"  points3D.txt → {args.output}/points3D.txt")
    print(f"\nTo train 3DGS, set --source_path to the scene root and make sure")
    print(f"  <scene>/sparse/0/   (or wherever your trainer reads COLMAP from)")
    print(f"  contains these three files.")


if __name__ == '__main__':
    main()
