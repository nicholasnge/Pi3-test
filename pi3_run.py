#!/usr/bin/env python3
"""
pi3_run.py

For a single scene:
  1. Runs Pi3XVO inference on images/
  2. Saves PLY point cloud  (same as example_vo.py)
  3. Saves COLMAP text format (cameras.txt, images.txt, points3D.txt)
  4. Reads ground-truth COLMAP poses from sparse/
  5. Aligns Pi3 poses to GT via Sim3, then reports per-view and aggregate errors

Usage:
    python pi3_run.py --scene_dir ~/3DGSDATASETS/bonsai \
                      --ckpt ~/Pi3/ckpts/model.safetensors \
                      --out_dir ~/Pi3/outputs/bonsai
"""

import argparse, math, os, struct
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

from pi3.models.pi3x import Pi3X
from pi3.pipe.pi3x_vo import Pi3XVO
from pi3.utils.geometry import depth_edge
from pi3.utils.basic import write_ply


# ─────────────────────────────────────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────────────────────────────────────

def load_images(image_dir, interval=1, pixel_limit=255000):
    exts = ('.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.JPEG')
    all_files = sorted([f for f in os.listdir(image_dir) if f.endswith(exts)])
    selected  = all_files[::interval]

    first = Image.open(os.path.join(image_dir, selected[0])).convert('RGB')
    W0, H0 = first.size
    scale = math.sqrt(pixel_limit / (W0 * H0))
    k, m  = round(W0 * scale / 14), round(H0 * scale / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W0 / H0: k -= 1
        else:                m -= 1
    TW, TH = max(1, k) * 14, max(1, m) * 14
    print(f"  Model input : {TW}x{TH}  (original {W0}x{H0})")

    to_t = transforms.ToTensor()
    tensors = []
    for f in selected:
        img = Image.open(os.path.join(image_dir, f)).convert('RGB')
        tensors.append(to_t(img.resize((TW, TH), Image.Resampling.LANCZOS)))

    return torch.stack(tensors), selected, (H0, W0), (TH, TW)


# ─────────────────────────────────────────────────────────────────────────────
# Focal length estimation from local_points
# ─────────────────────────────────────────────────────────────────────────────

def estimate_focal(local_points, conf, H, W):
    device = local_points.device
    ys = torch.arange(H, device=device).float()
    xs = torch.arange(W, device=device).float()
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    dx, dy = gx - W / 2.0, gy - H / 2.0

    xc, yc, zc = local_points[..., 0], local_points[..., 1], local_points[..., 2]
    vx = (zc > 0) & (dx.abs() > W * 0.05) & (conf > 0.15)
    vy = (zc > 0) & (dy.abs() > H * 0.05) & (conf > 0.15)

    with torch.no_grad():
        rx = xc / zc.clamp(min=1e-6)
        ry = yc / zc.clamp(min=1e-6)
        fxv = (dx.unsqueeze(0).abs() / rx.abs().clamp(min=1e-6))[vx]
        fyv = (dy.unsqueeze(0).abs() / ry.abs().clamp(min=1e-6))[vy]

    fx = fxv.abs().median().item() if fxv.numel() > 0 else float(max(H, W))
    fy = fyv.abs().median().item() if fyv.numel() > 0 else float(max(H, W))
    return fx, fy


# ─────────────────────────────────────────────────────────────────────────────
# Rotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def rotmat_to_quat(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])

def quat_to_rotmat(q):
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])

def rotation_angle_deg(R):
    """Angle of rotation matrix in degrees."""
    cos = (np.trace(R) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cos, -1, 1))))


# ─────────────────────────────────────────────────────────────────────────────
# COLMAP file readers (binary + text)
# ─────────────────────────────────────────────────────────────────────────────

def _read_next_bytes(f, num_bytes, fmt):
    data = f.read(num_bytes)
    return struct.unpack(fmt, data)

CAMERA_MODEL_NUM_PARAMS = {0:3, 1:4, 2:4, 3:5, 4:5, 5:8, 6:8, 7:5, 8:5, 9:6, 10:6}

def read_cameras_bin(path):
    cameras = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            cid  = struct.unpack('<I', f.read(4))[0]   # uint32
            mid  = struct.unpack('<i', f.read(4))[0]   # int32
            w    = struct.unpack('<Q', f.read(8))[0]   # uint64
            h    = struct.unpack('<Q', f.read(8))[0]   # uint64
            np_  = CAMERA_MODEL_NUM_PARAMS.get(mid, 4)
            params = struct.unpack(f'<{np_}d', f.read(8*np_))
            cameras[cid] = {'model_id': mid, 'width': w, 'height': h, 'params': params}
    return cameras

def read_images_bin(path):
    images = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            iid  = struct.unpack('<I', f.read(4))[0]
            qvec = np.array(struct.unpack('<4d', f.read(32)))   # qw qx qy qz
            tvec = np.array(struct.unpack('<3d', f.read(24)))
            cid  = struct.unpack('<I', f.read(4))[0]
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00': break
                name += c
            np2d = struct.unpack('<Q', f.read(8))[0]
            f.read(np2d * 24)   # skip 2D points
            images[iid] = {
                'name': name.decode(),
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': cid,
            }
    return images

def read_cameras_txt(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts = line.split()
            cid = int(parts[0])
            cameras[cid] = {
                'model': parts[1],
                'width': int(parts[2]),
                'height': int(parts[3]),
                'params': list(map(float, parts[4:]))
            }
    return cameras

def read_images_txt(path):
    images = {}
    with open(path) as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    i = 0
    iid = 0
    while i < len(lines):
        parts = lines[i].split()
        iid  = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])))
        tvec = np.array(list(map(float, parts[5:8])))
        cid  = int(parts[8])
        name = parts[9]
        images[iid] = {'name': name, 'qvec': qvec, 'tvec': tvec, 'camera_id': cid}
        i += 2   # skip 2D points line
    return images

def load_gt_colmap(sparse_dir):
    """Load GT cameras and images from sparse/ (tries binary then text)."""
    sparse_dir = Path(sparse_dir)
    # Try sparse/0/ first, then sparse/ directly
    for d in [sparse_dir / '0', sparse_dir]:
        cam_bin = d / 'cameras.bin'
        img_bin = d / 'images.bin'
        cam_txt = d / 'cameras.txt'
        img_txt = d / 'images.txt'
        if cam_bin.exists() and img_bin.exists():
            print(f"  GT COLMAP (binary): {d}")
            return read_cameras_bin(str(cam_bin)), read_images_bin(str(img_bin))
        if cam_txt.exists() and img_txt.exists():
            print(f"  GT COLMAP (text): {d}")
            return read_cameras_txt(str(cam_txt)), read_images_txt(str(img_txt))
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# COLMAP writers
# ─────────────────────────────────────────────────────────────────────────────

def write_cameras_txt(out_dir, W, H, fx, fy, cx, cy):
    with open(os.path.join(out_dir, 'cameras.txt'), 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {W} {H} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

def write_images_txt(out_dir, filenames, poses_c2w):
    with open(os.path.join(out_dir, 'images.txt'), 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(filenames)}\n")
        for i, (fname, pose) in enumerate(zip(filenames, poses_c2w)):
            R_c2w, t_c2w = pose[:3, :3], pose[:3, 3]
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            qw, qx, qy, qz = rotmat_to_quat(R_w2c)
            f.write(f"{i+1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{t_w2c[0]:.9f} {t_w2c[1]:.9f} {t_w2c[2]:.9f} 1 {fname}\n\n")

def write_points3d_txt(out_dir, xyz, rgb):
    with open(os.path.join(out_dir, 'points3D.txt'), 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write(f"# Number of points: {len(xyz)}\n")
        for i, (pt, col) in enumerate(zip(xyz, rgb)):
            f.write(f"{i+1} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                    f"{int(col[0])} {int(col[1])} {int(col[2])} 0.0\n")


# ─────────────────────────────────────────────────────────────────────────────
# Sim3 alignment (Umeyama) — align Pi3 poses to GT
# ─────────────────────────────────────────────────────────────────────────────

def align_sim3(src_pts, tgt_pts):
    """
    Find sim3 T such that T @ src ≈ tgt (least squares).
    src_pts, tgt_pts: (N, 3) camera centres.
    Returns (scale, R, t) where aligned = scale * R @ src + t
    """
    N = src_pts.shape[0]
    mu_s = src_pts.mean(0)
    mu_t = tgt_pts.mean(0)
    sc = src_pts - mu_s
    tc = tgt_pts - mu_t
    var_s = (sc ** 2).sum() / N
    H = sc.T @ tc / N
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    scale = (S * np.diag(D)).sum() / var_s
    t = mu_t - scale * R @ mu_s
    return scale, R, t


# ─────────────────────────────────────────────────────────────────────────────
# Pose comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_poses(pi3_poses_c2w, filenames, gt_images, report_path):
    """
    pi3_poses_c2w : (N, 4, 4) numpy, camera-to-world
    filenames      : list of N image filenames (sorted, as Pi3 loaded them)
    gt_images      : dict from read_images_bin/txt  {iid: {name, qvec, tvec}}
    """
    # Build GT lookup: filename -> c2w pose
    gt_by_name = {}
    for v in gt_images.values():
        R_w2c = quat_to_rotmat(v['qvec'])
        t_w2c = v['tvec']
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        pose_c2w = np.eye(4)
        pose_c2w[:3, :3] = R_c2w
        pose_c2w[:3,  3] = t_c2w
        gt_by_name[v['name']] = pose_c2w

    # Match Pi3 frames to GT by filename
    matched_pi3, matched_gt, matched_names = [], [], []
    for i, fname in enumerate(filenames):
        if fname in gt_by_name:
            matched_pi3.append(pi3_poses_c2w[i])
            matched_gt.append(gt_by_name[fname])
            matched_names.append(fname)

    if len(matched_pi3) < 3:
        print(f"  WARNING: only {len(matched_pi3)} matched frames, skipping pose comparison.")
        return

    matched_pi3 = np.stack(matched_pi3)   # (M, 4, 4)
    matched_gt  = np.stack(matched_gt)    # (M, 4, 4)
    M = len(matched_names)
    print(f"  Matched {M}/{len(filenames)} frames to GT")

    # Camera centres
    pi3_centres = matched_pi3[:, :3, 3]   # (M, 3)
    gt_centres  = matched_gt[:,  :3, 3]   # (M, 3)

    # Align Pi3 to GT via Sim3
    scale, R_align, t_align = align_sim3(pi3_centres, gt_centres)
    pi3_aligned_centres = (scale * (R_align @ pi3_centres.T).T + t_align)

    # Also rotate the orientation part of Pi3 poses
    pi3_aligned_R = np.stack([R_align @ matched_pi3[i, :3, :3] for i in range(M)])

    # Per-view errors
    rot_errs   = []
    trans_errs = []
    for i in range(M):
        R_rel = matched_gt[i, :3, :3].T @ pi3_aligned_R[i]
        rot_errs.append(rotation_angle_deg(R_rel))
        trans_errs.append(np.linalg.norm(pi3_aligned_centres[i] - gt_centres[i]))

    rot_errs   = np.array(rot_errs)
    trans_errs = np.array(trans_errs)
    ate_rmse   = float(np.sqrt((trans_errs ** 2).mean()))

    # Write report
    os.makedirs(os.path.dirname(report_path) or '.', exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Pi3 vs GT COLMAP Pose Deviation Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Matched frames : {M}\n")
        f.write(f"Sim3 scale     : {scale:.6f}\n\n")

        f.write(f"{'Image':<40} {'Rot err (deg)':>14} {'Trans err (m)':>14}\n")
        f.write("-" * 70 + "\n")
        for i, name in enumerate(matched_names):
            f.write(f"{name:<40} {rot_errs[i]:>14.4f} {trans_errs[i]:>14.4f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Aggregate Statistics\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Metric':<30} {'Rotation (deg)':>16} {'Translation (m)':>16}\n")
        f.write("-" * 64 + "\n")
        f.write(f"{'Mean':<30} {rot_errs.mean():>16.4f} {trans_errs.mean():>16.4f}\n")
        f.write(f"{'Median':<30} {np.median(rot_errs):>16.4f} {np.median(trans_errs):>16.4f}\n")
        f.write(f"{'Std':<30} {rot_errs.std():>16.4f} {trans_errs.std():>16.4f}\n")
        f.write(f"{'Min':<30} {rot_errs.min():>16.4f} {trans_errs.min():>16.4f}\n")
        f.write(f"{'Max':<30} {rot_errs.max():>16.4f} {trans_errs.max():>16.4f}\n")
        f.write(f"{'ATE RMSE':<30} {'':>16} {ate_rmse:>16.4f}\n")

    print(f"  Rot  error — mean: {rot_errs.mean():.3f}°  median: {np.median(rot_errs):.3f}°")
    print(f"  Trans error — mean: {trans_errs.mean():.4f}m  ATE RMSE: {ate_rmse:.4f}m")
    print(f"  Report saved: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', required=True,
                        help='Scene root, e.g. ~/3DGSDATASETS/bonsai')
    parser.add_argument('--out_dir',   required=True,
                        help='Output directory for this scene')
    parser.add_argument('--ckpt',      required=True,
                        help='Path to model.safetensors')
    parser.add_argument('--interval',  type=int, default=1)
    parser.add_argument('--max_pts',   type=int, default=200_000,
                        help='Max points in points3D.txt')
    parser.add_argument('--chunk',     type=int, default=8,
                        help='VO chunk size (default 8 for A100-40, use 16 for A100-80)')
    parser.add_argument('--overlap',   type=int, default=4,
                        help='VO overlap (default 4)')
    parser.add_argument('--probe_n',   type=int, default=12,
                        help='Frames used to estimate intrinsics (default 12)')
    parser.add_argument('--device',    default='cuda')
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    out_dir   = Path(args.out_dir)
    image_dir = scene_dir / 'images'
    sparse_dir= scene_dir / 'sparse'
    device    = torch.device(args.device)

    os.makedirs(out_dir, exist_ok=True)
    colmap_dir = out_dir / 'sparse' / '0'
    os.makedirs(colmap_dir, exist_ok=True)

    # ── 1. Load images ────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading images from {image_dir}")
    imgs_t, filenames, (H0, W0), (Hm, Wm) = \
        load_images(str(image_dir), interval=args.interval)
    N = len(filenames)
    imgs_batch = imgs_t.unsqueeze(0).to(device)

    # ── 2. Load model ─────────────────────────────────────────────────────────
    print(f"\n[2/5] Loading Pi3X from {args.ckpt}")
    model = Pi3X(use_multimodal=False).eval()
    from safetensors.torch import load_file
    model.load_state_dict(load_file(args.ckpt), strict=False)
    model.disable_multimodal()
    model = model.to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # ── 3. Estimate intrinsics from first chunk ───────────────────────────────
    print(f"\n[3/5] Estimating intrinsics")
    probe_n = min(args.probe_n, N)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        probe = model(imgs_batch[:, :probe_n], with_prior=False)
    lp   = probe['local_points'][0].float()
    conf = torch.sigmoid(probe['conf'][0, ..., 0]).float()
    fx_m, fy_m = estimate_focal(lp, conf, Hm, Wm)
    cx_m, cy_m = Wm / 2.0, Hm / 2.0
    sx, sy = W0 / Wm, H0 / Hm
    fx, fy, cx, cy = fx_m*sx, fy_m*sy, cx_m*sx, cy_m*sy
    print(f"  fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}  (at {W0}x{H0})")
    del probe, lp, conf
    torch.cuda.empty_cache()

    # ── 4. Run Pi3XVO ─────────────────────────────────────────────────────────
    print(f"\n[4/5] Running Pi3XVO on {N} frames")
    pipe = Pi3XVO(model)
    vo   = pipe(imgs_batch, chunk_size=args.chunk, overlap=args.overlap, conf_thre=0.1, dtype=dtype,
                intrinsics=(fx_m, fy_m, cx_m, cy_m))

    all_pts   = vo['points'][0].float().cpu()        # (N, H, W, 3)
    all_poses = vo['camera_poses'][0].float().cpu()  # (N, 4, 4)
    all_conf  = vo['conf'][0].float().cpu()          # (N, H, W)
    del vo; torch.cuda.empty_cache()

    # ── 5. Save outputs ───────────────────────────────────────────────────────
    print(f"\n[5/5] Saving outputs to {out_dir}")

    # 5a. PLY (same as example_vo.py)
    ply_path = out_dir / 'point_cloud.ply'
    conf_mask = all_conf > 0.1
    no_edge   = ~depth_edge(all_pts[..., 2], rtol=0.03)
    valid     = conf_mask & no_edge
    imgs_np   = imgs_t.permute(0, 2, 3, 1).numpy()
    write_ply(all_pts[valid].numpy(),
              (imgs_np[valid.numpy()] * 255).astype(np.uint8),
              str(ply_path))
    print(f"  PLY: {ply_path}  ({valid.sum().item():,} points)")

    # 5b. COLMAP cameras.txt
    write_cameras_txt(str(colmap_dir), W0, H0, fx, fy, cx, cy)

    # 5c. COLMAP images.txt
    write_images_txt(str(colmap_dir), filenames, all_poses.numpy())

    # 5d. COLMAP points3D.txt (subsampled)
    pts_all = all_pts[valid].numpy()
    rgb_all = (imgs_np[valid.numpy()] * 255).astype(np.uint8)
    if len(pts_all) > args.max_pts:
        idx     = np.random.choice(len(pts_all), args.max_pts, replace=False)
        pts_all = pts_all[idx]
        rgb_all = rgb_all[idx]
    write_points3d_txt(str(colmap_dir), pts_all, rgb_all)
    print(f"  COLMAP sparse: {colmap_dir}  ({len(pts_all):,} points in points3D.txt)")

    # 5e. Pose comparison vs GT COLMAP
    print(f"\n  Loading GT COLMAP from {sparse_dir}")
    gt_cams, gt_imgs = load_gt_colmap(str(sparse_dir))
    if gt_imgs is not None:
        report_path = str(out_dir / 'pose_report.txt')
        compare_poses(all_poses.numpy(), filenames, gt_imgs, report_path)
    else:
        print("  No GT COLMAP found — skipping pose comparison")

    print(f"\nDone. Outputs in {out_dir}/")
    print(f"  point_cloud.ply")
    print(f"  sparse/0/cameras.txt")
    print(f"  sparse/0/images.txt")
    print(f"  sparse/0/points3D.txt")
    print(f"  pose_report.txt")


if __name__ == '__main__':
    main()
