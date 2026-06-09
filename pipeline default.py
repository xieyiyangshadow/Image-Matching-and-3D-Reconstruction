#!/usr/bin/env python3
"""
Integrated Image Matching & 3D Reconstruction Pipeline
Combines: 00_setup + 01_retrieval_shortlist + 02_mast3r_matching + 03_colmap_reconstruction

Usage:
    python pipeline.py
    python pipeline.py --image_root ../image-matching-challenge-2025/small_train
"""

# ================================================================
# §1 — Standard Library Imports
# ================================================================
print("="*60)
print("If encountering any package or library issue, please ensure all dependencies are installed.")
print("all needed package can be seen within requirements.txt")
print("MASt3R package is already included in this project. If not, please clone it from its github repository")
import os, sys, json, time, gc, shutil, subprocess, sqlite3, pickle
import urllib.request
import io, contextlib
from pathlib import Path
from collections import defaultdict

# ================================================================
# §2 — Path Setup (must precede MASt3R/dust3r imports)
# ================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent / 'mast3r'))
_DUST3R_PATH = Path(__file__).resolve().parent / 'mast3r' / 'dust3r'
if str(_DUST3R_PATH) not in sys.path:
    sys.path.insert(0, str(_DUST3R_PATH))

# ================================================================
# §3 — Scientific / ML Imports
# ================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ================================================================
# §4 — Environment Variables
# ================================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['GLOG_minloglevel'] = '2'
os.environ['HF_HOME'] = str(Path(__file__).resolve().parent / 'models')

# ================================================================
# §5 — Paths & Directories
# ================================================================
BASE = Path(__file__).resolve().parent
IMAGE_ROOT = BASE / 'image-matching-challenge-2025' / 'small_train'
MODEL_DIR = BASE / 'models'
CHECKPOINT_DIR = BASE / 'checkpoints'
OUTPUT_DIR = 'output'
RETRIEVAL_OUTPUT = BASE / OUTPUT_DIR / 'retrieval'
MATCH_OUTPUT = BASE / OUTPUT_DIR / 'mast3r_matching'
COLMAP_OUTPUT = BASE / OUTPUT_DIR / 'colmap_reconstruction'
SUBMISSION_PATH = BASE / OUTPUT_DIR / 'submission.csv'
OUTPUT_DIR = BASE / OUTPUT_DIR

for d in [MODEL_DIR, CHECKPOINT_DIR, RETRIEVAL_OUTPUT, MATCH_OUTPUT, COLMAP_OUTPUT]:
    d.mkdir(parents=True, exist_ok=True)

# ================================================================
# §6 — Model Weight Paths
# ================================================================
MAST3R_PATH = str(MODEL_DIR / 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')
ASMK_CKPT = str(CHECKPOINT_DIR / 'mast3r_asmk' / 'best_model.pth')
SPOC_CKPT = str(CHECKPOINT_DIR / 'mast3r_spoc' / 'best_model.pth')
SP_WEIGHTS_PATH = str(MODEL_DIR / 'superpoint' / 'superpoint_v1.pth')

# ================================================================
# §7 — Device
# ================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ================================================================
# §8 — Retrieval Parameters (01)
# ================================================================
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
PER_MODEL_TOPK = {'asmk': 25, 'spoc': 10, 'dino': 10, 'isc': 10}
ASMK_FPS_N = 10

DINO_INPUT = 224
DINO_DIM = 768
ASMK_IMG = 224
ASMK_DIM = 4096
MAST3R_ENC_DIM = 1024
SPOC_IMG = 224
SPOC_DIM = 256
ISC_IMG = 384
ISC_DIM = 256

# ================================================================
# §9 — MASt3R Matching Parameters (02)
# ================================================================
MATCH_SIZE = 384
SUBSAMPLE = 8
PIXEL_TOL = 5
MATCH_THRESHOLD = 1.001
MIN_MATCHES = 15
USE_AMP = device.type == 'cuda'
EMPTY_CACHE_EVERY_PAIR = True

ALIKED_MAX_KP = 4096
ALIKED_RESIZE = 1280
SP_MAX_KP = 4096
SP_RESIZE = 1600
SP_THRESHOLD = 0.0005

# ================================================================
# §9b — Model Toggle Switches (set to False to skip a model)
# ================================================================
# Retrieval models (Phase 2)
USE_DINO = True
USE_ASMK = True
USE_SPOC = True
USE_ISC  = True

# External keypoint detectors (Phase 3, Step 1)
USE_ALIKED     = True
USE_SUPERPOINT = True

# ================================================================
# §10 — COLMAP Reconstruction Parameters (03)
# ================================================================
CAMERA_MODEL = 'SIMPLE_PINHOLE'
SINGLE_CAMERA = False
MIN_MODEL_SIZE = 3
MAX_NUM_MODELS = 25

MAX_IMAGE_ID = 2147483647  # COLMAP pair_id encoding

COLMAP_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (camera_id INTEGER PRIMARY KEY AUTOINCREMENT, model INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, params BLOB, prior_focal_length INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS images (image_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, camera_id INTEGER NOT NULL, prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL, prior_tx REAL, prior_ty REAL, prior_tz REAL, CONSTRAINT image_id_camera_id FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
CREATE TABLE IF NOT EXISTS keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB, FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS matches (pair_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB);
CREATE TABLE IF NOT EXISTS two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB, config INTEGER NOT NULL, F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB);
CREATE INDEX IF NOT EXISTS idx_image_name ON images(name);
CREATE INDEX IF NOT EXISTS idx_keypoints_image_id ON keypoints(image_id);
CREATE INDEX IF NOT EXISTS idx_matches_pair_id ON matches(pair_id);
CREATE INDEX IF NOT EXISTS idx_two_view_geometries_pair_id ON two_view_geometries(pair_id);
"""

_TLK = {'weights_only': False, 'map_location': 'cpu'}

# ================================================================
# §11 — Utility Functions
# ================================================================

def load_json(p):
    with open(p) as f:
        return json.load(f)

def empty_cuda_cache():
    if device.type == 'cuda':
        torch.cuda.empty_cache()

def download_with_resume(url, dest_path, chunk_size=8*1024*1024):
    """Download with resume support."""
    dest_path = Path(dest_path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    total_size = 0
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=30) as resp:
            total_size = int(resp.headers.get('Content-Length', 0))
    except Exception:
        pass
    downloaded = dest_path.stat().st_size if dest_path.exists() else 0
    if total_size > 0 and downloaded >= total_size:
        return
    req = urllib.request.Request(url)
    if downloaded:
        req.add_header('Range', f'bytes={downloaded}-')
    mode = 'ab' if downloaded else 'wb'
    with urllib.request.urlopen(req, timeout=60) as resp:
        with open(dest_path, mode) as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)

def tensor_to_list(x):
    if isinstance(x, torch.Tensor):
        return [[float(v) for v in row] for row in x.detach().cpu().numpy().reshape(-1, 2)]
    return []

def array_to_str(arr):
    return ';'.join([f'{x:.09f}' for x in arr])

def nan_str(n):
    return ';'.join(['nan'] * n)

def make_transform(size):
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def cos_sim(f):
    fn = F.normalize(f, dim=-1)
    return torch.mm(fn, fn.t())

def add_topk_to_shortlist(shortlist_sets, sim_matrix, k):
    N = sim_matrix.shape[0]
    if N <= 1:
        return shortlist_sets
    s = sim_matrix.clone()
    s.fill_diagonal_(-float('inf'))
    _, idxs = torch.topk(s, k=min(k, N - 1), dim=-1)
    for i in range(N):
        shortlist_sets[i].update(idxs[i].tolist())
    return shortlist_sets

def colmap_pair_id(image_id1, image_id2):
    i1, i2 = sorted((int(image_id1), int(image_id2)))
    return i1 * MAX_IMAGE_ID + i2

# ================================================================
# §12 — Model Classes
# ================================================================

class DINOv2Global(nn.Module):
    """DINOv2 global descriptor via MAC pooling."""
    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(str(MODEL_DIR / 'dinov2-base'), local_files_only=True)
    def forward(self, x):
        outputs = self.encoder(x)
        return F.normalize(outputs.last_hidden_state[:, 1:].max(dim=1)[0], dim=1, p=2)


class MAST3RBackbone(nn.Module):
    """MAST3R encoder-only backbone (ViT-Large)."""
    def __init__(self, path):
        super().__init__()
        from mast3r.model import AsymmetricMASt3R
        m = AsymmetricMASt3R.from_pretrained(path)
        self.patch_embed, self.enc_blocks, self.enc_norm = m.patch_embed, m.enc_blocks, m.enc_norm
    def forward(self, img):
        x, pos = self.patch_embed(img)
        for blk in self.enc_blocks:
            x = blk(x, pos)
        return self.enc_norm(x)


class ASMKHead(nn.Module):
    def __init__(self, dim=MAST3R_ENC_DIM, cb=ASMK_DIM, tk=100):
        super().__init__()
        self.codebook = nn.Parameter(torch.empty(cb, dim))
        nn.init.xavier_uniform_(self.codebook)
        self.alpha = nn.Parameter(torch.tensor(3.0))
        self.top_k = tk
    def forward(self, f):
        fn, cn = F.normalize(f, dim=-1), F.normalize(self.codebook, dim=-1)
        s = torch.einsum('bnd,kd->bkn', fn, cn) * self.alpha
        k = min(self.top_k, s.shape[-1])
        tv, _ = torch.topk(s, k, dim=-1)
        return F.normalize(F.softmax(tv, dim=-1).sum(dim=-1), dim=-1)


class SPoCHead(nn.Module):
    def __init__(self, dim=MAST3R_ENC_DIM, hd=2048, od=256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(dim, hd), nn.BatchNorm1d(hd),
            nn.ReLU(inplace=True), nn.Linear(hd, od))
    def forward(self, f):
        return F.normalize(self.projection(f.sum(dim=1)), dim=-1)


class FineMASt3R(nn.Module):
    """Full MASt3R for image matching."""
    def __init__(self, path):
        super().__init__()
        from mast3r.model import AsymmetricMASt3R
        self.model = AsymmetricMASt3R.from_pretrained(path)
    def forward(self, v1, v2):
        return self.model(v1, v2)


class SuperPointNet(nn.Module):
    """Custom SuperPoint encoder + detector."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True))
        self.detector = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True), nn.Conv2d(256, 65, 1))
        self.descriptor = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True), nn.Conv2d(256, 256, 1))
    def forward(self, x):
        f = self.encoder(x)
        return self.detector(f), F.normalize(self.descriptor(f), dim=1)


class SuperPointWrapper(nn.Module):
    def __init__(self, weights_path, max_kp=4096, threshold=0.0005):
        super().__init__()
        self.net = SuperPointNet()
        self.net.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False), strict=False)
        self.max_kp = max_kp
        self.threshold = threshold
    @torch.no_grad()
    def forward(self, img):
        gray = img[:, :1, :, :]
        H, W = gray.shape[-2:]
        if max(H, W) > SP_RESIZE:
            scale = SP_RESIZE / max(H, W)
            gray = F.interpolate(gray, (int(H * scale), int(W * scale)), mode='bilinear', align_corners=False)
        else:
            scale = 1.0
        score, _ = self.net(gray)
        B, C, Hs, Ws = score.shape
        score = score[:, :64, :, :]
        score = score.reshape(B, 8, 8, Hs, Ws).permute(0, 1, 3, 2, 4).reshape(B, Hs * 8, Ws * 8)
        mask = score > self.threshold
        flat_score = score.reshape(-1)
        flat_mask = mask.reshape(-1)
        valid_scores = flat_score[flat_mask]
        if len(valid_scores) > self.max_kp:
            thresh = torch.topk(valid_scores, self.max_kp).values[-1]
            mask = mask & (score >= thresh)
        ys, xs = torch.where(mask[0])
        kpts = torch.stack([xs.float(), ys.float()], dim=-1)
        return {'keypoints': kpts.unsqueeze(0)}


class ImageDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img = transforms.ToTensor()(Image.open(p).convert('RGB'))
        return img, str(p)

# ================================================================
# §13 — Feature Extraction
# ================================================================

@torch.no_grad()
def extract_features_batch(model_or_tuple, model_name, paths, transform, batch_size,
                           isc_preproc=None, **kwargs):
    models = list(model_or_tuple) if isinstance(model_or_tuple, (tuple, list)) else [model_or_tuple]
    for m in models:
        m.eval()
    features = []
    for i in tqdm(range(0, len(paths), batch_size), desc=f'  {model_name}', leave=False):
        batch_paths = paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert('RGB')
            if isc_preproc:
                imgs.append(isc_preproc(img).unsqueeze(0))
            else:
                imgs.append(transform(img).unsqueeze(0))
        x = torch.cat(imgs, dim=0).to(device)
        if model_name == 'asmk':
            f = models[0](x)
            f = kwargs['asmk_head'](f)
        elif model_name == 'spoc':
            f = models[0](x)
            f = kwargs['spoc_head'](f)
        else:
            f = models[0](x)
        features.append(f.cpu())
    return torch.cat(features, dim=0)

# ================================================================
# §14 — MASt3R Matching Functions
# ================================================================

def run_mast3r_forward(model, t1, t2):
    empty_cuda_cache()
    try:
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
            return model({'img': t1, 'instance': []}, {'img': t2, 'instance': []})
    except torch.cuda.OutOfMemoryError:
        empty_cuda_cache()
        raise

def load_mast3r():
    model = FineMASt3R(MAST3R_PATH).to(device).eval()
    print('MASt3R loaded to GPU')
    return model

match_transform = make_transform(MATCH_SIZE)

@torch.no_grad()
def match_pair(model, p1, p2, transform_fn):
    img1 = Image.open(p1).convert('RGB')
    img2 = Image.open(p2).convert('RGB')
    H1, W1 = img1.height, img1.width
    H2, W2 = img2.height, img2.width
    t1 = transform_fn(img1).unsqueeze(0).to(device)
    t2 = transform_fn(img2).unsqueeze(0).to(device)
    result = {'matches': [], 'mkpts1': [], 'mkpts2': [], 'num_matches': 0}
    try:
        res1, res2 = run_mast3r_forward(model, t1, t2)
    except torch.cuda.OutOfMemoryError:
        return result

    def _unwrap(desc_raw):
        if desc_raw is None:
            return None
        t = desc_raw[-1] if isinstance(desc_raw, (list, tuple)) else desc_raw
        t = t[0]
        if t.dim() == 3:
            if t.shape[-1] < t.shape[0] and t.shape[-1] < t.shape[1]:
                t = t.permute(2, 0, 1)
            return t
        if t.dim() == 2:
            N, D = t.shape
            s = int(N ** 0.5)
            if s * s == N:
                return t.reshape(s, s, D).permute(2, 0, 1)
            return None
        return None

    d1 = _unwrap(res1.get('desc'))
    d2 = _unwrap(res2.get('desc'))
    if d1 is None or d2 is None:
        return result
    D1, Hd1, Wd1 = d1.shape
    D2, Hd2, Wd2 = d2.shape

    ys1 = torch.arange(0, Hd1, SUBSAMPLE, device=device)
    xs1 = torch.arange(0, Wd1, SUBSAMPLE, device=device)
    ys2 = torch.arange(0, Hd2, SUBSAMPLE, device=device)
    xs2 = torch.arange(0, Wd2, SUBSAMPLE, device=device)
    d1_s = d1[:, ys1[:, None], xs1[None, :]]
    d2_s = d2[:, ys2[:, None], xs2[None, :]]
    d1_flat = d1_s.reshape(D1, -1).t()
    d2_flat = d2_s.reshape(D2, -1).t()
    d1_flat = F.normalize(d1_flat, dim=-1)
    d2_flat = F.normalize(d2_flat, dim=-1)

    sim = torch.mm(d1_flat, d2_flat.t())
    nn12 = sim.argmax(dim=1)
    nn21 = sim.argmax(dim=0)
    mutual = torch.zeros(len(nn12), dtype=torch.bool, device=device)
    for i in range(len(nn12)):
        if nn21[nn12[i]] == i:
            mutual[i] = True
    if mutual.sum() == 0:
        return result

    idx1 = torch.where(mutual)[0]
    idx2 = nn12[mutual]
    h_scale1, w_scale1 = H1 / Hd1, W1 / Wd1
    h_scale2, w_scale2 = H2 / Hd2, W2 / Wd2
    y1 = (idx1 // (Wd1 // SUBSAMPLE)) * SUBSAMPLE * h_scale1
    x1 = (idx1 % (Wd1 // SUBSAMPLE)) * SUBSAMPLE * w_scale1
    y2 = (idx2 // (Wd2 // SUBSAMPLE)) * SUBSAMPLE * h_scale2
    x2 = (idx2 % (Wd2 // SUBSAMPLE)) * SUBSAMPLE * w_scale2
    mkpts1 = torch.stack([x1, y1], dim=-1).cpu().numpy()
    mkpts2 = torch.stack([x2, y2], dim=-1).cpu().numpy()
    matches = np.concatenate([mkpts1, mkpts2], axis=1)
    result['matches'] = [list(map(float, m)) for m in matches]
    result['mkpts1'] = [list(map(float, m)) for m in mkpts1]
    result['mkpts2'] = [list(map(float, m)) for m in mkpts2]
    result['num_matches'] = len(matches)
    return result


@torch.no_grad()
def match_sparse_using_keypoints(model, kp1, kp2, p1, p2, transform_fn, max_kp=4096):
    img1 = Image.open(p1).convert('RGB')
    img2 = Image.open(p2).convert('RGB')
    H1, W1 = img1.height, img1.width
    H2, W2 = img2.height, img2.width
    t1 = transform_fn(img1).unsqueeze(0).to(device)
    t2 = transform_fn(img2).unsqueeze(0).to(device)
    result = {'matches': [], 'mkpts1': [], 'mkpts2': [], 'matched_idx': [], 'scores': [], 'num_matches': 0}
    try:
        res1, res2 = run_mast3r_forward(model, t1, t2)
    except torch.cuda.OutOfMemoryError:
        return result

    def _unwrap(desc_raw):
        if desc_raw is None:
            return None
        t = desc_raw[-1] if isinstance(desc_raw, (list, tuple)) else desc_raw
        t = t[0]
        if t.dim() == 3:
            if t.shape[-1] < t.shape[0] and t.shape[-1] < t.shape[1]:
                t = t.permute(2, 0, 1)
            return t
        return None

    d1 = _unwrap(res1.get('desc'))
    d2 = _unwrap(res2.get('desc'))
    if d1 is None or d2 is None:
        return result
    _, Hd1, Wd1 = d1.shape
    _, Hd2, Wd2 = d2.shape

    kp1 = kp1.to(device).float()
    kp2 = kp2.to(device).float()
    sel1 = torch.arange(len(kp1), device=device)
    sel2 = torch.arange(len(kp2), device=device)
    if len(kp1) > max_kp:
        sel1 = torch.randperm(len(kp1), device=device)[:max_kp]
        kp1 = kp1[sel1]
    if len(kp2) > max_kp:
        sel2 = torch.randperm(len(kp2), device=device)[:max_kp]
        kp2 = kp2[sel2]
    if len(kp1) == 0 or len(kp2) == 0:
        return result

    kp1_desc = kp1.clone()
    kp1_desc[:, 0] = kp1[:, 0] * (Wd1 / W1)
    kp1_desc[:, 1] = kp1[:, 1] * (Hd1 / H1)
    kp2_desc = kp2.clone()
    kp2_desc[:, 0] = kp2[:, 0] * (Wd2 / W2)
    kp2_desc[:, 1] = kp2[:, 1] * (Hd2 / H2)
    kp1_norm = kp1_desc.clone()
    kp1_norm[:, 0] = kp1_norm[:, 0] / (Wd1 - 1) * 2 - 1
    kp1_norm[:, 1] = kp1_norm[:, 1] / (Hd1 - 1) * 2 - 1
    kp2_norm = kp2_desc.clone()
    kp2_norm[:, 0] = kp2_norm[:, 0] / (Wd2 - 1) * 2 - 1
    kp2_norm[:, 1] = kp2_norm[:, 1] / (Hd2 - 1) * 2 - 1

    desc1 = F.grid_sample(d1.unsqueeze(0), kp1_norm.unsqueeze(0).unsqueeze(0), mode='bilinear', align_corners=True)
    desc2 = F.grid_sample(d2.unsqueeze(0), kp2_norm.unsqueeze(0).unsqueeze(0), mode='bilinear', align_corners=True)
    desc1 = F.normalize(desc1[0, :, 0, :].t(), dim=-1)
    desc2 = F.normalize(desc2[0, :, 0, :].t(), dim=-1)

    sim = torch.mm(desc1, desc2.t())
    nn12 = sim.argmax(dim=1)
    nn21 = sim.argmax(dim=0)
    mutual = torch.zeros(len(nn12), dtype=torch.bool, device=device)
    for i in range(len(nn12)):
        if nn21[nn12[i]] == i:
            mutual[i] = True
    if mutual.sum() == 0:
        return result

    idx1 = torch.where(mutual)[0]
    idx2 = nn12[mutual]
    scores = sim[idx1, idx2]
    valid = scores > MATCH_THRESHOLD
    idx1 = idx1[valid]
    idx2 = idx2[valid]
    scores = scores[valid]
    if len(idx1) == 0:
        return result

    mkpts1 = kp1[idx1]
    mkpts2 = kp2[idx2]
    matched_idx = torch.stack([sel1[idx1], sel2[idx2]], dim=-1)
    matches = torch.cat([mkpts1, mkpts2], dim=-1).cpu().numpy()
    result['matches'] = [list(map(float, m)) for m in matches]
    result['mkpts1'] = [list(map(float, m)) for m in mkpts1.cpu().numpy()]
    result['mkpts2'] = [list(map(float, m)) for m in mkpts2.cpu().numpy()]
    result['matched_idx'] = [list(map(int, m)) for m in matched_idx.cpu().numpy()]
    result['scores'] = [float(s) for s in scores.detach().cpu().numpy()]
    result['num_matches'] = len(matches)
    return result


def collate_to_max_size(batch):
    imgs, paths = zip(*batch)
    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)
    orig_sizes = [(img.shape[1], img.shape[2]) for img in imgs]
    padded = []
    for img in imgs:
        ph, pw = max_h - img.shape[1], max_w - img.shape[2]
        padded.append(F.pad(img, (0, pw, 0, ph)) if ph > 0 or pw > 0 else img)
    return torch.stack(padded), list(paths), orig_sizes


@torch.no_grad()
def extract_kps_batch(img_batch, detector, max_dim, orig_sizes):
    kps_list = [torch.zeros((0, 2)) for _ in range(img_batch.shape[0])]
    try:
        H, W = img_batch.shape[-2:]
        if max(H, W) > max_dim:
            scale = max_dim / max(H, W)
            img_r = F.interpolate(img_batch, (int(H * scale), int(W * scale)), mode='bilinear', align_corners=False)
        else:
            img_r, scale = img_batch, 1.0
        out = detector(img_r)
        if isinstance(out, (list, tuple)):
            out = out[0]
        kps = out.keypoints if hasattr(out, 'keypoints') else out['keypoints'] if isinstance(out, dict) else out
        if isinstance(kps, torch.Tensor):
            if kps.dim() == 2:
                kps = kps.unsqueeze(0)
            for i in range(min(len(kps_list), kps.shape[0])):
                ki = kps[i] / scale
                oh, ow = orig_sizes[i]
                mask = (ki[:, 0] < ow) & (ki[:, 1] < oh)
                kps_list[i] = ki[mask].cpu()
    except Exception as e:
        print(f'    {type(detector).__name__} batch failed: {type(e).__name__}: {e}')
    return kps_list

# ================================================================
# §15 — COLMAP Database Builder
# ================================================================

def build_colmap_db_from_matches(dataset_name, ds_data, db_path, camera_model='SIMPLE_RADIAL', spatial_tol=1):
    matches_data = load_json(ds_data['match_file'])
    path_mapping = load_json(ds_data['path_mapping_file'])
    keypoint_data = {}
    if ds_data.get('keypoints_file'):
        keypoint_data = load_json(ds_data['keypoints_file'])

    all_images = set()
    for pair_key, match_info in matches_data.items():
        n1, n2 = pair_key.split('::')
        all_images.add(match_info.get('image1', n1))
        all_images.add(match_info.get('image2', n2))
    all_images = sorted(all_images)

    print(f'  Dataset: {dataset_name}, Images in matches: {len(all_images)}, Match pairs: {len(matches_data)}')

    img_sizes = {}
    for name in all_images:
        img_path = path_mapping.get(name)
        if img_path and Path(img_path).exists():
            with Image.open(img_path) as im:
                img_sizes[name] = im.size
        else:
            img_sizes[name] = (1024, 768)

    img_keypoints = defaultdict(dict)
    pair_key_matches = defaultdict(list)

    def dense_key(img_name, x, y):
        W = img_sizes[img_name][0]
        gx = int(round(x)) // spatial_tol
        gy = int(round(y)) // spatial_tol
        gw = (W + spatial_tol - 1) // spatial_tol
        key = ('dense', gy * gw + gx)
        coord = (gx * spatial_tol + spatial_tol / 2.0, gy * spatial_tol + spatial_tol / 2.0)
        img_keypoints[img_name].setdefault(key, coord)
        return key

    def sparse_key(img_name, detector, idx, fallback_xy=None):
        key = (detector, int(idx))
        coords = keypoint_data.get(detector, {}).get(img_name, [])
        if 0 <= int(idx) < len(coords):
            coord = tuple(float(v) for v in coords[int(idx)][:2])
        elif fallback_xy is not None:
            coord = tuple(float(v) for v in fallback_xy[:2])
        else:
            return None
        img_keypoints[img_name].setdefault(key, coord)
        return key

    def add_xy_matches(pair_key, n1, n2, xy_matches):
        for m in xy_matches:
            if len(m) < 4:
                continue
            x1, y1, x2, y2 = m[:4]
            k1 = dense_key(n1, x1, y1)
            k2 = dense_key(n2, x2, y2)
            pair_key_matches[pair_key].append((k1, k2))

    sparse_corr = 0
    dense_corr = 0
    for pair_key, match_info in tqdm(matches_data.items(), desc='  Pairs', leave=False):
        n1, n2 = pair_key.split('::')
        n1 = match_info.get('image1', n1)
        n2 = match_info.get('image2', n2)
        if 'dense' in match_info or 'sparse' in match_info:
            dense_matches = match_info.get('dense', {}).get('matches', [])
            add_xy_matches(pair_key, n1, n2, dense_matches)
            dense_corr += len(dense_matches)
            for detector, sparse_info in match_info.get('sparse', {}).items():
                matched_idx = sparse_info.get('matched_idx', [])
                mkpts1 = sparse_info.get('mkpts1', [])
                mkpts2 = sparse_info.get('mkpts2', [])
                for row_id, ij in enumerate(matched_idx):
                    if len(ij) < 2:
                        continue
                    k1 = sparse_key(n1, detector, ij[0], mkpts1[row_id] if row_id < len(mkpts1) else None)
                    k2 = sparse_key(n2, detector, ij[1], mkpts2[row_id] if row_id < len(mkpts2) else None)
                    if k1 is not None and k2 is not None:
                        pair_key_matches[pair_key].append((k1, k2))
                        sparse_corr += 1
        else:
            xy_matches = match_info.get('matches', [])
            add_xy_matches(pair_key, n1, n2, xy_matches)
            dense_corr += len(xy_matches)

    img_kp_to_idx = defaultdict(dict)
    for img_name, kp_dict in img_keypoints.items():
        for idx, key in enumerate(kp_dict.keys()):
            img_kp_to_idx[img_name][key] = idx

    total_kps = sum(len(v) for v in img_keypoints.values())
    print(f'  Keypoints: {total_kps} ({sparse_corr} sparse correspondences, {dense_corr} dense correspondences)')

    if Path(db_path).exists():
        os.remove(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(COLMAP_DB_SCHEMA)

    img_to_im_id = {}
    for img_name in all_images:
        W, H = img_sizes[img_name]
        focal = 1.2 * max(W, H)
        cx, cy = W / 2.0, H / 2.0
        if camera_model == 'SIMPLE_RADIAL':
            model_id, params = 2, np.array([focal, cx, cy, 0.0], dtype=np.float64)
        elif camera_model == 'SIMPLE_PINHOLE':
            model_id, params = 0, np.array([focal, cx, cy], dtype=np.float64)
        else:
            model_id, params = 1, np.array([focal, focal, cx, cy], dtype=np.float64)
        cur = conn.execute(
            'INSERT INTO cameras (model, width, height, params, prior_focal_length) VALUES (?,?,?,?,0)',
            (model_id, W, H, params.tobytes()))
        cur = conn.execute('INSERT INTO images (name, camera_id) VALUES (?,?)', (img_name, cur.lastrowid))
        img_to_im_id[img_name] = cur.lastrowid

    for img_name in tqdm(all_images, desc='  Keypoints', leave=False):
        kp_list = list(img_keypoints.get(img_name, {}).values())
        if not kp_list:
            continue
        kp_arr = np.array(kp_list, dtype=np.float32)
        conn.execute('INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?,?,?,?)',
                     (img_to_im_id[img_name], kp_arr.shape[0], kp_arr.shape[1], kp_arr.tobytes()))

    match_count = 0
    for pair_key, key_matches in tqdm(pair_key_matches.items(), desc='  Matches', leave=False):
        n1, n2 = pair_key.split('::')
        if n1 not in img_to_im_id or n2 not in img_to_im_id:
            continue
        colmap_matches = []
        seen = set()
        for k1, k2 in key_matches:
            idx1 = img_kp_to_idx[n1].get(k1)
            idx2 = img_kp_to_idx[n2].get(k2)
            if idx1 is None or idx2 is None:
                continue
            item = (idx1, idx2)
            if item not in seen:
                seen.add(item)
                colmap_matches.append(item)
        if not colmap_matches:
            continue
        imid1, imid2 = img_to_im_id[n1], img_to_im_id[n2]
        if imid1 > imid2:
            imid1, imid2 = imid2, imid1
            colmap_matches = [(b, a) for a, b in colmap_matches]
        m_arr = np.array(colmap_matches, dtype=np.uint32)
        pair_id = colmap_pair_id(imid1, imid2)
        conn.execute('INSERT OR REPLACE INTO matches (pair_id, rows, cols, data) VALUES (?,?,?,?)',
                     (pair_id, m_arr.shape[0], m_arr.shape[1], m_arr.tobytes()))
        conn.execute('INSERT OR REPLACE INTO two_view_geometries (pair_id, rows, cols, data, config) VALUES (?,?,?,?,1)',
                     (pair_id, m_arr.shape[0], m_arr.shape[1], m_arr.tobytes()))
        match_count += 1

    conn.commit()
    conn.close()
    print(f'  DB ready: {match_count} match pairs, {total_kps} keypoints')
    return all_images, img_to_im_id, img_sizes


# ================================================================
# §16 — MAIN PIPELINE
# ================================================================

def main():
    t_total = time.time()

    # ----------------------------------------------------------
    # Phase 1 — Verify models / checkpoints exist
    # ----------------------------------------------------------
    print('\n' + '=' * 60)
    print('PHASE 1: Model Verification')
    print('=' * 60)
    mast3r_file = Path(MAST3R_PATH)
    if mast3r_file.exists() and mast3r_file.stat().st_size > 100 * 1024**2:
        print(f'MAST3R checkpoint OK: {mast3r_file}')
    else:
        print(f'MAST3R checkpoint MISSING or too small: {mast3r_file}')

    for name, path in [('MAST3R-ASMK', ASMK_CKPT), ('MAST3R-SPoC', SPOC_CKPT)]:
        if Path(path).exists():
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            print(f'{name}: OK (epoch {ckpt.get("epoch", "?")}, loss {ckpt.get("loss", "?"):.4f})')
        else:
            print(f'{name}: MISSING ({path})')

    # ----------------------------------------------------------
    # Phase 2 — Retrieval: per-dataset features + shortlists
    # ----------------------------------------------------------
    print('\n' + '=' * 60)
    print('PHASE 2: Image Retrieval & Shortlist Generation')
    print('=' * 60)

    # Discover datasets
    datasets_retrieval = {}
    for d in sorted(IMAGE_ROOT.iterdir()):
        if d.is_dir() and d.name != 'outliers':
            images = sorted(p for p in d.rglob('*') if p.suffix.lower() in IMAGE_EXTS)
            if images:
                datasets_retrieval[d.name] = images

    print(f'Datasets: {len(datasets_retrieval)}')
    for ds_name, ds_images in datasets_retrieval.items():
        print(f'  {ds_name}: {len(ds_images)} images')

    # Save path mappings
    for ds_name, ds_images in datasets_retrieval.items():
        ds_out = RETRIEVAL_OUTPUT / ds_name
        ds_out.mkdir(parents=True, exist_ok=True)
        path_mapping = {p.name: str(p) for p in ds_images}
        with open(ds_out / 'image_paths.json', 'w') as f:
            json.dump(path_mapping, f, indent=2)

    # Load retrieval models
    print('\nLoading retrieval models...')

    dino_model = None
    if USE_DINO:
        dino_model = DINOv2Global().to(device).eval()
        print(f'DINOv2 loaded. Dim: {DINO_DIM}')
    else:
        print('DINOv2: SKIPPED')

    asmk_backbone = asmk_head = None
    if USE_ASMK:
        asmk_backbone = MAST3RBackbone(MAST3R_PATH).to(device).eval()
        asmk_head = ASMKHead().to(device)
        ckpt = torch.load(ASMK_CKPT, **_TLK)
        bb_state = {k.replace('mast3r.', ''): v for k, v in ckpt['backbone_state'].items()
                    if k.startswith('mast3r.patch_embed') or k.startswith('mast3r.enc_blocks') or k.startswith('mast3r.enc_norm')}
        asmk_backbone.load_state_dict(bb_state, strict=True)
        asmk_head.load_state_dict(ckpt['asmk_state'])
        asmk_head.eval()
        print(f'MAST3R-ASMK loaded (epoch {ckpt["epoch"]}). Dim: {ASMK_DIM}')
    else:
        print('MAST3R-ASMK: SKIPPED')

    spoc_backbone = spoc_head = None
    if USE_SPOC:
        spoc_backbone = MAST3RBackbone(MAST3R_PATH).to(device).eval()
        spoc_head = SPoCHead().to(device)
        ckpt = torch.load(SPOC_CKPT, **_TLK)
        bb_state = {k.replace('mast3r.', ''): v for k, v in ckpt['backbone_state'].items()
                    if k.startswith('mast3r.patch_embed') or k.startswith('mast3r.enc_blocks') or k.startswith('mast3r.enc_norm')}
        spoc_backbone.load_state_dict(bb_state, strict=True)
        spoc_head.load_state_dict(ckpt['spoc_head_state'])
        spoc_head.eval()
        print(f'MAST3R-SPoC loaded (epoch {ckpt["epoch"]}). Dim: {SPOC_DIM}')
    else:
        print('MAST3R-SPoC: SKIPPED')

    isc_model = isc_preprocessor = None
    if USE_ISC:
        from isc_feature_extractor import create_model
        isc_model, isc_preprocessor = create_model(weight_name='isc_ft_v107', device=device)
        isc_model.eval()
        print(f'ISC loaded. Dim: {ISC_DIM}')
    else:
        print('ISC: SKIPPED')

    # Extract features and generate shortlists
    print('\nExtracting features per dataset...')
    t0 = time.time()
    all_features = {}

    for ds_name, ds_images in tqdm(datasets_retrieval.items(), desc='Datasets'):
        image_paths = [str(p) for p in ds_images]
        ds_out = RETRIEVAL_OUTPUT / ds_name
        ds_out.mkdir(parents=True, exist_ok=True)

        feat_dict = {}
        if USE_DINO:
            feat_dict['dino'] = extract_features_batch(dino_model, 'dino', image_paths, make_transform(DINO_INPUT), 32)
        if USE_ASMK:
            feat_dict['asmk'] = extract_features_batch((asmk_backbone,), 'asmk', image_paths, make_transform(ASMK_IMG), 8, asmk_head=asmk_head)
        if USE_SPOC:
            feat_dict['spoc'] = extract_features_batch((spoc_backbone,), 'spoc', image_paths, make_transform(SPOC_IMG), 8, spoc_head=spoc_head)
        if USE_ISC:
            feat_dict['isc'] = extract_features_batch(isc_model, 'isc', image_paths, None, 32, isc_preproc=isc_preprocessor)

        torch.save(feat_dict, ds_out / 'features.pt')
        all_features[ds_name] = feat_dict

    print(f'Feature extraction done in {time.time() - t0:.0f}s')

    # Generate shortlists
    print('\nGenerating per-dataset shortlists...')
    for ds_name, ds_images in tqdm(datasets_retrieval.items(), desc='Shortlist'):
        ds_out = RETRIEVAL_OUTPUT / ds_name
        feat = all_features[ds_name]
        image_names = [p.name for p in ds_images]
        N_ds = len(image_names)

        sims = {}
        active_keys = []
        for k in ['dino', 'asmk', 'spoc', 'isc']:
            if k in feat:
                sims[k] = cos_sim(feat[k])
                active_keys.append(k)
        sims_np = {k: v.cpu().numpy() for k, v in sims.items()}
        np.savez(ds_out / 'per_model_similarity.npz', **sims_np)

        shortlist_sets = {i: set() for i in range(N_ds)}
        for key in active_keys:
            add_topk_to_shortlist(shortlist_sets, sims[key], PER_MODEL_TOPK[key])

        shortlist = {}
        for i in range(N_ds):
            name = image_names[i]
            cands = []
            for j in shortlist_sets[i]:
                if i == j:
                    continue
                nbr = image_names[j]
                score = max(sims[k][i, j].item() for k in sims)
                cands.append({'idx': j, 'name': nbr, 'score': float(score)})
            cands.sort(key=lambda x: x['score'], reverse=True)
            shortlist[name] = cands

        out = {k: [{'idx': c['idx'], 'name': c['name'], 'score': c['score']} for c in v]
               for k, v in shortlist.items()}
        with open(ds_out / 'shortlist.json', 'w') as f:
            json.dump(out, f, indent=2)

    print('Retrieval complete.')

    # Free retrieval models from GPU
    for m in [dino_model, asmk_backbone, asmk_head, spoc_backbone, spoc_head, isc_model]:
        if m is not None: del m
    gc.collect()
    torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # Phase 3 — MASt3R Hybrid Matching
    # ----------------------------------------------------------
    print('\n' + '=' * 60)
    print('PHASE 3: MASt3R Hybrid Matching')
    print('=' * 60)

    # Discover datasets from retrieval output
    datasets_match = {}
    for ds_dir in sorted(RETRIEVAL_OUTPUT.iterdir()):
        if ds_dir.is_dir():
            sl_path = ds_dir / 'shortlist.json'
            pm_path = ds_dir / 'image_paths.json'
            if sl_path.exists() and pm_path.exists():
                datasets_match[ds_dir.name] = {
                    'shortlist': load_json(sl_path),
                    'path_mapping': load_json(pm_path),
                }

    print(f'Datasets with shortlists: {len(datasets_match)}')

    # Load ALIKED detector
    aliked_detector = None
    if USE_ALIKED:
        try:
            from kornia.feature import ALIKED
            aliked_detector = ALIKED(max_num_keypoints=ALIKED_MAX_KP, detection_threshold=0.01)
            aliked_detector = aliked_detector.to(device).eval()
            print(f'ALIKED loaded via kornia (max_kp={ALIKED_MAX_KP})')
        except (ImportError, TypeError):
            pass
    else:
        print('ALIKED: SKIPPED')

    # Load SuperPoint detector
    superpoint_detector = None
    if USE_SUPERPOINT:
        try:
            from kornia.feature import SuperPoint
            superpoint_detector = SuperPoint(pretrained=True).to(device).eval()
            print('SuperPoint loaded via kornia')
        except ImportError:
            pass
        if superpoint_detector is None and Path(SP_WEIGHTS_PATH).exists():
            superpoint_detector = SuperPointWrapper(SP_WEIGHTS_PATH, max_kp=SP_MAX_KP, threshold=SP_THRESHOLD).to(device).eval()
            print(f'SuperPoint loaded via custom impl (max_kp={SP_MAX_KP})')
    else:
        print('SuperPoint: SKIPPED')

    # Step 1: Extract keypoints on GPU (MASt3R not loaded yet)
    print('\nExtracting per-image keypoints (MASt3R not loaded yet)...')
    aliked_keypoints = {}
    sp_keypoints = {}
    BATCH_SIZE = 8

    for ds_name, ds_data in tqdm(datasets_match.items(), desc='Keypoints'):
        path_mapping = ds_data['path_mapping']
        all_images = sorted(path_mapping.keys())
        all_paths = [str(Path(path_mapping[n])) for n in all_images if path_mapping.get(n) and Path(path_mapping[n]).exists()]

        if not all_paths:
            aliked_keypoints[ds_name] = {}
            sp_keypoints[ds_name] = {}
            continue

        dataset = ImageDataset(all_paths)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_to_max_size, num_workers=0)
        aliked_kps = {}
        sp_kps = {}

        for img_batch, path_batch, orig_sizes in tqdm(loader, desc=f'  {ds_name}', leave=False):
            img_batch = img_batch.to(device)
            if aliked_detector is not None:
                for p, kps in zip(path_batch, extract_kps_batch(img_batch, aliked_detector, ALIKED_RESIZE, orig_sizes)):
                    if kps.numel() > 0:
                        aliked_kps[Path(p).name] = kps
            if superpoint_detector is not None:
                for p, kps in zip(path_batch, extract_kps_batch(img_batch, superpoint_detector, SP_RESIZE, orig_sizes)):
                    if kps.numel() > 0:
                        sp_kps[Path(p).name] = kps
            del img_batch
            torch.cuda.empty_cache()

        aliked_keypoints[ds_name] = aliked_kps
        sp_keypoints[ds_name] = sp_kps

    print('Keypoint extraction complete.')

    # Step 2: Release detectors, load MASt3R, run matching
    print('\nHybrid MASt3R matching...')
    if aliked_detector is not None:
        del aliked_detector
    if superpoint_detector is not None:
        del superpoint_detector
    aliked_detector = None
    superpoint_detector = None
    gc.collect()
    torch.cuda.empty_cache()

    mast3r = load_mast3r()

    for ds_name, ds_data in tqdm(datasets_match.items(), desc='Datasets'):
        ds_out = MATCH_OUTPUT / ds_name
        ds_out.mkdir(parents=True, exist_ok=True)

        shortlist = ds_data['shortlist']
        path_mapping = ds_data['path_mapping']
        pairs = set()
        for img, cands in shortlist.items():
            for c in cands:
                pairs.add(tuple(sorted([img, c['name']])))
        pairs = list(pairs)

        keypoint_payload = {
            'aliked': {name: tensor_to_list(kps) for name, kps in aliked_keypoints.get(ds_name, {}).items()},
            'superpoint': {name: tensor_to_list(kps) for name, kps in sp_keypoints.get(ds_name, {}).items()},
        }
        with open(ds_out / 'keypoints.json', 'w') as f:
            json.dump(keypoint_payload, f)

        t0 = time.time()
        hybrid_matches = {}
        dense_matches_compat = {}

        for n1, n2 in tqdm(pairs, desc=f'  {ds_name}', leave=False):
            p1 = path_mapping.get(n1)
            p2 = path_mapping.get(n2)
            if not p1 or not p2:
                continue

            dense_result = match_pair(mast3r, p1, p2, match_transform)
            sparse_results = {}
            combined_lists = []
            if dense_result['num_matches'] > 0:
                combined_lists.append(dense_result['matches'])

            ak = aliked_keypoints.get(ds_name, {})
            if ak and n1 in ak and n2 in ak:
                aliked_result = match_sparse_using_keypoints(mast3r, ak[n1], ak[n2], p1, p2, match_transform, max_kp=ALIKED_MAX_KP)
                sparse_results['aliked'] = aliked_result
                if aliked_result['num_matches'] > 0:
                    combined_lists.append(aliked_result['matches'])

            sk = sp_keypoints.get(ds_name, {})
            if sk and n1 in sk and n2 in sk:
                sp_result = match_sparse_using_keypoints(mast3r, sk[n1], sk[n2], p1, p2, match_transform, max_kp=SP_MAX_KP)
                sparse_results['superpoint'] = sp_result
                if sp_result['num_matches'] > 0:
                    combined_lists.append(sp_result['matches'])

            combined = []
            seen = set()
            for match_list in combined_lists:
                for m in match_list:
                    key = (round(m[0]), round(m[1]), round(m[2]), round(m[3]))
                    if key not in seen:
                        seen.add(key)
                        combined.append(m)

            if len(combined) >= MIN_MATCHES:
                pair_key = f'{n1}::{n2}'
                hybrid_matches[pair_key] = {
                    'image1': n1, 'image2': n2,
                    'dense': dense_result, 'sparse': sparse_results,
                    'matches': combined, 'num_matches': len(combined),
                }
                dense_matches_compat[pair_key] = {'matches': combined, 'num_matches': len(combined)}

            if EMPTY_CACHE_EVERY_PAIR:
                empty_cuda_cache()

        with open(ds_out / 'hybrid_matches.json', 'w') as f:
            json.dump(hybrid_matches, f)
        with open(ds_out / 'dense_matches.json', 'w') as f:
            json.dump(dense_matches_compat, f)

    print('Matching complete.')

    # Free MASt3R
    del mast3r
    gc.collect()
    torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # Phase 4 — COLMAP Reconstruction & Submission
    # ----------------------------------------------------------
    print('\n' + '=' * 60)
    print('PHASE 4: COLMAP Reconstruction & Submission')
    print('=' * 60)

    import pycolmap

    datasets_recon = {}
    for ds_dir in sorted(MATCH_OUTPUT.iterdir()):
        if ds_dir.is_dir():
            hybrid_path = ds_dir / 'hybrid_matches.json'
            dense_path = ds_dir / 'dense_matches.json'
            keypoint_path = ds_dir / 'keypoints.json'
            match_path = hybrid_path if hybrid_path.exists() else dense_path
            pm_path = RETRIEVAL_OUTPUT / ds_dir.name / 'image_paths.json'
            if match_path.exists() and pm_path.exists():
                datasets_recon[ds_dir.name] = {
                    'match_file': match_path,
                    'keypoints_file': keypoint_path if keypoint_path.exists() else None,
                    'path_mapping_file': pm_path,
                    'match_format': 'hybrid' if match_path == hybrid_path else 'dense',
                }

    print(f'Datasets with matches: {len(datasets_recon)}')

    if len(datasets_recon) == 0:
        print('ERROR: No dataset match files found! Run Phase 3 first.')
        return

    # Check COLMAP
    USE_PYCOLMAP = True
    COLMAP_PATH_CLI = 'colmap'
    COLMAP_OK = False
    try:
        import pycolmap
        print(f'pycolmap: {pycolmap.__version__}')
        COLMAP_OK = True
    except ImportError:
        print('pycolmap not found')

    COLMAP_CLI_BIN = shutil.which(COLMAP_PATH_CLI)
    if COLMAP_CLI_BIN is None:
        for p in ['/usr/bin/colmap', '/usr/local/bin/colmap', '/kaggle/bin/colmap', '/opt/conda/bin/colmap']:
            if os.path.exists(p):
                COLMAP_CLI_BIN = p
                break
    USE_COLMAP_CLI = COLMAP_CLI_BIN is not None and os.path.exists(COLMAP_CLI_BIN)

    all_results = {}
    all_predictions = {}
    mapper_opts = pycolmap.IncrementalPipelineOptions()
    mapper_opts.min_model_size = MIN_MODEL_SIZE
    mapper_opts.max_num_models = MAX_NUM_MODELS

    for ds_name, ds_data in tqdm(datasets_recon.items(), desc='Reconstructing'):
        ds_out = COLMAP_OUTPUT / ds_name
        ds_out.mkdir(parents=True, exist_ok=True)
        db_path = ds_out / 'database.db'
        sparse_dir = ds_out / 'sparse'
        sparse_dir.mkdir(exist_ok=True)

        path_mapping = load_json(ds_data['path_mapping_file'])
        all_dataset_images = list(path_mapping.keys())

        matched_images, img_to_im_id, img_sizes = build_colmap_db_from_matches(
            ds_name, ds_data, db_path, camera_model=CAMERA_MODEL)

        img_flat_dir = ds_out / 'images'
        img_flat_dir.mkdir(exist_ok=True)
        for name in matched_images:
            src = str(IMAGE_ROOT / ds_name / name)
            if Path(src).exists():
                dst = img_flat_dir / name
                if not dst.exists():
                    try:
                        shutil.copy2(src, dst)
                    except Exception:
                        pass

        if USE_COLMAP_CLI and not USE_PYCOLMAP:
            subprocess.run([
                COLMAP_CLI_BIN, 'mapper',
                '--database_path', str(db_path),
                '--image_path', str(img_flat_dir),
                '--output_path', str(sparse_dir),
                '--Mapper.min_model_size', str(MIN_MODEL_SIZE),
                '--Mapper.max_num_models', str(MAX_NUM_MODELS),
                '--Mapper.init_min_num_inliers', '15',
                '--Mapper.abs_pose_min_num_inliers', '10',
                '--Mapper.min_num_matches', '5',
            ], capture_output=True, text=True)
            recs = {}
            for sub_dir in sorted(sparse_dir.glob('*')):
                if sub_dir.is_dir():
                    try:
                        rec = pycolmap.Reconstruction(str(sub_dir))
                        if rec.num_reg_images() > 0:
                            recs[len(recs)] = rec
                    except Exception:
                        pass
        else:
            try:
                recs = pycolmap.incremental_mapping(
                    database_path=str(db_path), image_path=str(img_flat_dir),
                    output_path=str(sparse_dir), options=mapper_opts)
            except Exception as e:
                print(f'  Mapper error: {e}')
                recs = {}

        predictions = []
        img_to_cluster = {}
        img_to_pose = {}

        if recs:
            for idx, r in recs.items():
                r.write(str(sparse_dir / str(idx)))
            for sub_id, rec in recs.items():
                for im_id, image in rec.images.items():
                    cam_from_world = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
                    R = cam_from_world.rotation.matrix()
                    t = cam_from_world.translation
                    img_to_cluster[image.name] = sub_id
                    img_to_pose[image.name] = (R, t)

            tr = len(img_to_cluster)
            tp = sum(r.num_points3D() for r in recs.values())
            sizes = [r.num_reg_images() for r in recs.values()]
            all_results[ds_name] = {'registered': tr, 'total_images': len(all_dataset_images),
                                    'points': tp, 'sub_models': len(recs), 'sizes': sorted(sizes, reverse=True)}
        else:
            all_results[ds_name] = {'registered': 0, 'total_images': len(all_dataset_images),
                                    'points': 0, 'sub_models': 0}

        for img_name in all_dataset_images:
            if img_name in img_to_cluster:
                R, t = img_to_pose[img_name]
                predictions.append({'image': img_name, 'scene': f'cluster{img_to_cluster[img_name]}',
                                    'R': R.flatten(), 't': t})
            else:
                predictions.append({'image': img_name, 'scene': 'outliers', 'R': None, 't': None})
        all_predictions[ds_name] = predictions

    # Summary
    successful = sum(1 for r in all_results.values() if r['registered'] > 2)
    total_reg = sum(r['registered'] for r in all_results.values())
    total_imgs = sum(r['total_images'] for r in all_results.values())
    total_pts = sum(r['points'] for r in all_results.values())
    print(f'\nReconstruction Complete: {successful} datasets, {total_reg}/{total_imgs} registered, {total_pts:,} points')

    # Generate submission.csv
    with open(SUBMISSION_PATH, 'w') as f:
        f.write('dataset,scene,image,rotation_matrix,translation_vector\n')
        for ds_name, preds in all_predictions.items():
            for pred in preds:
                rotation = nan_str(9) if pred['R'] is None else array_to_str(pred['R'])
                translation = nan_str(3) if pred['t'] is None else array_to_str(pred['t'])
                f.write(f'{ds_name},{pred["scene"]},{pred["image"]},{rotation},{translation}\n')

    print(f'Submission saved to: {SUBMISSION_PATH}')
    print(f'Total time: {time.time() - t_total:.0f}s')


if __name__ == '__main__':
    main()
