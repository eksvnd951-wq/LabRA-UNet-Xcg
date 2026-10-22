
import torch
import cv2
import glob
import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from albumentations import Compose, Resize, Normalize, ToTensorV2

# 기존 프로젝트 파일들에서 import
try:
    from models import AttentionUNet
except ImportError:
    print("❌ Error: 'models.py'가 같은 폴더에 있어야 합니다.")
    exit()

# ==========================================
# 1. Auto-Merge (후처리 핵심 로직)
# ==========================================
def apply_density_merge(pred_mask, merge_dist=30, min_cluster_size=500):
    """
    흩어진 BP(Class 1) 점들이 모여있으면 SP(Class 2)로 변환하는 함수
    """
    bp_mask = (pred_mask == 1).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_dist, merge_dist))
    connected_mask = cv2.morphologyEx(bp_mask, cv2.MORPH_CLOSE, kernel)
    nb, lbl, stats, _ = cv2.connectedComponentsWithStats(connected_mask)

    new_mask = pred_mask.copy()
    for i in range(1, nb):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_cluster_size:
            new_mask[lbl == i] = 2
            
    return new_mask

# ==========================================
# 2. 헬퍼 함수 (객체 카운팅 및 시각화)
# ==========================================
def get_objects(mask, score_map=None):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    nb, lbl, stats, _ = cv2.connectedComponentsWithStats(mask)
    objects = []
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] < 10: continue
        obj_mask = (lbl == i).astype(np.uint8)
        conf = np.mean(score_map[obj_mask == 1]) if score_map is not None else 1.0
        objects.append({'mask': obj_mask == 1, 'conf': conf})
    return objects

def draw_legend(img, counts, areas, colors, names):
    overlay = img.copy()
    h, w = img.shape[:2]
    box_w, box_h = 240, 100
    cv2.rectangle(overlay, (w - box_w - 10, 10), (w - 10, 10 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, "Detection Stats", (w - box_w + 5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y = 60
    for cls_idx in [1, 2]:
        name = names[cls_idx]; color = colors[cls_idx]
        cnt = counts.get(name, 0); px = areas.get(name, 0)
        cv2.rectangle(img, (w - box_w + 5, y - 10), (w - box_w + 25, y + 5), color, -1)
        text = f"{name}: {cnt} ea | {px} px"
        cv2.putText(img, text, (w - box_w + 35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
    return img

# ==========================================
# 3. 메인 실행 함수
# ==========================================
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best_model.pth')
    parser.add_argument('--source', type=str, default='test_images')
    # 기본 경로를 'runs/inference'로 설정하여 이 안에 exp1, exp2...가 생기도록 함
    parser.add_argument('--output', type=str, default='runs/inference') 
    parser.add_argument('--auto-merge', action='store_true')
    parser.add_argument('--merge-dist', type=int, default=30)
    parser.add_argument('--min-cluster', type=int, default=500)
    args = parser.parse_args()

    # 📂 [수정된 로직] 자동 exp 폴더 생성
    base_dir = args.output
    os.makedirs(base_dir, exist_ok=True)
    
    # 기존에 존재하는 expN 폴더들 확인
    existing_exps = [d for d in os.listdir(base_dir) if d.startswith('exp') and d[3:].isdigit()]
    if not existing_exps:
        new_exp_num = 1
    else:
        new_exp_num = max([int(d[3:]) for d in existing_exps]) + 1
    
    # 최종 결과 저장 경로 확정
    save_dir = os.path.join(base_dir, f'exp{new_exp_num}')
    mask_dir = os.path.join(save_dir, "masks")
    vis_dir = os.path.join(save_dir, "visualizations")
    
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 모델 로드
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionUNet(n_cls=3).to(dev)
    model.load_state_dict(torch.load(args.weights, map_location=dev))
    model.eval()

    # 이미지 리스트
    imgs = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        imgs.extend(glob.glob(os.path.join(args.source, ext)))

    tf = Compose([Resize(512, 512), Normalize((0.5,)*3, (0.5,)*3), ToTensorV2()])
    COLORS = {1: (0, 0, 255), 2: (255, 0, 0)}; NAMES = {1: 'BP', 2: 'SP'}
    csv_data = []

    print(f"🚀 Inference results will be saved to: {save_dir}")

    for p in tqdm(imgs, desc="Inferencing"):
        img_origin = cv2.imread(p)
        if img_origin is None: continue
        h0, w0 = img_origin.shape[:2]

        img_lab = cv2.cvtColor(img_origin, cv2.COLOR_BGR2LAB)
        inp = tf(image=img_lab)['image'].unsqueeze(0).to(dev)

        with torch.no_grad():
            logits = model(inp)
            probs = torch.softmax(logits, dim=1)
            pred_mask = torch.argmax(probs, dim=1).cpu().numpy()[0]
            prob_bp = probs[0, 1].cpu().numpy()
            prob_sp = probs[0, 2].cpu().numpy()

        mask_resized = cv2.resize(pred_mask.astype('uint8'), (w0, h0), interpolation=cv2.INTER_NEAREST)
        prob_bp = cv2.resize(prob_bp, (w0, h0)); prob_sp = cv2.resize(prob_sp, (w0, h0))

        if args.auto_merge:
            mask_resized = apply_density_merge(mask_resized, args.merge_dist, args.min_cluster)

        save_name = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(f"{mask_dir}/{save_name}.png", mask_resized)

        vis_img = img_origin.copy()
        current_counts = {}; current_areas = {'BP': np.sum(mask_resized == 1), 'SP': np.sum(mask_resized == 2)}

        for cls_idx in [1, 2]:
            score_map = prob_bp if cls_idx == 1 else prob_sp
            objects = get_objects((mask_resized == cls_idx), score_map)
            current_counts[NAMES[cls_idx]] = len(objects)
            for obj in objects:
                cnts, _ = cv2.findContours(obj['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, cnts, -1, COLORS[cls_idx], 2)

        vis_final = draw_legend(vis_img, current_counts, current_areas, COLORS, NAMES)
        cv2.imwrite(f"{vis_dir}/{save_name}.jpg", vis_final)

        csv_data.append({
            'Filename': save_name,
            'BP_Count': current_counts.get('BP', 0), 'BP_Area': current_areas.get('BP', 0),
            'SP_Count': current_counts.get('SP', 0), 'SP_Area': current_areas.get('SP', 0)
        })

    pd.DataFrame(csv_data).to_csv(f"{save_dir}/analysis_report.csv", index=False)
    print(f"✅ Completed! Folder: {save_dir}")

if __name__ == '__main__':
    run()
