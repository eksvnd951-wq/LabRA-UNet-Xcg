
import argparse, os, torch, glob, numpy as np, pandas as pd, time, datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from albumentations import Compose, Resize, Normalize, ToTensorV2, HorizontalFlip, VerticalFlip, RandomRotate90, GridDistortion, ElasticTransform
from models import AttentionUNet; from dataset import LeafDataset
from metrics import compute_metrics; from losses import ComboLoss
from analysis_utils import plot_timeline

def train():
    p = argparse.ArgumentParser(); p.add_argument('--data', default=None)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--patience', type=int, default=15)
    args = p.parse_args()

    st = time.time(); st_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st))
    exp_dir = './runs/train/exp'; os.makedirs(os.path.dirname(exp_dir), exist_ok=True)
    sd = f"{exp_dir}{len(glob.glob(f'{exp_dir}*'))+1}"; os.makedirs(sd, exist_ok=True)

    writer = SummaryWriter(f"{sd}/events")
    print(f"🚀 Start: {st_str} | Save: {sd}")

    dpath = args.data
    if dpath is None or not os.path.exists(dpath):
        candidates = glob.glob("**/train/images", recursive=True)
        if candidates: dpath = candidates[0].split('/train')[0]; print(f"📍 Auto-located: {dpath}")
        else: return print("❌ Error: Dataset not found.")

    # [증강 수정] 큰 객체(SP)의 형태 유지를 위해 GridDistortion 추가, Elastic 유지
    tf = Compose([
        Resize(512,512),
        HorizontalFlip(p=0.5), VerticalFlip(p=0.5), RandomRotate90(p=0.5),
        GridDistortion(num_steps=5, distort_limit=0.3, p=0.3), # 큰 객체 변형 대응
        ElasticTransform(p=0.3, alpha=120, sigma=120*0.05),
        Normalize((0.5,)*3,(0.5,)*3), ToTensorV2()
    ])

    ds = LeafDataset(f"{dpath}/train/images", f"{dpath}/train/masks", 512, tf, is_train=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionUNet(n_cls=3).to(dev); crit = ComboLoss().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=0.0003)

    print(f"{'Epoch':^5} | {'Loss':^7} | {'mIoU':^6} | {'mF1':^6} | {'mAP':^6} | {'Fit.':^6} | {'Time':^9} | {'Pat.':^4}")
    h = {'loss':[], 'mIoU':[], 'mF1':[], 'mAP':[], 'mPre':[], 'mRec':[]}
    best_fitness = 0.0; patience_counter = 0; last_met = {}

    for e in range(args.epochs):
        et = time.time(); model.train(); ls=0; em={}
        pbar = tqdm(dl, desc=f"Ep {e+1}/{args.epochs}", leave=False)
        for i, (im, mk) in enumerate(pbar):
            im, mk = im.to(dev), mk.to(dev); opt.zero_grad()
            out = model(im); loss = crit(out, mk)
            loss.backward(); opt.step(); ls+=loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            with torch.no_grad():
                m = compute_metrics(out, mk)
                for k,v in m.items(): em.setdefault(k,[]).append(v)

        avg_l = ls/len(dl); res = {k: np.mean(v) for k,v in em.items()}; last_met = res
        current_fitness = (res['mIoU'] * 0.6) + (res['mAP'] * 0.4)
        writer.add_scalar('Train/Loss', avg_l, e+1)

        if current_fitness > best_fitness:
            best_fitness = current_fitness; patience_counter = 0
            torch.save(model.state_dict(), f"{sd}/best_model.pth"); saved_mark = "💾"
        else: patience_counter += 1

        for k in h.keys():
             if k == 'loss': h[k].append(avg_l)
             elif k in res: h[k].append(res[k])

        time_str = str(datetime.timedelta(seconds=int(time.time()-et)))
        print(f"{e+1:^5} | {avg_l:^7.4f} | {res['mIoU']:^6.3f} | {res['mF1']:^6.3f} | {res['mAP']:^6.3f} | {current_fitness:^6.3f} | {time_str:>9} | {patience_counter:^4} {saved_mark}")
        if patience_counter >= args.patience: print(f"\n🛑 Early Stopping"); break

    writer.close()
    plot_timeline(h, f"{sd}/training_timeline.png")

    df = pd.DataFrame([last_met]).round(4)
    print("\n🏆 Final Metrics:"); print(df.to_string(index=False))
    df.to_csv(f"{sd}/final_metrics.csv", index=False)
    print(f"⏱️ Total Time: {str(datetime.timedelta(seconds=int(time.time()-st)))}")

if __name__=='__main__': train()
