
import roboflow, os, glob, cv2, numpy as np, argparse
from tqdm import tqdm
def run():
    p = argparse.ArgumentParser(); p.add_argument('--api_key'); p.add_argument('--workspace'); p.add_argument('--project'); p.add_argument('--version', type=int)
    args = p.parse_args()
    try: ds = roboflow.Roboflow(api_key=args.api_key).workspace(args.workspace).project(args.project).version(args.version).download("yolov8")
    except: return print("❌ Download Failed")
    print("🔄 Processing Masks...")
    for split in ['train','valid','test']:
        mdir = f"{ds.location}/{split}/masks"; os.makedirs(mdir, exist_ok=True)
        for img in tqdm(glob.glob(f"{ds.location}/{split}/images/*"), desc=split):
            mask = np.zeros(cv2.imread(img).shape[:2], np.uint8)
            lbl = img.replace('images','labels').replace(os.path.splitext(img)[-1], '.txt')
            if os.path.exists(lbl):
                for l in open(lbl):
                    parts = list(map(float, l.split()))
                    cls = int(parts[0])+1; pts = np.array(parts[1:]).reshape(-1,2)*[mask.shape[1], mask.shape[0]]
                    cv2.fillPoly(mask, [pts.astype(int)], cls if cls<=2 else 2)
            cv2.imwrite(f"{mdir}/{os.path.splitext(os.path.basename(img))[0]}.png", mask)

if __name__=='__main__': run()
