
import torch, numpy as np
from sklearn.metrics import average_precision_score

def compute_metrics(pred_logits, target, num_classes=3):
    probs = torch.softmax(pred_logits, dim=1); pred_mask = torch.argmax(probs, dim=1)
    probs_np = probs.detach().cpu().numpy(); target_np = target.detach().cpu().numpy()
    pred_mask = pred_mask.detach().cpu(); target = target.detach().cpu()

    stats = {k: [] for k in ['iou', 'f1', 'prec', 'rec', 'ap']}

    for cls in range(num_classes):
        p = (pred_mask == cls); t = (target == cls)
        tp = (p & t).sum().float(); fp = (p & ~t).sum().float()
        fn = (~p & t).sum().float(); union = (p | t).sum().float()

        iou = (tp / (union + 1e-6)).item()
        prec = (tp / (tp + fp + 1e-6)).item()
        rec = (tp / (tp + fn + 1e-6)).item()
        f1 = 2 * (prec * rec) / (prec + rec + 1e-6)

        stats['iou'].append(iou); stats['f1'].append(f1)
        stats['prec'].append(prec); stats['rec'].append(rec)

        y_true = (target_np == cls).astype(int).reshape(-1)
        y_score = probs_np[:, cls].reshape(-1)
        ap = average_precision_score(y_true[::10], y_score[::10]) if np.sum(y_true) > 0 else 0.0
        stats['ap'].append(ap)

    results = {}
    for i, n in enumerate(['Bg', 'BP', 'SP']):
        results[f'{n}_IoU'] = stats['iou'][i]; results[f'{n}_F1'] = stats['f1'][i]
        results[f'{n}_Pre'] = stats['prec'][i]; results[f'{n}_Rec'] = stats['rec'][i]
        results[f'{n}_AP'] = stats['ap'][i]

    results['mIoU'] = np.mean(stats['iou']); results['mF1'] = np.mean(stats['f1'])
    results['mPre'] = np.mean(stats['prec']); results['mRec'] = np.mean(stats['rec'])
    results['mAP'] = np.mean(stats['ap'])
    return results
