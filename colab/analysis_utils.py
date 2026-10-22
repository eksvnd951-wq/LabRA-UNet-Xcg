
import matplotlib.pyplot as plt, seaborn as sns, numpy as np, cv2
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

def plot_cm(yt, yp, cls, path):
    cm = confusion_matrix(yt, yp, labels=range(len(cls)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=cls, yticklabels=cls, cmap='Blues')
    plt.title('Normalized Confusion Matrix (Pixel-wise)')
    plt.tight_layout(); plt.savefig(path); plt.close()

def plot_object_curves(y_true, y_scores, name, d):
    if len(y_true) == 0: return
    p, r, t = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    f1 = 2 * (p * r) / (p + r + 1e-6)

    plt.figure(figsize=(7,5))
    plt.plot(r, p, color='darkorange', lw=2, label=f'AP={ap:.4f}')
    plt.fill_between(r, p, color='darkorange', alpha=0.3)
    plt.title(f'{name} Precision-Recall Curve')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{d}/{name}_PR_Curve.png"); plt.close()

    metrics = [('Precision', p[:-1], 'blue'), ('Recall', r[:-1], 'green'), ('F1_Score', f1[:-1], 'red')]
    for m_name, vals, col in metrics:
        plt.figure(figsize=(7,5))
        plt.plot(t, vals, color=col, lw=2)
        plt.fill_between(t, vals, color=col, alpha=0.3)
        plt.title(f'{name} {m_name} over Threshold')
        plt.xlabel('Confidence Threshold'); plt.ylabel(m_name)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{d}/{name}_{m_name}_Curve.png"); plt.close()

def plot_timeline(h, path):
    epochs = range(1, len(h['loss'])+1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    keys = [('loss','Loss','r'), ('mIoU','mIoU','b'), ('mF1','mF1','g'),
            ('mAP','mAP','purple'), ('mPre','Precision','orange'), ('mRec','Recall','cyan')]
    for i, (k, title, color) in enumerate(keys):
        r_idx, c_idx = divmod(i, 3)
        if k in h and len(h[k]) > 0:
            axes[r_idx, c_idx].plot(epochs, h[k], marker='o', color=color)
            axes[r_idx, c_idx].fill_between(epochs, h[k], color=color, alpha=0.1)
            axes[r_idx, c_idx].set_title(title); axes[r_idx, c_idx].grid(True)
    plt.tight_layout(); plt.savefig(path); plt.close()

def save_4_column_comparison(img, gt_mask, pred_mask, path):
    # BP(Small)=Red, SP(Large)=Yellow
    h, w = img.shape[:2]
    COLOR_BP = (0, 0, 255); COLOR_SP = (0, 255, 255)

    col1 = img.copy()
    col2 = np.zeros((h, w, 3), dtype=np.uint8)
    col2[gt_mask == 1] = COLOR_BP; col2[gt_mask == 2] = COLOR_SP
    col3 = np.zeros((h, w, 3), dtype=np.uint8)
    col3[pred_mask == 1] = COLOR_BP; col3[pred_mask == 2] = COLOR_SP
    overlay = img.copy()
    overlay[pred_mask == 1] = COLOR_BP; overlay[pred_mask == 2] = COLOR_SP
    col4 = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX; s = 2.0; t = 2; c = (255,255,255)
    cv2.putText(col1, "Input", (20, 50), font, s, c, t)
    cv2.putText(col2, "GT", (20, 50), font, s, c, t)
    cv2.putText(col3, "Prediction", (20, 50), font, s, c, t)
    cv2.putText(col4, "Overlay", (20, 50), font, s, c, t)

    final = np.hstack((col1, col2, col3, col4))
    cv2.imwrite(path, final)
