import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, classification_report
)
from imblearn.over_sampling import SMOTE

np.random.seed(42)

# ── 1. DATASET ──────────────────────────────────
def generate_dataset(n=20000, fraud_ratio=0.025):
    X, y = make_classification(
        n_samples=n, n_features=28, n_informative=15,
        n_redundant=6, n_clusters_per_class=2,
        weights=[1-fraud_ratio, fraud_ratio],
        flip_y=0.008, random_state=42
    )
    df = pd.DataFrame(X, columns=[f'V{i}' for i in range(1,29)])
    df['Time']   = np.sort(np.random.uniform(0, 172800, n))
    df['Hour']   = (df['Time'] % 86400 // 3600).astype(int)
    df['Amount'] = np.where(
        y==1,
        np.random.lognormal(4.0, 1.5, n),
        np.random.lognormal(3.5, 1.2, n)
    ).clip(0.5, 5000)
    df['log_Amount'] = np.log1p(df['Amount'])
    df['Class']  = y
    n_fraud = y.sum()
    print(f"Dataset: {n:,} txns | {n_fraud} fraud ({n_fraud/n*100:.1f}%) | {n-n_fraud:,} legit")
    return df

# ── 2. MODELS ────────────────────────────────────
def get_models():
    return {
        'Logistic Regression': LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, subsample=0.8, random_state=42),
    }

# ── 3. TRAIN & EVALUATE ──────────────────────────
def train(df):
    feat = [c for c in df.columns if c not in ('Class','Time')]
    X = df[feat].values; y = df['Class'].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)

    sm = SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=5)
    Xr, yr = sm.fit_resample(Xtr, ytr)
    print(f"After SMOTE: {len(yr):,} samples ({yr.sum():,} fraud)\n")

    results = {}
    for name, mdl in get_models().items():
        print(f"Training {name}...")
        mdl.fit(Xr, yr)
        yp  = mdl.predict(Xte)
        ypr = mdl.predict_proba(Xte)[:,1]
        auc = roc_auc_score(yte, ypr)
        ap  = average_precision_score(yte, ypr)
        f1  = f1_score(yte, yp)
        fpr, tpr, _ = roc_curve(yte, ypr)
        prec, rec, _ = precision_recall_curve(yte, ypr)
        cm  = confusion_matrix(yte, yp)
        tn,fp,fn,tp = cm.ravel()
        print(f"  AUC={auc:.4f}  AP={ap:.4f}  F1={f1:.4f}  TP={tp} FP={fp} FN={fn}")
        results[name] = dict(model=mdl, yp=yp, ypr=ypr, auc=auc, ap=ap, f1=f1,
                             cm=cm, fpr=fpr, tpr=tpr, prec=prec, rec=rec,
                             report=classification_report(yte, yp, target_names=['Legit','Fraud']))

    fi = pd.Series(results['Random Forest']['model'].feature_importances_, index=feat).sort_values(ascending=False).head(15)
    return results, yte, fi

# ── 4. DASHBOARD ─────────────────────────────────
BG, PANEL, BORDER = '#0D0F14', '#141720', '#1E2535'
TEXT, MUTED = '#E8EAF0', '#6B7280'
MC = ['#4D9FFF','#00E5A0','#FFB347']

def setup_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=10, pad=7, fontfamily='monospace')
    ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.7)

def plot_dashboard(results, yte, fi, df):
    fig = plt.figure(figsize=(20,16), facecolor=BG)
    gs  = gridspec.GridSpec(3,4, figure=fig, hspace=0.5, wspace=0.38,
                            left=0.05, right=0.97, top=0.93, bottom=0.06)

    fig.text(0.5,0.967,'Financial Fraud Detection  ·  ML Dashboard',
             ha='center', fontsize=18, fontweight='bold', color=TEXT, fontfamily='monospace')
    fig.text(0.5,0.948,f"20,000 synthetic credit-card transactions  |  3-model ensemble  |  SMOTE resampling",
             ha='center', fontsize=9, color=MUTED, fontfamily='monospace')

    names = list(results.keys())

    # ROC
    ax = fig.add_subplot(gs[0,0:2]); setup_ax(ax,'ROC Curve')
    for (nm,r),c in zip(results.items(),MC):
        ax.plot(r['fpr'],r['tpr'],color=c,lw=2,label=f"{nm}  AUC={r['auc']:.3f}")
    ax.plot([0,1],[0,1],'--',color=MUTED,lw=1)
    ax.set_xlabel('False Positive Rate',color=MUTED,fontsize=8)
    ax.set_ylabel('True Positive Rate',color=MUTED,fontsize=8)
    ax.legend(fontsize=7.5,facecolor=BG,labelcolor=TEXT,edgecolor=BORDER)

    # PR
    ax = fig.add_subplot(gs[0,2:4]); setup_ax(ax,'Precision–Recall Curve')
    for (nm,r),c in zip(results.items(),MC):
        ax.plot(r['rec'],r['prec'],color=c,lw=2,label=f"{nm}  AP={r['ap']:.3f}")
    ax.set_xlabel('Recall',color=MUTED,fontsize=8)
    ax.set_ylabel('Precision',color=MUTED,fontsize=8)
    ax.legend(fontsize=7.5,facecolor=BG,labelcolor=TEXT,edgecolor=BORDER)

    # Confusion matrices
    for i,(nm,r) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1,i]); ax.set_facecolor(PANEL)
        cm = r['cm']
        ax.imshow(cm,cmap='Blues',aspect='auto')
        for rr in range(2):
            for cc in range(2):
                ax.text(cc,rr,f"{cm[rr,cc]:,}",ha='center',va='center',fontsize=13,
                        fontweight='bold',color='white' if cm[rr,cc]>cm.max()//2 else MUTED)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Pred Legit','Pred Fraud'],color=MUTED,fontsize=7.5)
        ax.set_yticklabels(['Act Legit','Act Fraud'],color=MUTED,fontsize=7.5)
        ax.set_title(f"{nm}\nF1 = {r['f1']:.3f}",color=TEXT,fontsize=9,pad=6,fontfamily='monospace')
        for sp in ax.spines.values(): sp.set_color(BORDER)

    # Metric bar
    ax = fig.add_subplot(gs[1,3]); setup_ax(ax,'Model Comparison')
    x,w = np.arange(3),0.22
    mlabels = ['ROC-AUC','Avg Prec','F1']
    for j,(nm,r) in enumerate(results.items()):
        vals = [r['auc'],r['ap'],r['f1']]
        bars = ax.bar(x+j*w,vals,w,color=MC[j],alpha=0.9,label=nm)
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.005,f'{v:.2f}',
                    ha='center',va='bottom',fontsize=6.5,color=TEXT)
    ax.set_xticks(x+w); ax.set_xticklabels(mlabels,color=MUTED,fontsize=8)
    ax.set_ylim(0,1.12)
    ax.legend(fontsize=7,facecolor=BG,labelcolor=TEXT,edgecolor=BORDER)
    ax.grid(axis='y',color=BORDER,linewidth=0.4)
    ax.grid(axis='x',visible=False)

    # Feature importance
    ax = fig.add_subplot(gs[2,0:2]); setup_ax(ax,'Top 15 Feature Importances  (Random Forest)')
    cols = ['#00E5A0' if i<3 else '#4D9FFF' for i in range(len(fi))]
    ax.barh(fi.index[::-1],fi.values[::-1],color=cols[::-1],height=0.6)
    ax.set_xlabel('Importance',color=MUTED,fontsize=8)
    ax.grid(axis='x',color=BORDER,linewidth=0.4); ax.grid(axis='y',visible=False)

    # Amount dist
    ax = fig.add_subplot(gs[2,2]); setup_ax(ax,'Transaction Amount Distribution')
    bins = np.linspace(0,1000,50)
    ax.hist(df[df['Class']==0]['Amount'].clip(0,1000),bins=bins,color='#4D9FFF',alpha=0.55,label='Legit',density=True)
    ax.hist(df[df['Class']==1]['Amount'].clip(0,1000),bins=bins,color='#FF4C6A',alpha=0.75,label='Fraud',density=True)
    ax.set_xlabel('Amount (₹)',color=MUTED,fontsize=8)
    ax.set_ylabel('Density',color=MUTED,fontsize=8)
    ax.legend(fontsize=8,facecolor=BG,labelcolor=TEXT,edgecolor=BORDER)

    # Fraud by hour
    ax = fig.add_subplot(gs[2,3]); setup_ax(ax,'Fraud Rate by Hour of Day')
    fr = (df[df['Class']==1].groupby('Hour').size() /
          (df.groupby('Hour').size() + 1e-9) * 100).reindex(range(24),fill_value=0)
    ax.bar(fr.index,fr.values,color='#FF4C6A',alpha=0.85,width=0.8)
    ax.set_xlabel('Hour',color=MUTED,fontsize=8)
    ax.set_ylabel('Fraud Rate (%)',color=MUTED,fontsize=8)
    ax.grid(axis='y',color=BORDER,linewidth=0.4); ax.grid(axis='x',visible=False)

    out = '/mnt/user-data/outputs/fraud_detection_dashboard.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"\nDashboard saved → {out}")

# ── MAIN ─────────────────────────────────────────
if __name__ == '__main__':
    print("="*55)
    print("  FINANCIAL FRAUD DETECTION SYSTEM")
    print("="*55+"\n")
    df = generate_dataset()
    results, yte, fi = train(df)
    print("\n── Classification Reports ──")
    for nm,r in results.items():
        print(f"\n{nm}:\n{r['report']}")
    print("\nRendering dashboard...")
    plot_dashboard(results, yte, fi, df)
    best = max(results, key=lambda k: results[k]['auc'])
    print(f"\nBest model: {best}  (AUC = {results[best]['auc']:.4f})")
