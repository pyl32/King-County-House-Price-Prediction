import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FIG = '/home/claude/king-county-project/figures'
C = {'green': '#2C6E49', 'blue': '#1E3A5F', 'warm': '#C17A35',
     'red': '#C0392B', 'gray': '#5D6D7E', 'light': '#EAF2F8'}

plt.rcParams.update({
    'figure.dpi': 150, 'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 12, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9
})

# ── LOAD ──────────────────────────────────────────────────────────────────────
house = gpd.read_file('/home/claude/course2/project2/project2/Project2_dataset.gdb', layer=0)
df = pd.DataFrame(house.drop(columns='geometry'))
df['house_age'] = 2015 - df['yr_built']
df['was_renovated'] = (df['yr_renovated'] > 0).astype(int)
df['price_per_sqft'] = df['price'] / df['sqft_living']
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
print("Price stats:\n", df['price'].describe().round(0))

# ── FIG 1: Distribution map + histogram ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
sc = axes[0].scatter(house['long'], house['lat'],
    c=house['price']/1e6, cmap='RdYlGn_r', alpha=0.35, s=2, vmin=0.1, vmax=2.5)
cb = plt.colorbar(sc, ax=axes[0], fraction=0.03, pad=0.04)
cb.set_label('Price ($ millions)', fontsize=9)
axes[0].set_title('(A) House Sale Prices — King County, WA', fontweight='bold')
axes[0].set_xlabel('Longitude'); axes[0].set_ylabel('Latitude')
axes[0].set_facecolor('#F0F4F8')
axes[0].annotate('Bellevue / Kirkland\n(high price cluster)', xy=(-122.2, 47.62),
    xytext=(-121.9, 47.7), fontsize=7, color=C['red'],
    arrowprops=dict(arrowstyle='->', color=C['red'], lw=0.8))

axes[1].hist(df['price']/1e6, bins=80, color=C['green'], edgecolor='white', lw=0.3)
axes[1].axvline(df['price'].median()/1e6, color=C['warm'], lw=2, ls='--',
    label=f"Median: ${df['price'].median()/1e6:.2f}M")
axes[1].axvline(df['price'].mean()/1e6, color=C['red'], lw=2, ls='-.',
    label=f"Mean: ${df['price'].mean()/1e6:.2f}M")
axes[1].set_xlabel('Sale Price ($ millions)'); axes[1].set_ylabel('Number of Homes')
axes[1].set_title('(B) Price Distribution (Right-Skewed)', fontweight='bold')
axes[1].legend(fontsize=9); axes[1].set_xlim(0, 4)
axes[1].annotate('75th pct =\n$645K', xy=(0.645, 1200), fontsize=8, color=C['gray'])
plt.tight_layout()
plt.savefig(f'{FIG}/fig1_price_distribution.png', bbox_inches='tight')
plt.close()
print("Fig1 saved")

# ── FIG 2: Heatmap ────────────────────────────────────────────────────────────
num_cols = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors',
            'waterfront','view','condition','grade','sqft_above','sqft_basement',
            'house_age','lat','long','sqft_living15','sqft_lot15']
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
    center=0, vmin=-1, vmax=1, annot_kws={'size': 7}, linewidths=0.3,
    ax=ax, cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
ax.set_title('Correlation Matrix — King County House Price Features\n'
    '(key predictors: sqft_living r=0.70, grade r=0.67, bathrooms r=0.53, lat r=0.31)',
    fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG}/fig2_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("Fig2 saved")

# ── FIG 3: Bar chart correlations ─────────────────────────────────────────────
price_corr = corr['price'].drop('price').sort_values()
colors = [C['red'] if v < 0 else C['green'] for v in price_corr]
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(price_corr.index, price_corr.values, color=colors, edgecolor='white')
ax.axvline(0, color='black', lw=0.8)
for bar, val in zip(bars, price_corr.values):
    offset = 0.01 if val >= 0 else -0.01
    ax.text(val + offset, bar.get_y() + bar.get_height()/2,
        f'{val:.2f}', va='center', ha='left' if val >= 0 else 'right', fontsize=8)
ax.set_xlabel('Pearson Correlation Coefficient')
ax.set_title('Feature Correlations with House Price\n(green = positive, red = negative)',
    fontweight='bold')
ax.set_xlim(-0.5, 0.85)
plt.tight_layout()
plt.savefig(f'{FIG}/fig3_price_correlations.png', bbox_inches='tight')
plt.close()
print("Fig3 saved")

# ── FIG 4: Scatter top 4 predictors ──────────────────────────────────────────
feats = [('sqft_living', 'Living Area (sqft)'), ('grade', 'Construction Grade'),
         ('bathrooms', 'Bathrooms'), ('lat', 'Latitude')]
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, (feat, label) in zip(axes, feats):
    s = df.sample(3000, random_state=42)
    ax.scatter(s[feat], s['price']/1e6, alpha=0.2, s=6, color=C['green'])
    z = np.polyfit(s[feat], s['price']/1e6, 1)
    xl = np.linspace(s[feat].min(), s[feat].max(), 100)
    ax.plot(xl, np.poly1d(z)(xl), color=C['warm'], lw=2)
    r = np.corrcoef(s[feat], s['price'])[0,1]
    ax.set_xlabel(label); ax.set_ylabel('Price ($M)')
    ax.set_title(f'{label}\nr = {r:.2f}', fontweight='bold', fontsize=9)
plt.suptitle('Price vs Top Predictors — King County, WA (n=3,000 sample)',
    fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG}/fig4_scatter_predictors.png', bbox_inches='tight')
plt.close()
print("Fig4 saved")

# ── FIG 5: Boxplots by grade and view ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
grade_groups = [df[df['grade'] == g]['price']/1e6 for g in sorted(df['grade'].unique())]
bp1 = axes[0].boxplot(grade_groups, labels=sorted(df['grade'].unique()),
    patch_artist=True, medianprops=dict(color=C['warm'], lw=2),
    flierprops=dict(marker='.', markersize=2, alpha=0.2, color=C['gray']))
for patch in bp1['boxes']:
    patch.set_facecolor(C['light']); patch.set_edgecolor(C['blue'])
axes[0].set_xlabel('Construction Grade (1–13)'); axes[0].set_ylabel('Price ($ millions)')
axes[0].set_title('(A) Price by Construction Grade', fontweight='bold')

view_groups = [df[df['view'] == v]['price']/1e6 for v in sorted(df['view'].unique())]
bp2 = axes[1].boxplot(view_groups, labels=['None','Fair','Average','Good','Excellent'],
    patch_artist=True, medianprops=dict(color=C['warm'], lw=2),
    flierprops=dict(marker='.', markersize=2, alpha=0.2, color=C['gray']))
for patch in bp2['boxes']:
    patch.set_facecolor(C['light']); patch.set_edgecolor(C['blue'])
axes[1].set_xlabel('View Rating'); axes[1].set_ylabel('Price ($ millions)')
axes[1].set_title('(B) Price by View Rating', fontweight='bold')
plt.suptitle('House Price Distributions by Key Categorical Features', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG}/fig5_boxplots.png', bbox_inches='tight')
plt.close()
print("Fig5 saved")

# ── FIG 6: Spatial price density ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
sc2 = ax.scatter(house['long'], house['lat'],
    c=house['price']/1e6, cmap='hot_r', alpha=0.5, s=3, vmin=0.15, vmax=1.5)
cb2 = plt.colorbar(sc2, ax=ax, fraction=0.03)
cb2.set_label('Price ($ millions)', fontsize=9)
ax.set_title('Spatial Price Density — King County\nHigh prices cluster in N/NE (Bellevue, Kirkland, Redmond)',
    fontweight='bold', fontsize=10)
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_facecolor('#1a1a2e')
plt.tight_layout()
plt.savefig(f'{FIG}/fig6_spatial_density.png', bbox_inches='tight')
plt.close()
print("Fig6 saved")

# ── FEATURE ENGINEERING + MODELS ──────────────────────────────────────────────
FEATURES = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',
            'view','condition','grade','sqft_above','sqft_basement',
            'house_age','was_renovated','lat','long','sqft_living15','sqft_lot15']
X = df[FEATURES]; y = df['price']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain: {len(Xtr):,}  Test: {len(Xte):,}")

sc_obj = StandardScaler()
Xtr_sc = sc_obj.fit_transform(Xtr)
Xte_sc = sc_obj.transform(Xte)

# MLR
mlr = LinearRegression().fit(Xtr, ytr)
yp_mlr = mlr.predict(Xte)
mlr_r2 = r2_score(yte, yp_mlr)
mlr_mae = mean_absolute_error(yte, yp_mlr)
mlr_rmse = np.sqrt(mean_squared_error(yte, yp_mlr))

# RF (properly tuned — fixes Project 4's naive n_estimators=28)
rf = RandomForestRegressor(n_estimators=200, max_features=0.4,
    min_samples_leaf=2, random_state=42, n_jobs=-1).fit(Xtr, ytr)
yp_rf = rf.predict(Xte)
rf_r2 = r2_score(yte, yp_rf)
rf_mae = mean_absolute_error(yte, yp_rf)
rf_rmse = np.sqrt(mean_squared_error(yte, yp_rf))

# MLP (FIXED from Project 2: proper scaling + deeper arch + early stopping)
mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation='relu',
    learning_rate_init=0.001, max_iter=600, random_state=42,
    early_stopping=True, validation_fraction=0.1).fit(Xtr_sc, ytr)
yp_mlp = mlp.predict(Xte_sc)
mlp_r2 = r2_score(yte, yp_mlp)
mlp_mae = mean_absolute_error(yte, yp_mlp)
mlp_rmse = np.sqrt(mean_squared_error(yte, yp_mlp))

print(f"\nMLR  R²={mlr_r2:.4f}  MAE=${mlr_mae:,.0f}  RMSE=${mlr_rmse:,.0f}")
print(f"RF   R²={rf_r2:.4f}  MAE=${rf_mae:,.0f}  RMSE=${rf_rmse:,.0f}")
print(f"MLP  R²={mlp_r2:.4f}  MAE=${mlp_mae:,.0f}  RMSE=${mlp_rmse:,.0f}")

# ── FIG 7: Actual vs Predicted ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, yp, col, r2v) in zip(axes, [
    ('Multiple Linear\nRegression', yp_mlr, C['blue'], mlr_r2),
    ('Random Forest\n(tuned)', yp_rf, C['green'], rf_r2),
    ('Neural Network\n(MLP, fixed)', yp_mlp, C['warm'], mlp_r2)
]):
    ax.scatter(yte/1e6, yp/1e6, alpha=0.15, s=6, color=col)
    lim = 4.0
    ax.plot([0, lim], [0, lim], 'r--', lw=1.5, label='Perfect fit')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(f'{name}\nR² = {r2v:.3f}', fontweight='bold', fontsize=10)
    ax.set_xlabel('Actual Price ($M)'); ax.set_ylabel('Predicted Price ($M)')
    ax.legend(fontsize=8)
plt.suptitle('Actual vs Predicted — King County House Price Models',
    fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG}/fig7_actual_vs_predicted.png', bbox_inches='tight')
plt.close()
print("Fig7 saved")

# ── FIG 8: Model comparison ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
labels = ['MLR', 'Random\nForest', 'MLP\n(Fixed)']
r2s  = [mlr_r2, rf_r2, mlp_r2]
maes = [mlr_mae/1000, rf_mae/1000, mlp_mae/1000]
rmses= [mlr_rmse/1000, rf_rmse/1000, mlp_rmse/1000]
bar_colors = [C['blue'], C['green'], C['warm']]

for ax, vals, title, yl in zip(axes,
    [r2s, maes, rmses],
    ['R² Score (↑ better)', 'MAE ($K) — ↓ better', 'RMSE ($K) — ↓ better'],
    ['R²', 'MAE ($K)', 'RMSE ($K)']):
    bars = ax.bar(labels, vals, color=bar_colors, edgecolor='white', width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003*max(vals),
            f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_ylabel(yl); ax.set_ylim(0, max(vals)*1.18)
plt.suptitle('Model Performance Comparison — King County House Price Prediction',
    fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG}/fig8_model_comparison.png', bbox_inches='tight')
plt.close()
print("Fig8 saved")

# ── FIG 9: Feature importance ─────────────────────────────────────────────────
fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
bar_fi = ax.barh(fi.index, fi.values,
    color=[C['green'] if v > fi.median() else '#95B8A0' for v in fi], edgecolor='white')
ax.axvline(fi.median(), color=C['warm'], lw=1.5, ls='--', label='Median importance')
for bar, val in zip(bar_fi, fi.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
        f'{val:.3f}', va='center', fontsize=8)
ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)')
ax.set_title('Random Forest — Feature Importance\n(top feature: lat drives spatial premium)',
    fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG}/fig9_feature_importance.png', bbox_inches='tight')
plt.close()
print("Fig9 saved")

# ── FIG 10: Residuals ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
res_mlr = yte - yp_mlr; res_rf = yte - yp_rf

axes[0].scatter(yp_mlr/1e6, res_mlr/1e6, alpha=0.15, s=5, color=C['blue'], label='MLR')
axes[0].scatter(yp_rf/1e6,  res_rf/1e6,  alpha=0.15, s=5, color=C['green'], label='RF')
axes[0].axhline(0, color='red', lw=1.5, ls='--')
axes[0].set_xlabel('Predicted Price ($M)'); axes[0].set_ylabel('Residual ($M)')
axes[0].set_title('(A) Residuals vs Predicted\n(RF residuals tighter around zero)',
    fontweight='bold')
axes[0].legend(fontsize=9)

axes[1].hist(res_mlr/1e6, bins=60, alpha=0.6, color=C['blue'], label='MLR', density=True)
axes[1].hist(res_rf/1e6,  bins=60, alpha=0.6, color=C['green'], label='RF', density=True)
axes[1].axvline(0, color='red', lw=1.5, ls='--')
axes[1].set_xlabel('Residual ($M)'); axes[1].set_ylabel('Density')
axes[1].set_title('(B) Residual Distribution\n(RF residuals more symmetric)',
    fontweight='bold')
axes[1].legend(fontsize=9)
plt.suptitle('Residual Analysis — MLR vs Random Forest', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG}/fig10_residuals.png', bbox_inches='tight')
plt.close()
print("Fig10 saved")

# ── FIG 11: MLP loss curve + topology ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(mlp.loss_curve_, color=C['green'], lw=2, label='Training loss')
axes[0].set_xlabel('Iteration (Epoch)'); axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('(A) MLP Training Loss Curve\n(early stopping prevented overfitting)',
    fontweight='bold')
axes[0].legend(fontsize=9)
final_epoch = len(mlp.loss_curve_)
axes[0].annotate(f'Stopped at\nepoch {final_epoch}',
    xy=(final_epoch, mlp.loss_curve_[-1]),
    xytext=(final_epoch*0.7, mlp.loss_curve_[0]*0.8),
    fontsize=8, color=C['warm'],
    arrowprops=dict(arrowstyle='->', color=C['warm']))

# Topology schematic
ax2 = axes[1]; ax2.set_xlim(0, 5); ax2.set_ylim(0, 10); ax2.axis('off')
layers = [(0.5, 17, 'Input\n17 features', C['blue']),
          (1.5, 256, 'Hidden 1\n256 neurons', C['green']),
          (2.5, 128, 'Hidden 2\n128 neurons', C['green']),
          (3.5, 64, 'Hidden 3\n64 neurons', C['green']),
          (4.5, 1, 'Output\nPrice ($)', C['warm'])]
for xi, n, lbl, col in layers:
    show = min(n, 7)
    ys = np.linspace(2, 8, show)
    for y_c in ys:
        circle = plt.Circle((xi, y_c), 0.22, color=col, alpha=0.75, zorder=3)
        ax2.add_patch(circle)
    if n > 7:
        ax2.text(xi, 1.5, f'({n})', ha='center', fontsize=7, color=col)
    ax2.text(xi, 9.2, lbl, ha='center', fontsize=7.5, fontweight='bold', color=col)
# Arrows between layers
for i in range(len(layers)-1):
    ax2.annotate('', xy=(layers[i+1][0]-0.22, 5), xytext=(layers[i][0]+0.22, 5),
        arrowprops=dict(arrowstyle='->', color=C['gray'], lw=1.5))
ax2.set_title('(B) Neural Network Topology\nReLU | lr=0.001 | early stopping',
    fontweight='bold', fontsize=10)

plt.suptitle('MLP Neural Network — Architecture and Training', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG}/fig11_mlp_topology.png', bbox_inches='tight')
plt.close()
print("Fig11 saved")

# ── SAVE DATA ─────────────────────────────────────────────────────────────────
df_save = df[['id','price','bedrooms','bathrooms','sqft_living','sqft_lot',
    'floors','waterfront','view','condition','grade','sqft_above','sqft_basement',
    'yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15',
    'house_age','was_renovated','price_per_sqft']].copy()
df_save.to_csv('/home/claude/king-county-project/data/king_county_house_sales.csv', index=False)

results_df = pd.DataFrame([
    {'Model':'Multiple Linear Regression','R2':round(mlr_r2,4),'MAE':round(mlr_mae),'RMSE':round(mlr_rmse)},
    {'Model':'Random Forest (tuned)','R2':round(rf_r2,4),'MAE':round(rf_mae),'RMSE':round(rf_rmse)},
    {'Model':'MLP Neural Network (fixed)','R2':round(mlp_r2,4),'MAE':round(mlp_mae),'RMSE':round(mlp_rmse)},
])
results_df.to_csv('/home/claude/king-county-project/data/model_results.csv', index=False)
print("\nAll done!\n", results_df.to_string(index=False))

# Store results for report generation
import pickle
with open('/home/claude/king-county-project/results.pkl', 'wb') as f:
    pickle.dump({
        'mlr_r2': mlr_r2, 'mlr_mae': mlr_mae, 'mlr_rmse': mlr_rmse,
        'rf_r2': rf_r2, 'rf_mae': rf_mae, 'rf_rmse': rf_rmse,
        'mlp_r2': mlp_r2, 'mlp_mae': mlp_mae, 'mlp_rmse': mlp_rmse,
        'n_total': len(df), 'n_train': len(Xtr), 'n_test': len(Xte),
        'top_feature': fi.index[-1], 'top_fi': fi.values[-1],
    }, f)
