#!/usr/bin/env python3
"""
Script de génération automatique des graphiques pour le rapport GPGPU
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configuration style global
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300

print("🎨 Génération des graphiques...")

# ============================================================================
# Graphique 1: Distribution de la charge par kernel (Bar Chart)
# ============================================================================
print("\n[1/5] Distribution des kernels...")
df_kernels = pd.read_csv('data_kernels.csv')

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7b731', '#5f27cd']
bars = ax.bar(df_kernels['Kernel'], df_kernels['Percent'], color=colors, edgecolor='black', linewidth=1.2)

# Ajouter les valeurs sur les barres
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_xlabel('Kernel CUDA', fontweight='bold')
ax.set_ylabel('Pourcentage du temps total (%)', fontweight='bold')
ax.set_title('Distribution de la Charge Computationnelle par Kernel', fontweight='bold', pad=20)
ax.set_ylim(0, 50)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('graph_kernels.png', dpi=300, bbox_inches='tight')
print("   ✓ graph_kernels.png créé")
plt.close()

# ============================================================================
# Graphique 2: Performance vs Résolution (Dual-axis)
# ============================================================================
print("[2/5] Performance vs résolution...")
df_perf = pd.read_csv('data_performance.csv')

fig, ax1 = plt.subplots(figsize=(11, 6))

# Axe 1: FPS
color1 = '#4ecdc4'
ax1.plot(df_perf['Pixels']/1e6, df_perf['FPS_GPU'], 'o-', color=color1, 
         linewidth=3, markersize=12, label='FPS GPU', markeredgecolor='black', markeredgewidth=1.5)
ax1.axhline(y=30, color='#2ecc71', linestyle='--', linewidth=2, label='Temps réel (30 FPS)', alpha=0.8)
ax1.set_xlabel('Résolution (Megapixels)', fontweight='bold')
ax1.set_ylabel('FPS GPU', color=color1, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 35)
ax1.grid(True, alpha=0.3, linestyle='--')

# Axe 2: Mémoire
ax2 = ax1.twinx()
color2 = '#f7b731'
ax2.plot(df_perf['Pixels']/1e6, df_perf['Memory_MB'], 's-', color=color2, 
         linewidth=3, markersize=12, label='Mémoire GPU', markeredgecolor='black', markeredgewidth=1.5)
ax2.set_ylabel('Mémoire GPU (MB)', color=color2, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color2)

# Annotations
for i, row in df_perf.iterrows():
    ax1.annotate(row['Resolution'], 
                (row['Pixels']/1e6, row['FPS_GPU']),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))

plt.title('Scalabilité: FPS et Mémoire vs Résolution', fontweight='bold', pad=20)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9, edgecolor='black')
plt.tight_layout()
plt.savefig('graph_scalability.png', dpi=300, bbox_inches='tight')
print("   ✓ graph_scalability.png créé")
plt.close()

# ============================================================================
# Graphique 3: Utilisation GPU dans le temps (Multi-panel)
# ============================================================================
print("[3/5] Utilisation GPU temporelle...")
df_gpu = pd.read_csv('data_gpu_usage.csv')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

# Panel 1: SM et Mémoire
ax1.fill_between(df_gpu['Time_s'], df_gpu['SM_percent'], alpha=0.6, color='#4ecdc4', label='SM Usage', edgecolor='black', linewidth=1.5)
ax1.fill_between(df_gpu['Time_s'], df_gpu['Memory_percent'], alpha=0.6, color='#f7b731', label='Memory Usage', edgecolor='black', linewidth=1.5)
ax1.plot(df_gpu['Time_s'], df_gpu['SM_percent'], 'o-', color='#2c7a7b', linewidth=2, markersize=6)
ax1.plot(df_gpu['Time_s'], df_gpu['Memory_percent'], 's-', color='#c68e17', linewidth=2, markersize=6)
ax1.set_ylabel('Utilisation (%)', fontweight='bold')
ax1.legend(loc='upper right', framealpha=0.9, edgecolor='black')
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_title('Utilisation GPU au fil du temps', fontweight='bold', pad=15)

# Panel 2: Température et Puissance
color_temp = '#ff6b6b'
color_power = '#5f27cd'

ax2.plot(df_gpu['Time_s'], df_gpu['Temperature_C'], 'o-', color=color_temp, 
         linewidth=3, markersize=8, label='Température', markeredgecolor='black', markeredgewidth=1.5)
ax2.set_ylabel('Température (°C)', color=color_temp, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_temp)
ax2.set_ylim(60, 95)
ax2.axhline(y=90, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Seuil thermique')

ax3 = ax2.twinx()
ax3.plot(df_gpu['Time_s'], df_gpu['Power_W'], 's-', color=color_power, 
         linewidth=3, markersize=8, label='Puissance', markeredgecolor='black', markeredgewidth=1.5)
ax3.set_ylabel('Puissance (W)', color=color_power, fontweight='bold')
ax3.tick_params(axis='y', labelcolor=color_power)
ax3.set_ylim(0, 80)

ax2.set_xlabel('Temps (s)', fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right', framealpha=0.9, edgecolor='black')

plt.tight_layout()
plt.savefig('graph_gpu_usage.png', dpi=300, bbox_inches='tight')
print("   ✓ graph_gpu_usage.png créé")
plt.close()

# ============================================================================
# Graphique 4: Efficacité Mémoire (Scatter)
# ============================================================================
print("[4/5] Efficacité mémoire...")
df_perf['MB_per_Mpixel'] = df_perf['Memory_MB'] / (df_perf['Pixels'] / 1e6)

fig, ax = plt.subplots(figsize=(9, 6))
colors_scatter = ['#4ecdc4', '#f7b731']
scatter = ax.scatter(df_perf['Pixels']/1e6, df_perf['MB_per_Mpixel'], 
                    s=400, alpha=0.7, c=colors_scatter, edgecolors='black', linewidths=2)

for i, row in df_perf.iterrows():
    ax.annotate(f"{row['Resolution']}\n{row['MB_per_Mpixel']:.0f} MB/Mpix", 
                (row['Pixels']/1e6, row['MB_per_Mpixel']), 
                xytext=(15, 15), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='black', lw=1.5))

ax.set_xlabel('Résolution (Megapixels)', fontweight='bold')
ax.set_ylabel('Mémoire GPU (MB/Mpixel)', fontweight='bold')
ax.set_title('Efficacité Mémoire GPU par Résolution', fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(600, 850)
plt.tight_layout()
plt.savefig('graph_memory_efficiency.png', dpi=300, bbox_inches='tight')
print("   ✓ graph_memory_efficiency.png créé")
plt.close()

# ============================================================================
# Graphique 5: Projection Scalabilité Théorique
# ============================================================================
print("[5/5] Projection scalabilité...")
df_scale = pd.read_csv('data_scalability.csv')

fig, ax = plt.subplots(figsize=(11, 7))
colors_cat = {'Low': '#2ecc71', 'Medium': '#4ecdc4', 'High': '#f7b731', 'Ultra': '#ff6b6b'}

for cat in df_scale['Category'].unique():
    subset = df_scale[df_scale['Category'] == cat]
    ax.scatter(subset['Pixels_M'], subset['FPS_Theoretical'], 
              s=500, alpha=0.7, color=colors_cat[cat], label=cat,
              edgecolors='black', linewidths=2)
    
    # Annotations
    for _, row in subset.iterrows():
        ax.annotate(f"{row['Resolution']}\n{row['FPS_Theoretical']:.1f} FPS", 
                   (row['Pixels_M'], row['FPS_Theoretical']),
                   xytext=(15, 0), textcoords='offset points', 
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', alpha=0.9))

# Ligne temps réel
ax.axhline(y=30, color='#2ecc71', linestyle='--', linewidth=3, label='Temps réel (30 FPS)', alpha=0.8)
ax.axhline(y=24, color='#3498db', linestyle=':', linewidth=2, label='Cinéma (24 FPS)', alpha=0.6)

ax.set_xlabel('Résolution (Megapixels)', fontweight='bold')
ax.set_ylabel('FPS Théorique', fontweight='bold')
ax.set_title('Projection Scalabilité Multi-Résolution (Implémentation Actuelle)', fontweight='bold', pad=20)
ax.legend(loc='upper right', framealpha=0.9, edgecolor='black', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--', which='both')
ax.set_yscale('log')
ax.set_ylim(0.5, 50)
plt.tight_layout()
plt.savefig('graph_theoretical_scalability.png', dpi=300, bbox_inches='tight')
print("   ✓ graph_theoretical_scalability.png créé")
plt.close()

# ============================================================================
# BONUS: Graphique combiné - Vue d'ensemble
# ============================================================================
print("[BONUS] Vue d'ensemble synthétique...")
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Top-left: Kernels
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(df_kernels['Kernel'], df_kernels['Percent'], color=colors, edgecolor='black', linewidth=1)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.0f}%',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax1.set_ylabel('% Temps', fontweight='bold')
ax1.set_title('Distribution Kernels', fontweight='bold', fontsize=12)
ax1.tick_params(axis='x', rotation=45, labelsize=8)
ax1.grid(axis='y', alpha=0.3)

# Top-right: FPS
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(df_perf['Pixels']/1e6, df_perf['FPS_GPU'], 'o-', color='#4ecdc4', linewidth=2, markersize=10)
ax2.axhline(y=30, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.7)
ax2.set_xlabel('Résolution (Mpix)', fontweight='bold')
ax2.set_ylabel('FPS GPU', fontweight='bold')
ax2.set_title('Scalabilité FPS', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3)

# Bottom-left: GPU Usage
ax3 = fig.add_subplot(gs[1, 0])
ax3.fill_between(df_gpu['Time_s'], df_gpu['SM_percent'], alpha=0.5, color='#4ecdc4', label='SM')
ax3.fill_between(df_gpu['Time_s'], df_gpu['Memory_percent'], alpha=0.5, color='#f7b731', label='Mem')
ax3.set_xlabel('Temps (s)', fontweight='bold')
ax3.set_ylabel('Utilisation (%)', fontweight='bold')
ax3.set_title('Utilisation GPU', fontweight='bold', fontsize=12)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Bottom-right: Métriques clés (Texte)
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')
metrics_text = f"""
MÉTRIQUES CLÉS

Performance
• FPS Full HD: 20.35
• Débit: 114 Mpix/s
• Temps réel: 65%

Utilisation GPU
• SM: 95% (excellent)
• Mémoire: 35%
• Température: 89°C

Efficacité
• Mem: 743 MB/Mpix
• Goulots:
  - Reservoir: 40%
  - Hysteresis: 25%

Projection
• Actuel: 20 FPS
• Optimisé: 82 FPS
• Gain: +310%
"""
ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='black', linewidth=2, alpha=0.9))

plt.suptitle('Vue d\'Ensemble des Performances GPGPU', fontweight='bold', fontsize=16, y=0.98)
plt.savefig('graph_overview.png', dpi=300, bbox_inches='tight')
print("   ✓ graph_overview.png créé")
plt.close()

print("\n" + "="*60)
print("✅ TOUS LES GRAPHIQUES GÉNÉRÉS AVEC SUCCÈS!")
print("="*60)
print("\nFichiers créés:")
print("  1. graph_kernels.png - Distribution charge computationnelle")
print("  2. graph_scalability.png - FPS et mémoire vs résolution")
print("  3. graph_gpu_usage.png - Timeline utilisation GPU")
print("  4. graph_memory_efficiency.png - Efficacité mémoire")
print("  5. graph_theoretical_scalability.png - Projection multi-résolution")
print("  6. graph_overview.png - Vue d'ensemble synthétique")
print("\nUtilisation:")
print("  - Intégration LaTeX: \\includegraphics[width=0.8\\textwidth]{graph_*.png}")
print("  - Markdown: ![Description](graph_*.png)")
print("="*60)
