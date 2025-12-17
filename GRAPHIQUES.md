# Guide pour Génération des Graphiques

## Fichiers de Données Créés

1. **data_performance.csv** - Performance par résolution
2. **data_kernels.csv** - Distribution charge computationnelle
3. **data_gpu_usage.csv** - Utilisation GPU dans le temps
4. **data_scalability.csv** - Scalabilité théorique

## Graphiques Suggérés

### 1. Distribution de la Charge par Kernel (Bar Chart)

**Fichier**: `data_kernels.csv`

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data_kernels.csv')
plt.figure(figsize=(10, 6))
plt.bar(df['Kernel'], df['Percent'], color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7b731', '#5f27cd'])
plt.xlabel('Kernel CUDA')
plt.ylabel('% du temps total')
plt.title('Distribution de la Charge Computationnelle')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('graph_kernels.png', dpi=300)
```

**Interprétation**: Montre que `update_reservoir` (40%) et `hysteresis` (25%) sont les goulots.

### 2. Performance vs Résolution (Line + Scatter)

**Fichier**: `data_performance.csv`

```python
df = pd.read_csv('data_performance.csv')
fig, ax1 = plt.subplots(figsize=(10, 6))

# FPS
ax1.plot(df['Pixels']/1e6, df['FPS_GPU'], 'o-', color='#4ecdc4', linewidth=2, markersize=10, label='FPS GPU')
ax1.set_xlabel('Résolution (Megapixels)')
ax1.set_ylabel('FPS', color='#4ecdc4')
ax1.tick_params(axis='y', labelcolor='#4ecdc4')
ax1.axhline(y=30, color='red', linestyle='--', label='Temps réel (30 FPS)')

# Mémoire
ax2 = ax1.twinx()
ax2.plot(df['Pixels']/1e6, df['Memory_MB'], 's-', color='#f7b731', linewidth=2, markersize=10, label='Mémoire GPU')
ax2.set_ylabel('Mémoire GPU (MB)', color='#f7b731')
ax2.tick_params(axis='y', labelcolor='#f7b731')

plt.title('Scalabilité: FPS et Mémoire vs Résolution')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
plt.tight_layout()
plt.savefig('graph_scalability.png', dpi=300)
```

**Interprétation**: FPS stable ~20 malgré augmentation résolution → Bonne scalabilité.

### 3. Utilisation GPU dans le Temps (Stacked Area)

**Fichier**: `data_gpu_usage.csv`

```python
df = pd.read_csv('data_gpu_usage.csv')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# SM et Mémoire
ax1.fill_between(df['Time_s'], df['SM_percent'], alpha=0.5, color='#4ecdc4', label='SM Usage')
ax1.fill_between(df['Time_s'], df['Memory_percent'], alpha=0.5, color='#f7b731', label='Memory Usage')
ax1.set_ylabel('Utilisation (%)')
ax1.legend(loc='upper right')
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3)
ax1.set_title('Utilisation GPU au fil du temps')

# Température et Puissance
ax2_temp = ax2
ax2_temp.plot(df['Time_s'], df['Temperature_C'], 'o-', color='#ff6b6b', linewidth=2, label='Température')
ax2_temp.set_ylabel('Température (°C)', color='#ff6b6b')
ax2_temp.tick_params(axis='y', labelcolor='#ff6b6b')

ax2_power = ax2.twinx()
ax2_power.plot(df['Time_s'], df['Power_W'], 's-', color='#5f27cd', linewidth=2, label='Puissance')
ax2_power.set_ylabel('Puissance (W)', color='#5f27cd')
ax2_power.tick_params(axis='y', labelcolor='#5f27cd')

ax2.set_xlabel('Temps (s)')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_gpu_usage.png', dpi=300)
```

**Interprétation**: Plateau stable à 95% SM → GPU bien utilisé pendant traitement.

### 4. Efficacité Mémoire (Scatter)

**Fichier**: `data_performance.csv`

```python
df = pd.read_csv('data_performance.csv')
df['MB_per_Mpixel'] = df['Memory_MB'] / (df['Pixels'] / 1e6)

plt.figure(figsize=(8, 6))
plt.scatter(df['Pixels']/1e6, df['MB_per_Mpixel'], s=200, alpha=0.6, c=['#4ecdc4', '#f7b731'])
for i, row in df.iterrows():
    plt.annotate(row['Resolution'], (row['Pixels']/1e6, row['MB_per_Mpixel']), 
                xytext=(5, 5), textcoords='offset points')
plt.xlabel('Résolution (Megapixels)')
plt.ylabel('Mémoire GPU (MB/Mpixel)')
plt.title('Efficacité Mémoire GPU')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_memory_efficiency.png', dpi=300)
```

**Interprétation**: Décroissance MB/Mpixel → Plus efficace sur hautes résolutions.

### 5. Projection Scalabilité Théorique

**Fichier**: `data_scalability.csv`

```python
df = pd.read_csv('data_scalability.csv')
colors = {'Low': '#4ecdc4', 'Medium': '#45b7d1', 'High': '#f7b731', 'Ultra': '#ff6b6b'}

plt.figure(figsize=(10, 6))
for cat in df['Category'].unique():
    subset = df[df['Category'] == cat]
    plt.scatter(subset['Pixels_M'], subset['FPS_Theoretical'], 
               s=300, alpha=0.7, color=colors[cat], label=cat)
    plt.annotate(subset.iloc[0]['Resolution'], 
                (subset.iloc[0]['Pixels_M'], subset.iloc[0]['FPS_Theoretical']),
                xytext=(5, 5), textcoords='offset points', fontsize=10)

plt.axhline(y=30, color='green', linestyle='--', linewidth=2, label='Temps réel (30 FPS)')
plt.xlabel('Résolution (Megapixels)')
plt.ylabel('FPS Théorique')
plt.title('Projection Scalabilité Multi-Résolution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('graph_theoretical_scalability.png', dpi=300)
```

**Interprétation**: Full HD atteint presque temps réel, 4K hors portée sans optimisation.

## Génération Rapide (Python)

```bash
# Installer matplotlib si nécessaire
pip install matplotlib pandas

# Générer tous les graphiques
python3 << 'EOF'
import matplotlib.pyplot as plt
import pandas as pd

# Graphique 1: Distribution kernels
df = pd.read_csv('data_kernels.csv')
plt.figure(figsize=(10, 6))
plt.bar(df['Kernel'], df['Percent'], color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7b731', '#5f27cd'])
plt.xlabel('Kernel CUDA')
plt.ylabel('% du temps total')
plt.title('Distribution de la Charge Computationnelle')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('graph_kernels.png', dpi=300)
print("✓ graph_kernels.png créé")

# Graphique 2: Scalabilité
df = pd.read_csv('data_performance.csv')
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(df['Pixels']/1e6, df['FPS_GPU'], 'o-', color='#4ecdc4', linewidth=2, markersize=10)
ax1.axhline(y=30, color='red', linestyle='--', label='Temps réel')
ax1.set_xlabel('Résolution (Megapixels)')
ax1.set_ylabel('FPS GPU')
ax1.set_title('Performance vs Résolution')
plt.tight_layout()
plt.savefig('graph_scalability.png', dpi=300)
print("✓ graph_scalability.png créé")

print("\n✓ Tous les graphiques générés!")
EOF
```

## Intégration LaTeX

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{graph_kernels.png}
\caption{Distribution de la charge computationnelle par kernel}
\label{fig:kernels}
\end{figure}
```
