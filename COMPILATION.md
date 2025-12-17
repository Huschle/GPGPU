# Instructions de Compilation du Rapport

## Fichiers créés

1. **rapport.tex** - Version LaTeX complète (format scientifique professionnel)
2. **RAPPORT.md** - Version Markdown (consultation immédiate)

## Pour générer le PDF depuis LaTeX

### 1. Installer LaTeX (si nécessaire)

```bash
sudo apt install texlive-latex-base texlive-latex-extra texlive-lang-french texlive-science
```

### 2. Compiler le rapport

```bash
cd /home/tidjk/scia/gpgpu/project
pdflatex rapport.tex
pdflatex rapport.tex  # 2ème passe pour références croisées
```

Le fichier **rapport.pdf** sera généré.

## Contenu du rapport

- Introduction et objectifs
- Description détaillée de l'algorithme (5 étapes)
- Implémentation CPU vs GPU
- Résultats expérimentaux avec métriques précises
- Analyse de performance (95% utilisation GPU)
- Discussion et perspectives d'amélioration
- Approche scientifique et analytique

## Résultats clés

- ✓ Résultats qualitatifs **ACCEPTABLES**
- ✓ Utilisation GPU : **95%**
- ✓ Traitement vidéo complet : **17.19s** (GPU) vs >10s timeout (CPU)
- ✓ Pipeline complet fonctionnel
