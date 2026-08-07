# Bibliografía — fuentes nuevas candidatas (2026-06-22)

Fuentes encontradas para reforzar la tesis, organizadas por objetivo. Antes de citar,
**verificar el PDF y los datos de cada una** (año, autores, venue). Marcadas con prioridad.

## A. Detección por entropía / Chi² / archivos cifrados (Objetivos 1 y 2)

1. **★★★ Comparison of Entropy Calculation Methods for Ransomware Encrypted File Identification**
   - *Entropy* (MDPI), 2022. Usa **NapierOne** (mismo dataset que ustedes).
   - Compara 53 tests para diferenciar datos cifrados de otros tipos; evalúa 11 técnicas
     de entropía sobre >270.000 archivos. Respalda directamente su elección de Shannon + Chi² + Monte Carlo.
   - https://www.mdpi.com/1099-4300/24/10/1503  | PDF: https://arxiv.org/pdf/2210.13376

2. **★★★ EnCoD: Distinguishing Compressed and Encrypted File Fragments**
   - arXiv 2010.07754. Muestra que los métodos estadísticos **fallan** al separar
     archivos comprimidos de cifrados → **respalda empíricamente por qué su multiclase por
     archivos no discrimina** (Objetivo 2).
   - https://arxiv.org/pdf/2010.07754

3. **Intermittent File Encryption in Ransomware: Measurement, Modeling, and Detection**
   - arXiv 2510.15133. Cifrado intermitente y su impacto en la detección por entropía.
   - https://arxiv.org/html/2510.15133v1

4. **A novel framework for malware detection using entropy-based statistical features and ML across file types**
   - ResearchGate, 2024. Features estadísticas de entropía + ML por tipo de archivo.
   - https://www.researchgate.net/publication/391586464

## B. Clasificación de familias de ransomware con ML (Objetivos 3 y 4)

5. **★★ Enhancing ransomware defense: deep learning-based detection and family-wise classification of evolving threats**
   - PMC (PubMed Central), 2024. Detección 99.99%, clasificación por categoría 85.48%,
     **identificación de familia 74.65%** → buen punto de comparación para sus resultados.
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC11640932/

6. **Ransomware Classification and Detection With Machine Learning Algorithms**
   - arXiv 2207.00894. Comparativa RF, XGBoost, etc.
   - https://arxiv.org/pdf/2207.00894

7. **Application of Explainable ML in Detecting and Classifying Ransomware Families Based on API Call Analysis**
   - arXiv 2210.11235. Enfoque por API calls (contrastar con su enfoque post-ataque).
   - https://arxiv.org/pdf/2210.11235

8. **Ransomware Detection and Classification using Machine Learning** (arXiv 2311.16143)
   - https://ar5iv.labs.arxiv.org/html/2311.16143

## C. Análisis NLP de notas de rescate (núcleo de la tesis — Objetivo 4)

9. **★★★ In-Depth Analysis of Ransom Note Files**
   - 2021. Señala que **no había estudios académicos previos sobre archivos de notas de rescate**;
     usa LSA y ML para clasificar nombres/contenidos. (Ya tienen el PDF en `Leido/`; citarlo como
     antecedente directo y diferenciarse: ustedes hacen clasificación **multiclase por familia**.)
   - https://www.researchgate.net/publication/356019484

## Cómo seguir
- Verificar cada fuente y exportar a BibTeX hacia `latex_capitulos/referencias.bib`.
- Las marcadas ★★★ son las que más fortalecen el argumento; priorizar #1, #2 y #9.
- Pídeme: *"agrega las fuentes ★★★ a referencias.bib en formato BibTeX"* y lo hago.
