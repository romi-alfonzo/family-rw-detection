---
name: cluster-nidtec
description: "Cómo usar el cluster del NIDTEC (arandu) para los experimentos de la tesis — SLURM, python3.11, /scratch, sin Internet"
metadata: 
  node_type: memory
  type: reference
  originSessionId: f677c6c9-8bd5-463c-be1d-f93df8a59eaa
  modified: 2026-08-17T02:30:49.344Z
---

Cluster computacional del NIDTEC (FPUNA), acceso concedido el 2026-08-04 para la tesis
(ver [[tesis-ransomware-contexto]]). Guía operativa completa: `C:\Users\Romina\Tesis\SERVIDOR_PASOS_AHORA.md`;
scripts de trabajo SLURM ya escritos en `C:\Users\Romina\Tesis\2_codigo\slurm\`.

- Usuario `ralfonzo`. Master **`arandu`** (desde ahí se ejecuta), nodos de cómputo c1–c4.
- **Los programas se encolan con SLURM** (`sbatch job.sh`, `squeue -u ralfonzo`, `scancel`),
  no se ejecutan directo. La salida queda en `slurm-<nro>.out`.
- **`python3.11`** es la versión buena (`python3` apunta a 3.9); paquetes con `pip3.11`.
- Trabajar en **`/scratch/ralfonzo`** — el HOME no tiene espacio suficiente.
- **NO hay acceso a Internet** desde el cluster (reglamento): `pip install` puede fallar.
  Por eso `extractor_notas.py` tiene fallback regex y funciona sin `beautifulsoup4`.
- Hay **GPU** (torch disponible), pero scikit-learn es CPU: la GPU solo serviría para un
  experimento futuro con transformers.
- Almacenamiento **declarado** temporal (reglamento: borrado 60 días tras el fin de uso),
  pero **en la práctica NO se limpia**: Romina tiene archivos de más de un año en `/scratch`
  (verificado 2026-08-17, corrigiendo lo que decía antes esta memoria). Bajar los resultados
  igual por respaldo, pero **no usar el borrado como argumento de urgencia** — se lo dijeron
  varias veces y molesta.
- En los scripts, el paralelismo respeta `SLURM_CPUS_PER_TASK` (nunca `n_jobs=-1`, que
  sobresuscribiría el nodo).
- **Obligación del reglamento:** mencionar el uso del cluster en cualquier publicación →
  agradecimientos de la tesis (proyecto LABO16-167, CONACYT/PROCIENCIA, NIDTEC-FPUNA).
- Dataset allá: `Napierone-small` con **30 familias** desde el 2026-08-15 (BLACKBASTA subido;
  sigue faltando `Z-Safe`, los benignos — no hace falta para la multiclase).
- **Todo archivo que haya que subir al cluster se copia SIEMPRE a
  `C:\Users\Romina\Tesis\PARA_SUBIR_AL_CLUSTER\`** (plano, sin subcarpetas: los scripts `.py`
  y los `job_*.sh` van juntos, igual que en `/scratch/ralfonzo/tesis/`). Romina sube desde esa
  carpeta con WinSCP. No basta con editar en `2_codigo\` — hay que copiar ahí también
  (pedido explícito, 2026-08-16).
- **Finales de línea: LF, nunca CRLF.** `sbatch` rechaza scripts con `\r\n` («Batch script
  contains DOS line breaks»). Tras editar en Windows cualquier `.sh` o `.py` que vaya al
  cluster, convertir a LF antes de copiarlo a PARA_SUBIR (pasó el 2026-08-16). Arreglo de
  emergencia allá: `sed -i 's/\r$//' archivo`.
- **Al darle comandos del cluster, incluir SIEMPRE `cd /scratch/ralfonzo/tesis &&` adelante**,
  en cada bloque y no una sola vez al principio. Ella entra en `~` (`/home_data/ralfonzo`) y
  copia bloques sueltos; sin el `cd`, `srun` y `sbatch` no encuentran el script. Así están
  escritos todos los pasos de `SERVIDOR_PASOS_AHORA.md`.
- Los nodos **c1 y c3 tienen `/scratch` degradado** (un trabajo queda horas al 2 % de CPU sin
  leer datos): lanzar siempre con `--nodelist=c2`.
