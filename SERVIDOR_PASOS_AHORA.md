# Cluster NIDTEC (arandu) — qué hacer ahora, paso a paso

_Actualizado 2026-08-04, ya con acceso confirmado (usuario `ralfonzo`) y NapierOne-small en el cluster._

## Lo que hay que saber del entorno (del correo + reglamento)

| Punto | Implicancia para nosotras |
|---|---|
| Es un **cluster con SLURM** (master `arandu`, nodos `c1`–`c4`) | Los programas **no** se ejecutan directamente: se encolan con `sbatch`. Ya están hechos los 4 scripts de trabajo en `2_codigo/slurm/`. |
| `python3` = 3.9, **`python3.11`** es la versión nueva; paquetes con `pip3.11` | Todos los scripts de trabajo ya usan `python3.11`. |
| **El HOME no tiene espacio**; usar `/scratch/ralfonzo` | Trabajar TODO en `/scratch/ralfonzo`, incluido mover ahí NapierOne. |
| **No hay acceso a Internet** en el cluster (reglamento) | `pip3.11 install` puede fallar. El código ya funciona **sin `beautifulsoup4`** (fallback propio, verificado: las 28 notas HTML se extraen igual). Solo son imprescindibles `scikit-learn`, `numpy`, `pandas`. |
| Hay **GPU** disponible (ejemplo con torch) | No la usan estos trabajos (scikit-learn es CPU). Queda para un experimento futuro con transformers. |
| El almacenamiento es **temporal** y se borra 60 días después del fin de uso | Bajar los resultados a la PC siempre que termine un trabajo. |
> ⚠️ CORREGIDO 2026-08-17: El reglamento dice que el almacenamiento es temporal (borrado 60 dias tras el fin de uso), pero EN LA PRACTICA NO SE LIMPIA: Romina tiene archivos de mas de un anho en /scratch (verificado 2026-08-17). Bajar los resultados igual, por respaldo, pero NO usar el borrado como argumento de urgencia.
| Obligación de **mencionar el uso del cluster** en publicaciones | Anotado: va en los agradecimientos de la tesis (Bloque E). |

⚠️ **Además:** tu `ls` mostró **29 carpetas**, no 31. Faltan **`BLACKBASTA-small`** (una de las 30
familias) y **`Z-Safe`** (los archivos benignos, necesarios para la detección binaria).
El Experimento 2 se puede correr igual con 29 familias (y queda más limpio, sin la clase benigna),
pero conviene preguntar si el dataset completo está en otro lado.

---

## Paso 0 — Diagnóstico y preparación (10 min)

Conectarse a `arandu` y correr:

```bash
mkdir -p /scratch/ralfonzo/tesis && cd /scratch/ralfonzo/tesis
df -h /scratch/ralfonzo .
python3.11 --version
pip3.11 list 2>/dev/null | grep -iE "scikit|numpy|pandas|beautifulsoup|matplotlib"
sinfo                     # estado de los nodos y particiones
```

Mover NapierOne del HOME (sin espacio) a scratch, y buscar lo que falta:

```bash
mv ~/Napierone-small /scratch/ralfonzo/   ;   ls /scratch/ralfonzo/Napierone-small | wc -l
find / -maxdepth 4 \( -iname "*BLACKBASTA*" -o -iname "*Z-Safe*" \) 2>/dev/null
```

### Estado real del entorno (verificado el 2026-08-04)

| Componente | Estado |
|---|---|
| `/scratch/ralfonzo` | ✅ 4,2 TB libres |
| Python | ✅ 3.11.11 |
| `numpy` | ✅ 2.3.1 |
| `beautifulsoup4` | ✅ 4.13.4 (así que las notas HTML usan el parser bueno) |
| **`scikit-learn`** | ❌ **falta** |
| **`pandas`** | ❌ **falta** |

### Instalación offline de scikit-learn y pandas

Como el cluster no tiene Internet, los paquetes ya están descargados en tu PC:
**`wheels_cluster/`** (77 MB, 10 archivos `.whl` para Linux/Python 3.11).

1. Subir la carpeta completa a `/scratch/ralfonzo/tesis/wheels_cluster/`.
2. Instalar sin red:

```bash
cd /scratch/ralfonzo/tesis
pip3.11 install --user --no-index --find-links wheels_cluster scikit-learn pandas
```

3. Verificar:

```bash
python3.11 -c "import sklearn, pandas, numpy, bs4; print('sklearn', sklearn.__version__, '| pandas', pandas.__version__, '| numpy', numpy.__version__, '| bs4', bs4.__version__)"
```

Debe imprimir **`sklearn 1.6.1`** — la misma versión usada localmente, así los resultados del
cluster son directamente comparables con la corrida canónica de la PC. `numpy` y `bs4` ya están
instalados y pip no los toca.

Si diera error de plataforma (glibc vieja), avisame: se bajan wheels con etiqueta manylinux más antigua.

## Paso 0.b — Subir los archivos desde tu PC

A `/scratch/ralfonzo/tesis/` (con WinSCP o `scp`):

```
2_codigo/clasificador_notas_v2.py
2_codigo/extractor_notas.py
2_codigo/gridsearch_notas.py
2_codigo/deteccion_estructural.py
2_codigo/family-rw-detection/advanced_features.py
2_codigo/family-rw-detection/train_advanced.py
2_codigo/slurm/*.sh                    (los 4 scripts de trabajo)
3_datos/corpus_v2/                     (carpeta completa, 146 notas, ~1 MB)
wheels_cluster/                        (77 MB — los paquetes para instalar sin Internet)
```

Todo plano en la misma carpeta (los `.sh` también). El corpus queda en
`/scratch/ralfonzo/tesis/corpus_v2/` y los scripts lo detectan solo.

```bash
cd /scratch/ralfonzo/tesis && chmod +x *.sh && ls
```

---

## Paso 1 — Smoke test ▶ SIEMPRE PRIMERO (~3 min)

Verifica entorno, corpus y pipeline antes de encolar nada largo:

```bash
cd /scratch/ralfonzo/tesis
sbatch job_smoke.sh
squeue -u ralfonzo                      # ver la cola
cat slurm-smoke-*.out                   # cuando termine
```

**Debe decir:** `Notas: 146 | Familias: 30`, `Notas vacias (debe ser 0): 0` y terminar con
«BÚSQUEDA COMPLETADA». Si `bs4` no está instalado dirá que usa el fallback: **está bien**.
Si algo falla acá, mandame la salida antes de seguir.

## Paso 2 — Trabajo C: detección estructural (~2 min)

El más rápido y el que da un resultado **nuevo** para la tesis (Experimento 2b):

```bash
cd /scratch/ralfonzo/tesis
DATOS=/scratch/ralfonzo/Napierone-small sbatch --export=ALL,DATOS job_estructural.sh
cat slurm-estructural-*.out
```

**Qué esperar:** las familias con marca estructural detectada + exactitud multiclase LOO.
Validación contra tu `Pruebas.xlsx`: las 9 familias «SI*» (GANDCRAB, LORENZ, MAZE,
MEDUSALOCKER, PHOBOS, RYUK, SODINOKIBI, TESLACRYPT, WANNACRY) deberían aparecer con marca.
WANNACRY debería mostrar el prefijo `57414e4143525921` = «WANACRY!».

**Bajar:** `resultados_estructural/`

## Paso 3 — Trabajo B: 275 features estadísticas (~30-90 min)

Blinda el resultado negativo del Experimento 2 (hoy se apoya en solo 2 features):

```bash
cd /scratch/ralfonzo/tesis
DATOS=/scratch/ralfonzo/Napierone-small sbatch --export=ALL,DATOS job_features.sh
squeue -u ralfonzo
tail -f slurm-features-*.out            # Ctrl+C para dejar de mirar; el job sigue
```

Va a imprimir `[SKIP] Folder not found` para BLACKBASTA y Z-Safe: **es esperado**, sigue con las 29.

**Qué esperar:** exactitud multiclase que **siga siendo baja** (~10 %) aun con 275 features.
Ese es el resultado deseado: confirma que la indistinguibilidad no era falta de features.

**Bajar:** `advanced_features.csv` + `slurm-features-*.out`

## Paso 4 — Trabajo A: hiperparámetros (varias horas)

El más largo; dejarlo encolado al final:

```bash
cd /scratch/ralfonzo/tesis
sbatch job_gridsearch.sh
squeue -u ralfonzo
tail -f slurm-gridsearch-*.out
```

Tiene checkpoint tras cada bloque: si se corta, los CSV parciales quedan.
Si el cluster está muy ocupado, editar `#SBATCH --cpus-per-task=16` a un número menor
para entrar antes en la cola.

**Qué esperar:** macro-F1 por encima de la corrida sin tuning (P1 0,760 · P2 0,435).
En la prueba local con grilla mínima ya subió a P1 0,771 / P2 0,459.

**Bajar:** `resultados_gridsearch/` + `slurm-gridsearch-*.out`

---

## Comandos útiles de SLURM

```bash
squeue -u ralfonzo                 # mis trabajos en cola / corriendo
scancel <nro>                      # cancelar un trabajo
sacct -j <nro> --format=JobID,JobName,State,Elapsed,MaxRSS   # cómo terminó
sinfo                              # nodos disponibles
```
La salida de cada trabajo queda en `slurm-<nombre>-<nro>.out` en la carpeta desde donde se envió.

## Al volver con los resultados

Copiar a la PC dentro de `4_resultados/` (cada carpeta en su lugar) y decirme:
**«volvieron los resultados del servidor»** → agrego las secciones nuevas al capítulo 4.
Recordá bajar todo: el espacio del cluster es temporal.
