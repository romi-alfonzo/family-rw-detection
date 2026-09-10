# Señales sin explotar para mejorar la clasificación (2026-08-20)

Menú de señales que **ya están en disco** y que el clasificador de notas (hoy solo texto,
TF-IDF) no usa. Objetivo: mejorar la clasificación **sin recolectar más datos**. Todo lo de
abajo está verificado contra código/documento, no de memoria.

## Regla de oro (antes de tocar nada)

Toda señal nueva se evalúa bajo el **mismo protocolo P2ret + near-dup** (retener una
plantilla/variante entera por familia para test, coseno ≥ 0,90 para no partir casi-duplicados
entre train y test). Si no, el número sube por **fuga** (el modelo memoriza un IOC de un sample)
y la mejora es ilusoria. Es el error que todo el protocolo de la tesis existe para evitar.
Recordar además la trampa **campaña-vs-familia**.

## Señales de las NOTAS que hoy se ignoran

### 1. Nombre de archivo de la nota — la más fuerte y ya planificada
- Qué es: el nombre real que deja el malware (`# DECRYPT MY FILES #.txt`, `[id].README.txt`,
  `Info.hta`). Es **la señal principal de ID Ransomware**.
- ¿La tenemos? Sí: columna `archivo` del manifiesto (preservada, sin usar como feature).
- Valor: alto. Dato medido: al **renombrar** el archivo, solo **9/30** familias sobreviven por
  bytes → las otras 21 se identificaban por el **nombre/extensión**.
- Trampa: fragilísima (un rename la rompe); a veces trae el ID de la víctima → memorización. En
  ThreatLabz muchos nombres son etiquetas del curador (`conti1.txt`), no el nombre real.
- Estado: **es el experimento 2d** (elemento de acción 2 del tutor): cruzar
  **bytes × nombre × extensión** para las 4 familias que solo marcan el nombre.

### 2. Extensión original de la nota (`extension_original`: `.html`/`.hta`/`.txt`)
- Débil sola, pero casi gratis como feature categórica que acompaña al nombre. Documentada.

### 3. Marcadores / IOCs del texto — mejor candidato "nuevo"
- Qué es: URL, `.onion`, email, clave, ID, BTC. Ya los extrae `grafo_marcadores.py`.
  Hallados en el corpus: **URL 159 · ONION 106 · EMAIL 74 · CLAVE 37 · ID 9 · BTC 5**.
- Valor: alto en teoría — medido que **ningún IOC se comparte entre familias distintas**, y la
  fracción de pares de una familia unidos por un marcador compartido **correlaciona +0,425
  (Spearman) con el F1 por familia**.
- Trampa doble: (a) muchos IOC son de un solo sample → evaluar dejando fuera la plantilla entera
  o memoriza; (b) **11 de 97 plantillas no tienen ningún marcador** → para esas la vista no ve
  nada. Sirve como **vista adicional**, no como reemplazo del texto.

### 4. Idioma de la nota — potente pero CONGELADO a propósito
- Medido: el eje idioma es el más productivo para textos distintos (alemán/francés dan los
  cosenos más bajos), **pero choca con la cohesión de B.3** (Chimera lo prueba: el mismo mensaje
  en dos idiomas hunde su cohesión). **Decisión tomada: frenado hasta medir el efecto.** Usar
  como experimento acotado con esa advertencia, no como win gratis.

### 5. Formato/render de la nota (`.txt` vs `.html` vs `.hta`, ventana vs archivo dejado)
- Señal estructural muy débil, ya registrada. Baja prioridad.

## Señales de los ARCHIVOS cifrados (ya bastante exprimidas)

- **Magic bytes** (prefijo/sufijo) y **extensión añadida a los archivos de la víctima**: ya lo
  hace `deteccion_estructural.py` (Exp. 2b) — 9/30 familias por firma, con cobertura reportada.
  El margen que queda: **combinar** esas firmas con el nombre/extensión → otra vez el exp 2d.

## La palanca más grande — NO se decide sola

**Fusión de los dos frentes (majority voting notas + archivos).** Es lo que más subiría el
número global y es **pedido abierto del tutor**. Pero por regla del proyecto los dos frentes van
**separados** y hoy **no hay clasificador combinado**: es decisión de Romina + Cappo, no se hace
por iniciativa propia. Queda marcada como palanca #1 **pendiente de decisión del tutor**.

## Orden recomendado (sin recolectar nada más)

1. **Exp 2d**: bytes × nombre × extensión (ya está en el roadmap; junta señales 1 + 2 + magic bytes).
2. **Vista de marcadores/IOCs** como tercera vista del clasificador de notas, evaluada
   leave-one-template-out (señal 3).
3. **Idioma** como experimento acotado, con la advertencia de B.3 (señal 4).
4. **Fusión de frentes** → elevar al tutor, no ejecutar.

## Referencias en el repo (para retomar)

- `2_codigo/grafo_marcadores.py` → extracción de marcadores; salidas en
  `4_resultados/resultados_grafo_marcadores/`.
- `2_codigo/deteccion_estructural.py` → firmas de bytes + extensión añadida (Exp. 2b).
- `3_datos/manifiesto_corpus_v2.csv` → columnas `archivo` y `extension_original` (señales 1 y 2).
- Memoria del proyecto: `proximo-experimento-2d`, `comparacion-id-ransomware`.
- Detalle del choque idioma-vs-B.3 y de los marcadores: `ESTADO_TESIS.md` (bloques de CHIMERA,
  B.3 y marcadores).
