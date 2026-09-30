# Correcciones de contenido de la parte común (2026-09-29)

_Aprobadas por Romina. **No son parte del pase de forma**: cambian cifras y atribuciones a
propósito, así que el inventario de cifras cambia y ese cambio está documentado acá para que la
revisión independiente no lo lea como una pérdida._

## Diff de cifras, medido

`inventario_cifras.py` sobre `marco_teorico.tex`: **77 → 79 cifras**.

| | Cifras |
|---|---|
| **Quitadas** | `16,67\%` · `66,67\%` · `72\%` |
| **Agregadas** | `181` · `182` · `0,920` · `49\%` · `8\%` |

`inventario_estilo.py --comparar`: **0 alarmas**. Negritas, rayas, muletillas y cautelas
idénticas; palabras +187; páginas del capítulo 2 sin bajar. El documento pasa de 126 a
**127 páginas**.

## 1. Filiz et al. — atribución equivocada y una cifra invertida

**El problema.** El párrafo abría con «Filiz et al. evaluaron…» y a continuación daba tres
cifras que **son nuestras**, no del paper: 16,67 %, 66,67 % y 72 % salen de `Pruebas.xlsx`
(5/30, 20/30 y 41/57). Además presentaba el 43 % del paper como tasa **parcial**, cuando es su
tasa de **acierto**. Y cerraba concluyendo la «superioridad del análisis textual sobre el
análisis de archivos», que el capítulo 4 refuta.

**Verificado en la fuente primaria** (`5_bibliografia/Leido/Preprint-OnTheEffectiveness
OfRansomwareDecryptionTools.pdf`, 28 págs., pypdf):

- **ID Ransomware no aparece en el paper.** La única coincidencia sin distinción de mayúsculas
  está dentro de «ra**pid ransomware**».
- Crypto Sheriff **sí** aparece, y mucho. ⚠️ Un `grep` de `Sheriff` devuelve **0** porque el PDF
  usa la ligadura `ﬀ` (`Sheriﬀ`). No repetir ese grep sin normalizarla.
- Cifra textual, dos veces en el paper: *«Out of the 61 encrypted files and ransom notes
  uploaded, 43\% (n = 26) were correctly identified, a troubling 49\% (n = 30) were incorrectly
  identified and around a 8\% (n = 5) were partially identified.»*
- *«nearly half of the tools fail to recover compromised data satisfactorily»* sí es de ellos,
  del resumen, y es sobre las 28 herramientas de **descifrado**, no sobre identificación.

**La corrección.** El párrafo se parte en dos: lo que el paper dice (28 herramientas de 11
empresas, 61 muestras, casi la mitad falla; y su Crypto Sheriff con 43/49/8) y nuestra
evaluación como lo que es, con su base declarada y remitiendo a `sec:met_herramientas` y
`sec:res_herramientas` en vez de repetir las cifras fuera de contexto. Se elimina la frase de
superioridad. Se cita `cryptosheriff`, que estaba en el `.bib` sin usar.

## 2. Lemmou et al. — faltaba el alcance de la cifra

**El problema.** Decía «logró identificar correctamente el 99,45 %» sin el matiz que `CLAUDE.md`
marca como obligatorio. Cappo conoce el paper: leído así, 99,45 % en el capítulo 2 contra
cifras menores en el 4 se interpreta como que nos fue peor, cuando no es el mismo experimento.

**La corrección.** Se mantiene el crédito —sí clasifican familia, 181 de 182— y se agrega el
alcance: **mundo cerrado, sin separar entrenamiento de prueba**, con el LSA operando como
búsqueda de casi-duplicados sobre el mismo conjunto que clasifica. Se aclara además que su
componente de aprendizaje automático es **binario** (nombre de nota frente a nombre legítimo,
medida F de 0,920) y no interviene en la asignación de familia.

## Pendiente

El tono de estos dos párrafos ya está escrito en el registro aprobado por Romina (prosa
continua con conectores, sin negrita ni cursiva de énfasis, sin rayas como inciso, matices
intactos), así que el pase de forma sobre `marco_teorico.tex` **no necesita volver a tocarlos**.

---

# Introducción: el MISMO error de atribución

Al pasar por `introduccion.tex` apareció el mismo defecto del marco teórico, en el párrafo de
las herramientas: daba **16,67 %, 66,67 % y 72 %** citando a `paper_2_on_efectiveness`, es
decir atribuyéndole a Filiz et al. cifras que son nuestras. Se corrigió igual: lo que el paper
mide (43 % correctas, 49 % incorrectas, 8 % parciales sobre 61 muestras) con su cita, y nuestra
evaluación remitida al Capítulo 4 sin anticipar sus números.

**Conviene revisar si el patrón aparece en alguna otra sección**: apareció dos veces de forma
independiente, así que no era un descuido aislado.

Se agregó además a «Alcance y Limitaciones» la limitación de **una sola campaña por familia**,
como inciso (d). Es la de mayor alcance del trabajo y solo figuraba en el capítulo 4.

**Diff de cifras** (`introduccion.tex`, 14 → 16): quitadas `16,67\%`, `66,67\%`, `72\%`;
agregadas `43\%`, `49\%`, `8\%` y dos `4` de las remisiones al Capítulo 4.

# Metodología: §3.1 y §3.7

- **§3.7 estaba correcta.** Describe nuestra evaluación como nuestra, con la base bien declarada
  (57 notas de 22 familias) y las claves de cita correctas. No se tocó el contenido.
- **§3.1**: «tres experimentos» pasó a «tres líneas experimentales», con una frase que aclara que
  cada una derivó en una serie. La metodología anunciaba tres y el capítulo 4 entrega una docena.
- **Diff de cifras**: 121 → 122, solo un `4` de la remisión al capítulo.

# Grafía de la herramienta

El documento tenía «CryptoSheriff» y «Crypto Sheriff» conviviendo. El nombre correcto es
**Crypto Sheriff**, en dos palabras, como lo escriben No More Ransom y el paper de Filiz.
Unificado en los cuatro archivos. En `resultados.tex` las tres ocurrencias caían dentro de
§4.15 y §4.16, y el inventario de cifras de ese archivo **no se movió**: 1.866 antes y después.

# Estado de los controles

`inventario_estilo.py --comparar`: **0 alarmas** en los cuatro archivos. Ninguna negrita, raya,
muletilla ni cautela cambió. El documento pasó de 126 a **127 páginas**.

# Discusión (§4.16): pasada única, 2026-09-29

La editó una sola sesión, como se acordó, con los párrafos de los dos frentes enviados por
mensaje. **Las cifras que cambian vienen verificadas por el frente de archivos** contra
`cifras_finales_archivos.py` (commit `31d0390`); no las midió quien escribió la sección.

**Diff de cifras** (29 → 31): quitadas `89\%` y `100\%`; agregadas `88,6\,\%`, `86,7\,\%`,
`0,75` y `1`.

| Antes | Ahora | Por qué |
|---|---|---|
| «$\sim$89\% de exactitud» | **88,6 %** (Random Forest, Exp. 1) | la cifra exacta, de `exp1_binaria.csv` |
| «acierta el 100 % de los casos en que está presente» | se aplica al **86,7 %** y acierta en todos los casos | faltaba la cobertura al lado del acierto |
| «identifica la campaña y no la familia» | remite a la limitación del Exp. 2d | el 2d midió que la extensión determina la etiqueta **en este conjunto**, no que identifique la campaña |
| «seis familias» / «tres» | las mismas, con su umbral y su base | 6 con la media de 10 semillas del 2c, 3 con la de 5 semillas del 2g |

**Forma.** Se quitaron las negritas de énfasis y se pasaron las rayas de inciso a comas o
paréntesis. **Se conservaron las tres negritas de los rótulos del procedimiento por etapas**:
ahí la negrita nombra una etapa, que es lo más cercano a un término definido del capítulo,
mientras que las que se quitaron estaban sobre afirmaciones. Misma regla, resultado distinto
según qué hay debajo.

**No se tocó**, por pedido del frente de notas: «plantilla nunca vista» y «sobre algo más de la
mitad de las notas» (son la base de 0,9928 y 0,7417), la cautela sobre que una regla solo actúa
cuando el marcador reaparece en otra plantilla de la misma familia (la pidió Cappo), y el
«(146 notas, 95 contenidos distintos)», que es base vieja a propósito.

**Queda para Romina**, sin tocar: el «81,8 %» del tercer ítem del procedimiento, que no coincide
con el 83,9 % y el 81,2 % de la tabla comparativa. Es del frente de notas y está avisado.
