# Memoria de Claude — cómo restaurarla en la computadora nueva

Esta carpeta es una **copia de la memoria persistente de Claude Code** hecha el 2026-08-20
para el cambio de computadora. En la máquina vieja vivía en:

```
C:\Users\Romina\.claude\projects\C--Users-Romina-Tesis\memory\
```

## Pasos en la computadora nueva

1. Instalar Claude Code e iniciar sesión.
2. Clonar el repo y **abrir Claude Code una vez dentro de la carpeta de la tesis** (eso crea
   la carpeta del proyecto en `~\.claude\projects\`).
3. Buscar la carpeta que Claude Code creó para este proyecto:
   `C:\Users\<usuario>\.claude\projects\<clave-del-proyecto>\` — la clave se deriva de la
   ruta de la carpeta (si la tesis queda en `C:\Users\<usuario>\Tesis`, la clave será
   `C--Users-<usuario>-Tesis`).
4. Copiar **todos los `.md` de esta carpeta** adentro de `...\<clave-del-proyecto>\memory\`
   (crear `memory\` si no existe).
5. Abrir un chat y verificar con: *«¿qué tenés en memoria?»* — tiene que nombrar el índice
   (contexto de la tesis, cluster, Lemmou, ID Ransomware, experimento 2d, commits).

## Qué es cada archivo

- `MEMORY.md` — el índice; es lo que Claude carga en cada sesión.
- Los demás — una memoria por archivo (contexto, preferencias, cluster, correcciones de
  Lemmou e ID Ransomware, el experimento 2d pendiente, la regla de commits).

Si algo de esto quedó viejo respecto de `ESTADO_TESIS.md`, manda `ESTADO_TESIS.md`.

> Nota: `CLAUDE.md` (la raíz del repo) NO hay que restaurarlo a mano — viaja con el repo y
> Claude lo carga solo. Esta carpeta cubre lo único que el repo no llevaba: la memoria.
