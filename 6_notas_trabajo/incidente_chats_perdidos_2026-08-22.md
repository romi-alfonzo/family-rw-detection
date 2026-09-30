# Incidente: el sidebar perdió los chats de Claude Code (2026-08-22)

## Qué pasó
El panel de proyectos de la app dejó de listar los chats del proyecto
`family-rw-detection` (= este repo, `C:\Users\Romina\Tesis`, rama `develop`).
**No se perdió ninguna conversación.** Lo que se vació fue el índice del sidebar.

## Dónde vive cada cosa (los dos almacenes se confunden fácil)

| Qué | Ruta | Contenido |
|---|---|---|
| **Transcripts** (el contenido real) | `C:\Users\Romina\.claude\projects\C--Users-Romina-Tesis\*.jsonl` | 16 archivos, ~79 MB. **Nunca se perdieron.** |
| **Índice del sidebar** (títulos) | `%LOCALAPPDATA%\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming\Claude\claude-code-sessions\9e93ade2-…\b53ed9ac-…\local_*.json` | Se vació: quedó 1 de 13. |
| Chats de la app Claude (no Code) | `…\Roaming\Claude\local-agent-mode-sessions\` | 22 archivos, intactos. No es lo que faltaba. |

La app es MSIX: lo que escribe en `%APPDATA%` se redirige a
`%LOCALAPPDATA%\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming`. Por eso los
dos árboles se ven duplicados.

## Cómo se recuperó
Los 12 `local_*.json` del índice estaban en el backup de escritorio
(`ClaudeBackup`, hecho a las 00:21). Se copiaron de vuelta al directorio vivo con
`cp -n` (aditivo, sin pisar el de la sesión en curso) y se reinició la app.

Copia limpia del índice, para reintentar sin depender de `ClaudeBackup`:
`C:\Users\Romina\Desktop\RESPALDO_CHATS_TESIS\indice_claude_code\` (12 archivos)
Copia de los transcripts: `C:\Users\Romina\Desktop\RESPALDO_CHATS_TESIS\` (16 `.jsonl`)

## Causa raíz (del ClaudeSetup.log y el Visor de eventos)

1. `CoworkVMService` es un servicio **propiedad del paquete MSIX**
   (`WIN32_PACKAGED_PROCESS`, binario en `C:\Program Files\WindowsApps\Claude_...`,
   depende de `staterepository`). No se borra a mano: por eso el instalador falló con
   *Acceso denegado* al querer removerlo, y por eso también fallaron los `sc.exe
   delete` y `sc.exe sdset` hechos a mano.
2. El instalador intentó una desinstalación **preservando datos** y Windows la rechazó
   (`0x80073CFA, requires developer mode`). Cayó al plan B: *in-place update*.
3. La pasada de 23:46 funcionó. Las de 00:03 y 00:05 fallaron con `0x80073CF9`
   (`AddPackage failed`) porque **el instalador corrió con la app abierta**: el log solo
   verifica procesos «Squirrel» (el instalador viejo), no la app MSIX en ejecución.
4. Visor de eventos: el servicio se instaló/deshabilitó **6+ veces entre 23:43 y 00:26**,
   con `El servicio Claude se terminó de manera inesperada` a las 00:10:35.
5. El último re-registro fue **00:26:20**, y las carpetas `claude-code-sessions`,
   `local-agent-mode-sessions` y `claude-code` tienen fecha de modificación **00:26**.
   Ahí se reseteó el `LocalCache` del paquete. La medición propia lo capturó en vivo:
   `ORIGEN: 0 archivos, 0 GB`.
6. Los transcripts sobrevivieron porque viven en `C:\Users\Romina\.claude\`, **fuera**
   del sandbox del paquete. Se borró solo lo que estaba adentro.

`local-agent-mode-sessions` también se vació; se restauró a mano sin saberlo (el
`Copy-Item` que devolvió `Count: 22` fue correcto).

## Estado verificado el 2026-08-22 (~00:40)
- Payload íntegro: `app\claude.exe` 211,73 MB, 2.801 archivos, el manifiesto declara
  `Executable=app\Claude.exe` y ese archivo existe.
- `Get-AppxPackage`: `Status: Ok`. Servicio: existe, `STOPPED`, `AUTO_START`.
- Disco: 54 GB libres. No es falta de espacio.
- Carpeta de logs de la app: vacía (nunca arrancó desde el reset).
- El `sc.exe sdset` **no se aplicó**: el DACL sigue idéntico al original.
- Aun así la app no abre («Hay un problema con Claude»). Payload íntegro + `Status: Ok`
  + no abre apunta a registro del paquete / *state repository* inconsistente tras los
  re-despliegues, no a archivos faltantes.

## Regla para adelante
**Cerrar la app antes de correr el instalador**, y si falla no reintentar en el momento:
reiniciar la PC primero (libera el payload de `WindowsApps`, detiene el servicio
empaquetado y completa las operaciones de paquete pendientes).

## Callejones sin salida (no repetir)
- `sc.exe stop/delete CoworkVMService` → *Acceso denegado*. **Ahí no hay chats.**
  Pelear con permisos de servicios de Windows no recupera nada.
- Copiar los 11,6 GB enteros del paquete. El índice que hacía falta pesa ~2 MB.
- Buscar `projects` o `.claude` dentro del paquete MSIX: no están ahí, están en
  `C:\Users\Romina\.claude\`.

## Chats recuperados
Re-medir B.3 sobre corpus 155 · Carpeta recordada · Paso 1: remedir corpus 155
notas · Revisión de notas en familias · Auditoría CSV y detector estructural ·
Buenas sigamos · Resultados del servidor · ss · Revisar salidas cluster tesis ·
Correr M.1: cascada IOC→texto en notas · y los dos de `elipse_backend`.
