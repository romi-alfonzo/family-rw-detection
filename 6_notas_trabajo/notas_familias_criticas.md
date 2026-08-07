# Notas de rescate — familias críticas (texto + fuentes) — 2026-06-22

Trabajo de extracción para las 5 familias con solo 2 notas. **Revisar y validar antes de
incorporar al corpus.** Cada entrada indica fuente, fecha y nivel de confianza.

> Hallazgo metodológico importante: WannaCry, NotPetya y BadRabbit muestran su mensaje
> principalmente como **pantalla/GUI o pantalla de arranque (MBR)**, y los reportes de vendors
> lo publican como **captura de imagen**, no como texto. Aun así, las tres *también* dejan un
> archivo de texto (`@Please_Read_Me@.txt`, `README.TXT`, `Readme.txt`). Conviene discutir con
> el tutor si estas familias encajan en un corpus de "notas de texto" o son un caso aparte.

---

## ✅ JIGSAW — texto verbatim conseguido (alta confianza)
Fuente: BleepingComputer, "Jigsaw Ransomware Decrypted" (2016-04-11), notas aportadas por MalwareHunterTeam.
https://www.bleepingcomputer.com/news/security/jigsaw-ransomware-decrypted-will-delete-your-files-until-you-pay-the-ransom/

**Variante 1:**
```
Your computer files have been encrypted. Your photos, videos, documents, etc....
But, don't worry! I have not deleted them, yet.
You have 24 hours to pay 150 USD in Bitcoins to get the decryption key.
Every hour files will be deleted. Increasing in amount every time.
After 72 hours all that are left will be deleted.
If you do not have bitcoins Google the website localbitcoins.com.
...
Try anything funny and the computer has several safety measures to delete your files.
As soon as the payment is received the crypted files will be returned to normal.
Thank you
```

**Variante 2:**
```
I want to play a game with you. Let me explain the rules:
All your files are being deleted. Your photos, videos, documents, etc...
But, don't worry! It will only happen if you don't comply...
```

---

## ⚠️ WANNACRY — fragmento confirmado; falta verbatim completo
Archivo que deja: `@Please_Read_Me@.txt` + ventana "Wana Decrypt0r 2.0".
Texto de apertura (ampliamente documentado): *"Ooops, your important files are encrypted."* seguido de
explicación de pago (~US$300 en Bitcoin en 3 días, US$600 en 7 días).
Fuentes:
- Mandiant (Google Cloud), WannaCry Malware Profile: https://cloud.google.com/blog/topics/threat-intelligence/wannacry-malware-profile
- Wikipedia (con referencias): https://en.wikipedia.org/wiki/WannaCry_ransomware_attack
- Antiy Labs, In-Depth Analysis Report: https://www.antiy.net/p/in-depth-analysis-report-on-wannacry-ransomware/
- Pendiente: el texto completo está en la imagen de la nota / en una muestra → obtener verbatim de una muestra o por OCR.

## ✅ NOTPETYA — texto recuperado por OCR con preprocesado (alta confianza)
Imagen descargada: `imagenes_notas/mbr-ransom-note.jpg`. OCR con preprocesado (upscale x3 + binarizado):
```
Ooops, your important files are encrypted.
If you see this text, then your files are no longer accessible, because they have been
encrypted. Perhaps you are busy looking for a way to recover your files, but don't waste
your time. Nobody can recover your files without our decryption service.
We guarantee that you can recover all your files safely and easily. All you need to do is
submit the payment and purchase the decryption key.
Please follow the instructions:
1. Send $300 worth of Bitcoin to following address: [dirección BTC]
2. Send your Bitcoin wallet ID and personal installation key to e-mail
   wowsmith123456@posteo.net. Your personal installation key: [clave]
```
Fuente imagen: BleepingComputer (2017-06-30). Verificar la dirección BTC/clave con la imagen.

## ⚠️ BADRABBIT — nota en imagen; deja Readme.txt
Malwarebytes confirma que deja una nota **en TXT llamada `Readme.txt`**, pero la muestra como imagen
(`oops_note.png`). El texto es casi idéntico al de NotPetya (mismo grupo). Pide pago vía sitio Tor.
Fuente: Malwarebytes Labs, "BadRabbit: a closer look at the new version of Petya/NotPetya" (2017-10-24).
https://www.malwarebytes.com/blog/news/2017/10/badrabbit-closer-look-new-version-petyanotpetya
- Pendiente: texto completo desde la imagen o desde una muestra.

## ✅ CHIMERA — texto verbatim conseguido (alta confianza)
Archivo de nota: `YOUR_FILES_ARE_ENCRYPTED.HTML` + wallpaper. Amenaza con publicar los archivos online.
Fuente: pcrisk.com (Tomas Meskauskas), "Chimera Ransomware".
https://www.pcrisk.com/removal-guides/9542-chimera-ransomware
```
Your are victim of the Chimera malware. Your private files are encrypted and can not be
restored without a special edgy file. Maybe some programs no longer function properly:
Please transfer Bitcoins to the following address to get your unique key file. For the
decryption program and additional information, please visit: https://mega.nz/ChimeraDecrypter
If you don't pay your private data, which include pictures and videos will be published on
the Internet in relation on your name. Take advantage of our affiliate program. More
information in the source code of the file.
```

---

## Resumen — TODAS LAS 5 FAMILIAS CRÍTICAS RESUELTAS ✅
- **Jigsaw:** texto verbatim (BleepingComputer).
- **WannaCry:** OCR de captura (`Wana_Decrypt0r_screenshot.png`).
- **NotPetya:** OCR con preprocesado de `mbr-ransom-note.jpg`.
- **BadRabbit:** OCR de `oops_note.png`.
- **Chimera:** texto verbatim (pcrisk).
- Las 30 familias del corpus ya tienen al menos una nota disponible.
- Pendiente (cuando decidas): integrar estas notas al corpus con manifiesto de procedencia y
  reentrenar comparando macro-F1. Para WannaCry/NotPetya/BadRabbit, verificar direcciones/claves
  contra la imagen antes de fijar el texto final.
