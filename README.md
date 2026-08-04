# Worker RunPod Serverless — VoxCPM2 / nanovllm

Endpoint GPU per la sintesi VoxCPM2, pensato per affiancare le voci premium
di [audiobook-maker](https://github.com/gfrangiamone/audiobook-maker).

## Stato: verificato su GPU

Girato end-to-end su RTX 4090 il 2026-08-04. Numeri misurati, non stimati:

| | Valore |
|---|---|
| Caricamento modello + warmup `torch.compile` | 37,3 s |
| Cold start completo (pull immagine + avvio + build) | ~180 s |
| `sample_rate` | 48000 |
| Throughput, 11 919 char / 51 chunk / concorrenza 16 | **28,5x realtime** |
| Throughput, 749 char / 3 chunk | 16,2x realtime |
| Chunk falliti | 0 |
| Costo su RTX 4090 a $1.10/h | **~$0,91 / 1M char** |

I 57-92x realtime ipotizzati in fase di stima **non** sono stati raggiunti. Il
conto economico regge lo stesso (Speechify sta a $11,18 / 1M char), ma sotto le
~10-15 richieste in volo il batching continuo non si satura: capitoli corti
costano proporzionalmente di piu'.

Tre difetti trovati e corretti solo grazie all'esecuzione reale, tutti fatali e
nessuno visibile in locale:

1. **`spawn` re-importa `__main__`** — il pool avvia i propri server con
   `multiprocessing`, il figlio rieseguiva `runpod.serverless.start()` e
   diventava un secondo worker nello stesso container. Il processo moriva in
   silenzio fra i 50 e i 90 secondi. Da qui il guard `if __name__ ==
   "__main__"` in fondo a `handler.py`, che non e' una formalita'.
2. **`model_path` vuole una directory, non un repo id** —
   `VoxCPM2ServerImpl` fa `os.path.join(model_path, "config.json")`.
3. **Servono `gcc`/`g++` nell'immagine** — il motore usa `torch.compile` e
   Inductor compila i kernel Triton a runtime, al warmup.

Restano non verificati: il comportamento su tier GPU diversi dalla 4090 e la
fatturazione del container disk a endpoint fermo.

## Build e push: da GitHub Actions, non in locale

L'immagine supera i 10 GB. Costruita in CI il push resta dentro il datacenter
GitHub; da una postazione di lavoro sarebbe un upload di ore, oltre
all'installazione di Docker.

Workflow: `.github/workflows/voxcpm-image.yml`. Si attiva a mano (Actions >
*VoxCPM worker image* > Run workflow) oppure a ogni push su `main`.

Pubblica su `ghcr.io/<owner>/abm-voxcpm:latest` e `:<sha>`.

**Passo manuale una tantum:** al primo push il package GHCR nasce **privato**.
Va reso pubblico da *Packages > abm-voxcpm > Package settings > Change
visibility*, altrimenti RunPod non riesce a scaricarlo (in alternativa si
configurano le credenziali registry nelle impostazioni RunPod).

## Creazione dell'endpoint

Su https://console.runpod.io/serverless > *New Endpoint*:

| Campo | Valore |
|---|---|
| Container image | `ghcr.io/<owner>/abm-voxcpm:latest` |
| GPU | 24 GB (RTX 4090) |
| Worker type | Flex |
| Active workers | 0 |
| Max workers | 1-2 |
| Container disk | 25 GB (i pesi sono nell'immagine) |
| Idle timeout | 5 s |
| FlashBoot | attivo |
| Execution timeout | almeno 600 s |

Il primo controllo dopo la creazione e' `{"input": {"action": "health"}}`: e'
la via piu' economica per sapere se il pool si costruisce davvero sulla GPU.

Con 2 libri al giorno ogni invocazione e' un cold start: FlashBoot aiuta solo
se un worker era attivo di recente. E' latenza, non costo — l'inizializzazione
fatturata vale 0,02-0,06 USD. Irrilevante nella consegna batch via email che
l'app gia' implementa; percepibile in interattivo.

## Chiamata

```
POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync
Authorization: Bearer <RUNPOD_API_KEY>
```

`/runsync` attende la risposta (max ~5 minuti). Per un capitolo lungo usare
`/run`, che restituisce un id da interrogare su `/status/<id>`.

## Variabili d'ambiente

| Variabile | Default | Note |
|---|---|---|
| `ABM_VOXCPM_MODEL` | `openbmb/VoxCPM2` | pesi cotti nell'immagine |
| `ABM_VOXCPM_TIMESTEPS` | `10` | **costruttore**: un endpoint = una qualita' |
| `ABM_VOXCPM_CONCURRENCY` | `16` | default se il job non lo specifica |
| `ABM_VOXCPM_CFG` | `2.0` | aderenza al testo/riferimento |
| `ABM_VOXCPM_MAX_INLINE_BYTES` | `6000000` | ~60 s di audio; oltre serve S3 |
| `ABM_VOXCPM_S3_PREFIX` | `voxcpm` | prefisso delle chiavi generate dal worker |
| `ABM_VOXCPM_S3_PUT_ATTEMPTS` | `3` | tentativi sulla presigned PUT (mai sui 4xx) |
| `ABM_S3_PRESIGN_TTL_SEC` | `21600` | durata della presigned GET restituita |
| `ABM_S3_ENDPOINT` / `ABM_S3_BUCKET` / `ABM_S3_ACCESS_KEY` / `ABM_S3_SECRET_KEY` / `ABM_S3_REGION` / `ABM_S3_KEY_PREFIX` | — | riusabili quelli del cold storage R2. `ABM_S3_REGION` = `auto` per R2 |

## Unita' di lavoro: un job = un capitolo

Un libro intero sono 10-45 minuti di GPU: troppo per un singolo job
serverless, e un'interruzione del worker a meta' butterebbe via tutto. I
capitoli sono indipendenti e si ritentano singolarmente.

Dentro il capitolo i chunk sono sottomessi **in concorrenza**. Non e' tuning:
in sequenza si misura ~10xRT, i 57-92xRT su cui regge il conto economico
arrivano solo saturando il batching continuo. E' lecito perche' ogni chunk
dipende solo dal campione di riferimento — vedi la nota sul merge piu' sotto.

## Voce di riferimento

Due modi, entrambi accettati.

**1. Campione nella chiamata** (uso corrente): si passa `reference_wav_b64`,
il worker lo codifica al volo. I latenti restano in cache per hash del
campione, quindi su worker caldo lo stesso WAV non viene ricodificato — la
risposta lo segnala con `voice_from_cache`.

**2. Blob pre-codificato** (libreria di voci futura): `encode_latents()`
restituisce byte serializzabili, quindi una voce si codifica **una volta
sola**.

```json
{"input": {"action": "encode_voice", "wav_b64": "<base64 del campione>",
           "wav_format": "wav"}}
```

Risposta: `{"voice_latents": "<base64>", "bytes": N, "encode_seconds": N}`.
Il blob va conservato nel catalogo voci e passato come `voice_latents`.

**Il campione deve essere pulito.** Senza riferimento VoxCPM2 allucina anche
l'ambiente acustico, non solo il timbro; con un campione registrato male il
fruscio viene clonato insieme alla voce. Verificato: 15 secondi puliti bastano
per un output pulito gia' a 10 timesteps.

**Servono campioni di cui detieni i diritti.** VoxCPM2 non ha voci
preimpostate: esiste solo la clonazione. Clonare una voce di un fornitore
terzo reintrodurrebbe esattamente la dipendenza da cui questo lavoro vuole
uscire.

## Generazione

```json
{"input": {
  "action": "generate",
  "text": "Testo del capitolo, diviso in chunk dal worker.",
  "reference_wav_b64": "<base64 del campione di voce>",
  "cfg": 2.0,
  "max_chars": 300,
  "concurrency": 16,
  "output_format": "wav",
  "s3": {"put_url": "<presigned PUT firmata dal chiamante>"}
}}
```

| Campo | Default | Note |
|---|---|---|
| `text` **o** `chunks` | — | almeno uno |
| `reference_wav_b64` **o** `voice_latents` | — | almeno uno: senza, la richiesta e' rifiutata |
| `reference_format` | `wav` | estensione del campione |
| `prompt_wav_b64` + `prompt_text` | — | modalita' continuation: richiede la trascrizione **esatta** del campione |
| `cfg` | `2.0` | aderenza al testo/riferimento |
| `max_chars` | `300` | usato solo con `text` |
| `concurrency` | `16` | chunk in volo insieme |
| `output_format` | `wav` | `wav` o `pcm` |
| `s3` | — | consegna dell'audio, vedi sotto |

### Consegna dell'audio: inline solo per le prove

Il ritorno inline in `audio_b64` copre **circa 60 secondi** di audio
(`MAX_INLINE_BYTES` / 96 kB/s a 48 kHz mono 16 bit). Un capitolo vero ne fa
dieci-venti volte tanto: serve S3.

| Modo | Campo | Quando |
|---|---|---|
| **presigned PUT** | `s3.put_url` | **produzione.** Il chiamante firma con `storage_backend.presigned_put_url()` e passa solo l'URL. Il worker non riceve nessuna credenziale |
| credenziali | `ABM_S3_*` sull'endpoint, o `s3.{endpoint_url,access_key,secret_key,bucket,region,key_prefix}` | quando serve la chiave generata dal worker o una presigned GET di ritorno |
| inline | nessuno | prove manuali, sotto i 6 MB |

Con `put_url` la risposta contiene `s3.mode = "put_url"` e i byte caricati;
**l'URL firmato non viene rimandato indietro** (lo conosce gia' chi l'ha
generato, e contiene una firma di scrittura).

Con le credenziali la risposta contiene anche `s3.url`, una presigned **GET**
valida `ABM_S3_PRESIGN_TTL_SEC` secondi: senza, il chiamante riceverebbe una
chiave che non puo' leggere. Se `s3.key` manca, il worker la genera come
`<ABM_VOXCPM_S3_PREFIX>/<job_id>.wav` e segnala `s3_key_generated: true`.

Preferire `put_url`: le credenziali passate nel corpo della richiesta restano
memorizzate nel record del job lato RunPod. In nessun caso le credenziali
compaiono nella risposta.

Verifica della configurazione **senza spendere GPU** — put/head/presign/delete
di un oggetto da 16 byte:

```json
{"input": {"action": "diag", "s3_check": true}}
```

`timesteps` **non** e' un parametro di chiamata: e' del costruttore del pool.
Cambiarlo significa un endpoint diverso.

Passare `chunks` gia' divisi e' la strada preferibile in produzione: l'app ha
gia' `split_text_into_chunks()` in `tts_split.py`, validata. Lo split interno
al worker e' un ripiego per l'uso manuale.

Risposta:

```json
{"sample_rate": 48000, "timesteps": 10, "concurrency": 16, "chunks": 51,
 "max_chars": 300, "voice_from_cache": true, "failed_indices": [],
 "chars": 11919, "tts_seconds": 25.78, "audio_seconds": 733.76,
 "throughput_x_realtime": 28.46, "format": "wav", "bytes": 70441004,
 "s3": {"mode": "put_url", "status": 200, "bytes": 70441004}}
```

Prova minima da PowerShell:

```powershell
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("campione.wav"))
$body = @{ input = @{ action = "generate"
                      text = "Prova di sintesi vocale in italiano."
                      reference_wav_b64 = $b64 } } | ConvertTo-Json -Depth 5
$r = Invoke-RestMethod -Method Post -Uri "https://api.runpod.ai/v2/$EID/runsync" `
     -Headers @{ Authorization = "Bearer $KEY" } `
     -ContentType "application/json" -Body $body
[IO.File]::WriteAllBytes("out.wav", [Convert]::FromBase64String($r.output.audio_b64))
```

Un chunk fallito diventa un secondo di silenzio e il suo indice compare in
`failed_indices`: il capitolo resta allineato e il chiamante sa cosa ritentare.

`throughput_x_realtime` e' il numero da guardare — e' cio' che si paga al
secondo. L'RTF per richiesta, con sottomissione concorrente, non e' piu'
interpretabile come costo unitario.

## Il merge NON si usa

`merge_prompt_cache()` del pacchetto ufficiale concatena al riferimento le
feature dei chunk gia' generati. Misurato in ascolto su testo a due chunk,
stesso riferimento, 10 timesteps: **con merge compare un rumore di fondo dal
secondo ~20 che senza merge non esiste**, e la voce resta identica in entrambi
i casi. Le feature generate rientrano nel prompt e ricircolano il proprio
artefatto.

Il motore nanovllm non espone il merge, il che qui e' un vantaggio: rende i
chunk indipendenti e quindi parallelizzabili.

## Costo atteso

Base di calcolo misurata: 17,3 caratteri al secondo di parlato, cioe' ~16,1
ore di audio per milione di caratteri.

| Ipotesi | Throughput | Ore GPU | USD/1M char |
|---|---|---|---|
| ts10, prompt breve, conc. 64 (dichiarato) | 92xRT | 0,18 | 0,19 |
| ts10, clonazione, conc. 64 | 57xRT | 0,28 | 0,31 |
| ts25, clonazione, conc. 64 | ~21xRT | 0,77 | 0,84 |
| worst case x3 di sicurezza | ~7xRT | 2,30 | 2,53 |

A 1,10 USD/h per RTX 4090 flex. Confronto: Speechify costa 11,18 USD/1M char
(`speechify_tts.py:180`). La conclusione regge in ogni scenario, anche
sbagliando di un fattore 8 — l'incertezza sul throughput cambia la latenza, non
la convenienza.

Fuori dal calcolo: cold start (0,02-0,06 USD a invocazione), idle timeout
(5 s di default) e container disk per l'immagine (~1,50-2,00 USD/mese, che e'
il vero pavimento a traffico basso).

## Verifica in locale prima del deploy

Lo stesso percorso e' esercitabile dallo script di bench che vive nel repo
principale (`scripts/tts_supertonic_test.py`, non versionato), che usa la
medesima API e la stessa politica di concorrenza:

```bash
python tts_supertonic_test.py <libro>.abm --chapter 1 \
  --engine nanovllm --reference-wav <campione>.wav \
  --timesteps 10 --concurrency 16 --wav -o out.m4b
```

Richiede comunque una GPU NVIDIA: su CPU esce con un errore esplicito.

## Licenza

Nessuna licenza dichiarata: senza un file `LICENSE` il codice resta "tutti i
diritti riservati". Da allineare a `audiobook-maker` (AGPL-3.0) se questo
worker diventa parte del prodotto.
