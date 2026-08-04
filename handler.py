# -*- coding: utf-8 -*-
"""Worker RunPod Serverless per VoxCPM2 su motore nanovllm-voxcpm.

Gemello del percorso `--engine nanovllm` di scripts/tts_supertonic_test.py:
stessa API, stessa politica di concorrenza, stesse assunzioni sulla voce.
Quello che cambia e' solo il contorno (input JSON, output su S3).

MODELLO DI COSTO
    RunPod fattura tre fasi: inizializzazione del container e caricamento del
    modello in VRAM, esecuzione, e l'idle timeout (5s di default). Il pool
    viene quindi costruito UNA VOLTA a livello di modulo: se il worker resta
    caldo, le invocazioni successive non ripagano il caricamento.

UNITA' DI LAVORO
    Un job = un CAPITOLO, non un libro intero. Un libro sono 10-45 minuti di
    lavoro GPU: troppo per un singolo job serverless, con rischio di perdere
    tutto se il worker viene interrotto a meta'. I capitoli sono indipendenti,
    quindi si parallelizzano e si ritentano singolarmente.

VOCE DI RIFERIMENTO — due modi, entrambi accettati
    1. `reference_wav_b64`: il campione viaggia nella chiamata e viene
       codificato al volo. E' il modo d'uso corrente.
    2. `voice_latents`: blob gia' codificato con action="encode_voice". Costa
       un encode in meno per chiamata ed e' la strada per la futura libreria
       di voci precaricate.
    I latenti sono comunque tenuti in cache per hash del campione: su worker
    caldo lo stesso WAV non viene ricodificato.

CONSEGNA DELL'AUDIO — inline solo per le prove, S3 per i capitoli veri
    Il ritorno inline in base64 copre circa 60 secondi di audio
    (MAX_INLINE_BYTES / 96 kB/s a 48 kHz mono 16 bit): un capitolo vero ne fa
    dieci-venti volte tanto. Sopra la soglia l'audio va su S3/R2 e nella
    risposta torna un URL.

    Due modi, in ordine di preferenza:
    1. `s3.put_url` — il CHIAMANTE firma una presigned PUT (l'app ha gia'
       storage_backend.presigned_put_url) e passa solo quell'URL. Il worker
       non riceve alcuna credenziale: se il payload del job venisse letto da
       terzi, esporrebbe un permesso di scrittura su una singola chiave e a
       scadenza, non le chiavi del bucket. E' il modo con cui deve girare la
       produzione.
    2. Credenziali complete, dalle ABM_S3_* dell'ENDPOINT o — sconsigliato —
       dai campi della richiesta. Serve quando si vuole la chiave generata dal
       worker o una presigned GET di ritorno. Le credenziali nel corpo della
       richiesta restano memorizzate nel record del job lato RunPod: usare le
       variabili d'ambiente dell'endpoint.
    In nessun caso le credenziali vengono rimandate indietro nella risposta.
"""
import asyncio
import base64
import contextlib
import hashlib
import io
import os
import re
import shutil
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
import wave

import numpy as np
import runpod

# ---------------------------------------------------------------------------
# Configurazione (via variabili d'ambiente dell'endpoint RunPod)
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("ABM_VOXCPM_MODEL", "openbmb/VoxCPM2")
# I timesteps sono un parametro del COSTRUTTORE: non si possono cambiare per
# singola richiesta. Un endpoint = una configurazione di qualita'.
TIMESTEPS = int(os.environ.get("ABM_VOXCPM_TIMESTEPS", "10"))
MAX_NUM_SEQS = int(os.environ.get("ABM_VOXCPM_MAX_NUM_SEQS", "512"))
GPU_MEM_UTIL = float(os.environ.get("ABM_VOXCPM_GPU_MEM_UTIL", "0.9"))
DEFAULT_CONCURRENCY = int(os.environ.get("ABM_VOXCPM_CONCURRENCY", "16"))
DEFAULT_CFG = float(os.environ.get("ABM_VOXCPM_CFG", "2.0"))
SAMPLE_RATE_FALLBACK = 48000
# Oltre questa soglia il PCM non torna nella risposta JSON ma va su S3.
# A 48 kHz mono 16 bit sono 96 kB al secondo: 6 MB fanno ~62 secondi di audio.
MAX_INLINE_BYTES = int(os.environ.get("ABM_VOXCPM_MAX_INLINE_BYTES", "6000000"))
# Prefisso delle chiavi generate dal worker quando il chiamante non ne passa
# una. Namespacing esplicito: il bucket e' condiviso con il tiering dell'app.
S3_AUTO_PREFIX = os.environ.get("ABM_VOXCPM_S3_PREFIX", "voxcpm").strip("/")
# Durata della presigned GET restituita al chiamante. Sei ore: il tempo di
# scaricare un capitolo e ritentare qualche volta, non di condividerlo.
S3_PRESIGN_TTL = int(os.environ.get("ABM_S3_PRESIGN_TTL_SEC", "21600"))
# Tentativi per la PUT su presigned URL. La firma ha una scadenza: ritentare
# a lungo non ha senso, oltre un certo punto l'URL e' morto comunque.
S3_PUT_ATTEMPTS = int(os.environ.get("ABM_VOXCPM_S3_PUT_ATTEMPTS", "3"))
# Limite per chunk. 300 char e' il valore usato nei test di qualita': testi
# piu' lunghi allungano il contesto autoregressivo senza migliorare la resa.
DEFAULT_MAX_CHARS = int(os.environ.get("ABM_VOXCPM_CHUNK_MAX_CHARS", "300"))
# Quante voci tenere in cache sul worker caldo. Ogni voce sono pochi KB di
# latenti: il tetto serve solo a evitare crescita illimitata.
LATENT_CACHE_MAX = int(os.environ.get("ABM_VOXCPM_LATENT_CACHE", "32"))
# Tetto alla costruzione del pool. Serve a trasformare un blocco indefinito in
# un errore leggibile dal chiamante: un job appeso costa GPU fino all'execution
# timeout dell'endpoint e non lascia nulla da leggere.
BUILD_TIMEOUT = float(os.environ.get("ABM_VOXCPM_BUILD_TIMEOUT", "420"))

_POOL = None
_LOOP = None
_LOOP_THREAD = None
_SAMPLE_RATE = SAMPLE_RATE_FALLBACK
_LATENT_CACHE = {}
# L'handler puo' essere invocato da piu' thread se l'endpoint accetta piu' job
# in parallelo: senza lock due chiamate contemporanee creerebbero due loop o
# due pool, e il secondo pool non troverebbe VRAM libera.
# RLock e non Lock: _pool() lo tiene mentre chiama _await(), che a sua volta lo
# riprende per creare il loop — con un Lock semplice sarebbe uno stallo.
_INIT_LOCK = threading.RLock()


def _await(coro):
    """Esegue una coroutine sul loop dedicato e ne attende l'esito.

    Il worker RunPod invoca l'handler DENTRO un event loop gia' in esecuzione:
    un run_until_complete qui dentro solleva "Cannot run the event loop while
    another loop is running" — misurato sul campo, non teorico.

    Il loop vive quindi in un thread suo e resta acceso fra una chiamata e
    l'altra. Non e' solo un rimedio all'errore: il pool registra i propri task
    di background (lo scheduler del batching continuo) sul loop in cui nasce,
    e con run_until_complete quel loop girerebbe solo durante le singole
    chiamate — cioe' il batching su cui regge tutto il conto economico non
    avrebbe mai modo di lavorare.
    """
    global _LOOP, _LOOP_THREAD
    if _LOOP is None:
        with _INIT_LOCK:
            if _LOOP is None:
                loop = asyncio.new_event_loop()
                _LOOP_THREAD = threading.Thread(
                    target=loop.run_forever, name="voxcpm-loop", daemon=True)
                _LOOP_THREAD.start()
                _LOOP = loop
    return asyncio.run_coroutine_threadsafe(coro, _LOOP).result()


@contextlib.contextmanager
def _heartbeat(label, every=10.0):
    """Stampa una riga ogni `every` secondi finche' il blocco e' attivo.

    Serve a diagnosticare le morti silenziose: se il processo viene ucciso
    senza traccia (OOM killer, abort a livello C), l'ultima riga stampata dice
    a che secondo e' successo. Senza questo, fra l'avvio del worker e la morte
    non c'e' NULLA da leggere nei log.
    """
    stop = threading.Event()
    t0 = time.time()

    def _tick():
        while not stop.wait(every):
            print(f"[worker] {label}: {time.time() - t0:.0f}s in corso",
                  flush=True)

    th = threading.Thread(target=_tick, name=f"hb-{label}", daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        print(f"[worker] {label}: uscita dopo {time.time() - t0:.1f}s",
              flush=True)


def _resolve_model_path(spec):
    """Da repo id HuggingFace a directory locale.

    VoxCPM2ServerImpl fa os.path.join(model_path, "config.json"): vuole una
    CARTELLA, non un repo id. Passando "openbmb/VoxCPM2" cercava
    "openbmb/VoxCPM2/config.json" relativo alla cwd — misurato, FileNotFoundError.

    I pesi sono gia' nell'immagine (snapshot_download in fase di build), quindi
    si prova prima local_files_only: senza, un errore di cache si tradurrebbe in
    5 GB scaricati a runtime, pagati a GPU accesa.
    """
    if os.path.isdir(spec):
        return spec
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(spec, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        print(f"[worker] pesi non in cache ({type(e).__name__}), scarico da HF",
              flush=True)
        return snapshot_download(spec)


async def _build_pool():
    """Costruisce il pool DENTRO il loop dedicato.

    La costruzione avviene qui e non nel chiamante perche' il pool registra i
    propri task di background sul loop attivo al momento della creazione: se
    nascesse sul loop dell'handler, quei task morirebbero con la chiamata.
    """
    model_dir = _resolve_model_path(MODEL_PATH)
    print(f"[worker] pesi in {model_dir}", flush=True)
    # Nessun ripiego su VoxCPM.from_pretrained: la presenza di
    # AsyncVoxCPM2ServerPool e' verificata sul worker (nanovllm-voxcpm 2.0.3),
    # e il wrapper sincrono romperebbe comunque piu' avanti — encode_latents e
    # generate sono usati qui come API asincrone. Meglio un ImportError netto.
    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2ServerPool
    pool = AsyncVoxCPM2ServerPool(
        model_path=model_dir,
        inference_timesteps=TIMESTEPS,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=GPU_MEM_UTIL,
        devices=[0],
    )
    await pool.wait_for_ready()
    return pool


def _pool():
    """Costruisce il pool al primo uso e lo riusa (worker caldo)."""
    global _POOL, _SAMPLE_RATE
    if _POOL is not None:
        return _POOL
    with _INIT_LOCK:
        if _POOL is not None:
            return _POOL
        t0 = time.time()
        print(f"[worker] costruzione pool: model={MODEL_PATH} "
              f"timesteps={TIMESTEPS} max_num_seqs={MAX_NUM_SEQS} "
              f"gpu_mem_util={GPU_MEM_UTIL}", flush=True)
        with _heartbeat("build"):
            # wait_for e non attesa nuda: senza tetto un blocco nel caricamento
            # del modello resta appeso fino all'execution timeout dell'endpoint,
            # e il chiamante riceve un timeout generico invece di un errore.
            pool = _await(asyncio.wait_for(_build_pool(), timeout=BUILD_TIMEOUT))
        try:
            info = _await(pool.get_model_info())
            _SAMPLE_RATE = int(getattr(info, "sample_rate", SAMPLE_RATE_FALLBACK))
        except Exception:  # noqa: BLE001 — non bloccante
            _SAMPLE_RATE = SAMPLE_RATE_FALLBACK
        _POOL = pool
    print(f"[worker] pool pronto in {time.time() - t0:.1f}s  "
          f"timesteps={TIMESTEPS}  sample_rate={_SAMPLE_RATE}", flush=True)
    return _POOL


def _to_array(piece):
    """Normalizza un pezzo di forma d'onda a float32 monodimensionale."""
    for attr in ("wav", "waveform", "audio", "data"):
        if hasattr(piece, attr):
            piece = getattr(piece, attr)
            break
    if hasattr(piece, "cpu"):
        piece = piece.cpu().numpy()
    return np.asarray(piece, dtype=np.float32).reshape(-1)


def _to_pcm16(arr):
    """float32 [-1,1] -> PCM s16le. Il clip evita il wrap-around su overflow."""
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _to_wav(pcm, sample_rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Consegna su S3/R2
# ---------------------------------------------------------------------------
_S3_CLIENT = None
_S3_CLIENT_IDENT = None
_S3_LOCK = threading.Lock()


def _s3_conf(cfg):
    """Configurazione S3: prima i campi della richiesta, poi le ABM_S3_*.

    L'ordine e' quello e non il contrario perche' un job deve poter puntare a
    un bucket diverso da quello di default dell'endpoint senza riconfigurarlo.
    """
    cfg = cfg or {}

    def pick(field, env, default=""):
        v = cfg.get(field)
        if v is None or v == "":
            v = os.environ.get(env, default)
        return str(v).strip()

    return {
        "endpoint_url": pick("endpoint_url", "ABM_S3_ENDPOINT"),
        "access_key": pick("access_key", "ABM_S3_ACCESS_KEY"),
        "secret_key": pick("secret_key", "ABM_S3_SECRET_KEY"),
        "bucket": pick("bucket", "ABM_S3_BUCKET"),
        # "auto" e' il valore che vuole Cloudflare R2; per gli altri provider
        # S3-compatibili la regione va passata esplicitamente.
        "region": pick("region", "ABM_S3_REGION", "auto") or "auto",
        "prefix": pick("key_prefix", "ABM_S3_KEY_PREFIX").strip("/"),
        "ttl": int(cfg.get("presign_ttl") or S3_PRESIGN_TTL),
    }


def _s3_ready(conf):
    return all(conf[k] for k in
               ("endpoint_url", "access_key", "secret_key", "bucket"))


def _s3_missing(conf):
    return [k for k in ("endpoint_url", "access_key", "secret_key", "bucket")
            if not conf[k]]


def _s3_client(conf):
    """Client boto3 riusato fra le chiamate, ricostruito solo se cambia il target.

    Costruirlo a ogni upload significa ririsolvere l'endpoint e ricreare il
    pool di connessioni: su un capitolo da decine di MB e' spreco puro.
    signature_version s3v4 esplicito perche' R2 accetta solo quella.
    """
    global _S3_CLIENT, _S3_CLIENT_IDENT
    ident = (conf["endpoint_url"], conf["access_key"], conf["bucket"],
             conf["region"])
    with _S3_LOCK:
        if _S3_CLIENT is None or _S3_CLIENT_IDENT != ident:
            import boto3
            from botocore.config import Config
            _S3_CLIENT = boto3.client(
                "s3",
                endpoint_url=conf["endpoint_url"],
                aws_access_key_id=conf["access_key"],
                aws_secret_access_key=conf["secret_key"],
                region_name=conf["region"],
                config=Config(signature_version="s3v4",
                              retries={"max_attempts": 3, "mode": "standard"}),
            )
            _S3_CLIENT_IDENT = ident
    return _S3_CLIENT


def _s3_full_key(conf, key):
    k = key.lstrip("/")
    return f"{conf['prefix']}/{k}" if conf["prefix"] else k


def _s3_put_signed(put_url, data, content_type):
    """Carica su una presigned PUT firmata dal chiamante. Nessuna credenziale.

    Ritentato perche' un capitolo sono decine di MB su una connessione che puo'
    cadere, e rigenerare l'audio costa GPU mentre ritentare la PUT costa banda.
    Non si ritenta sui 4xx: una firma scaduta o una chiave sbagliata non
    diventano valide riprovando.
    """
    last = None
    for attempt in range(S3_PUT_ATTEMPTS):
        try:
            req = urllib.request.Request(
                put_url, data=data, method="PUT",
                headers={"Content-Type": content_type,
                         "Content-Length": str(len(data))})
            with urllib.request.urlopen(req, timeout=300) as r:
                return {"status": r.status, "bytes": len(data)}
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read()[:500].decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                pass
            if 400 <= e.code < 500:
                raise RuntimeError(
                    f"presigned PUT rifiutata con {e.code}: {body}") from e
            last = f"HTTP {e.code}: {body}"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        print(f"[worker] PUT tentativo {attempt + 1} fallito ({last})",
              flush=True)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"presigned PUT fallita dopo {S3_PUT_ATTEMPTS} "
                       f"tentativi: {last}")


def _s3_upload(data, conf, key, content_type):
    """Carica con credenziali complete e restituisce una presigned GET.

    upload_fileobj e non put_object: sopra la soglia boto3 passa da solo al
    multipart, che su decine di MB e' ritentabile a pezzi invece che da capo.
    """
    from boto3.s3.transfer import TransferConfig
    client = _s3_client(conf)
    full = _s3_full_key(conf, key)
    t0 = time.time()
    client.upload_fileobj(
        io.BytesIO(data), conf["bucket"], full,
        ExtraArgs={"ContentType": content_type},
        Config=TransferConfig(multipart_threshold=8 * 1024 * 1024,
                              multipart_chunksize=8 * 1024 * 1024,
                              max_concurrency=4),
    )
    out = {"bucket": conf["bucket"], "key": full, "bytes": len(data),
           "upload_seconds": round(time.time() - t0, 2)}
    try:
        # L'URL firmato e' l'unica parte utile al chiamante: senza, riceverebbe
        # una chiave che non puo' leggere. E' a scadenza, e le credenziali non
        # lasciano mai il worker.
        out["url"] = client.generate_presigned_url(
            "get_object", Params={"Bucket": conf["bucket"], "Key": full},
            ExpiresIn=conf["ttl"])
        out["url_expires_in"] = conf["ttl"]
    except Exception as e:  # noqa: BLE001 — l'oggetto e' caricato comunque
        out["presign_error"] = f"{type(e).__name__}: {e}"
    return out


def _deliver_audio(out, payload, inp, job_id, ext, content_type):
    """Decide come consegnare l'audio e annota l'esito in `out`.

    Regola: se il chiamante indica una destinazione la si usa sempre, anche per
    audio piccolo; altrimenti si va inline finche' ci sta e si passa a S3 solo
    quando serve. Cosi' le prove manuali restano a una sola chiamata e i
    capitoli veri non sbattono contro il limite.

    Attenzione: RunPod marca FAILED qualunque job la cui uscita contenga la
    chiave `error`. Va scritta solo quando l'audio non e' stato consegnato.
    """
    s3cfg = inp.get("s3") or {}
    put_url = s3cfg.get("put_url")
    key = s3cfg.get("key")
    conf = _s3_conf(s3cfg)
    too_big = len(payload) > MAX_INLINE_BYTES

    if put_url:
        try:
            res = _s3_put_signed(put_url, payload, content_type)
            # L'URL firmato NON viene rimandato indietro: contiene la firma di
            # scrittura ed e' gia' noto a chi l'ha generato.
            out["s3"] = dict(res, mode="put_url", key=key or None)
        except Exception as e:  # noqa: BLE001
            out["error"] = f"upload su presigned PUT fallito: {e}"
        return

    if key or (too_big and _s3_ready(conf)):
        if not _s3_ready(conf):
            out["error"] = (
                "'s3.key' richiesto ma configurazione S3 incompleta, mancano: "
                + ", ".join(_s3_missing(conf))
                + ". Impostare le ABM_S3_* sull'endpoint oppure passare "
                  "'s3.put_url' (presigned PUT, nessuna credenziale)")
            return
        if not key:
            # Chiave sull'id del job: un job = un capitolo, quindi e' univoca
            # e permette di risalire dal file al job che l'ha prodotto.
            key = f"{S3_AUTO_PREFIX}/{job_id or uuid.uuid4().hex}.{ext}"
            out["s3_key_generated"] = True
        try:
            out["s3"] = dict(_s3_upload(payload, conf, key, content_type),
                             mode="credentials")
        except Exception as e:  # noqa: BLE001
            out["error"] = f"upload S3 fallito: {type(e).__name__}: {e}"
        return

    if not too_big:
        out["audio_b64"] = base64.b64encode(payload).decode("ascii")
        return

    out["error"] = (
        f"audio di {len(payload):,} byte oltre il limite inline di "
        f"{MAX_INLINE_BYTES:,} e nessuna destinazione S3 configurata: passare "
        f"'s3.put_url' (presigned PUT) oppure impostare le ABM_S3_* "
        f"sull'endpoint")


_SENTENCE_END = re.compile(r"(?<=[.!?;:…])\s+|\n+")


def _split_text(text, max_chars):
    """Divide il testo in chunk <= max_chars rispettando i confini di frase.

    ATTENZIONE: e' un ripiego per usare l'endpoint con testo grezzo (curl,
    prove manuali). L'implementazione di riferimento resta
    `split_text_into_chunks()` di tts_split.py, che l'app gia' usa ed e'
    validata: il percorso di produzione dovrebbe passare `chunks` gia' divisi
    da li'. Duplicare quella logica qui significherebbe farla divergere.
    """
    chunks, cur = [], ""
    for piece in _SENTENCE_END.split(text.strip()):
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) > max_chars:
            # Frase piu' lunga del limite: si spezza sugli spazi, l'unico
            # punto di taglio che non tronca una parola a meta'.
            if cur:
                chunks.append(cur)
                cur = ""
            for word in piece.split(" "):
                # Una singola "parola" puo' superare il limite: URL lunghi, ma
                # soprattutto cinese e giapponese, che non separano le parole
                # con spazi — li' l'intero capitolo sarebbe una parola sola.
                # Taglio netto: meglio una cesura che un chunk fuori limite.
                while len(word) > max_chars:
                    if cur:
                        chunks.append(cur)
                        cur = ""
                    chunks.append(word[:max_chars])
                    word = word[max_chars:]
                if len(cur) + len(word) + 1 > max_chars and cur:
                    chunks.append(cur)
                    cur = ""
                cur = f"{cur} {word}".strip()
            continue
        if len(cur) + len(piece) + 1 > max_chars and cur:
            chunks.append(cur)
            cur = ""
        cur = f"{cur} {piece}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def _encode_cached(raw, fmt):
    """Codifica un campione in latenti, riusando la cache del worker caldo."""
    key = hashlib.sha256(raw).hexdigest()
    hit = _LATENT_CACHE.get(key)
    if hit is not None:
        return hit, True
    latents = _await(_pool().encode_latents(raw, fmt))
    if len(_LATENT_CACHE) >= LATENT_CACHE_MAX:
        _LATENT_CACHE.pop(next(iter(_LATENT_CACHE)))
    _LATENT_CACHE[key] = latents
    return latents, False


def _resolve_voice(inp):
    """Ricava i latenti della voce dai campi accettati, in ordine di costo.

    Ritorna (ref_latents, prompt_latents, from_cache) oppure solleva
    ValueError se non e' stato fornito alcun riferimento.
    """
    from_cache = False
    ref = None
    if inp.get("voice_latents"):
        ref = base64.b64decode(inp["voice_latents"])
    elif inp.get("reference_wav_b64"):
        ref, from_cache = _encode_cached(
            base64.b64decode(inp["reference_wav_b64"]),
            inp.get("reference_format", "wav"))

    prompt = None
    if inp.get("prompt_latents"):
        prompt = base64.b64decode(inp["prompt_latents"])
    elif inp.get("prompt_wav_b64"):
        prompt, _ = _encode_cached(
            base64.b64decode(inp["prompt_wav_b64"]),
            inp.get("prompt_format", "wav"))

    if ref is None and prompt is None:
        # Senza riferimento VoxCPM2 inventa una voce diversa a ogni chunk: il
        # capitolo suonerebbe letto da persone diverse. Meglio fallire subito.
        raise ValueError(
            "nessuna voce di riferimento: fornire 'reference_wav_b64' "
            "(campione WAV in base64) oppure 'voice_latents' (blob da "
            "encode_voice). Senza, la voce non e' stabile fra i chunk")
    return ref, prompt, from_cache


# ---------------------------------------------------------------------------
# Azioni
# ---------------------------------------------------------------------------
def _action_encode_voice(inp):
    """Codifica un campione in latenti riusabili: si fa UNA VOLTA per voce."""
    raw = base64.b64decode(inp["wav_b64"])
    fmt = inp.get("wav_format", "wav")
    t0 = time.time()
    latents = _await(_pool().encode_latents(raw, fmt))
    return {
        "voice_latents": base64.b64encode(latents).decode("ascii"),
        "bytes": len(latents),
        "encode_seconds": round(time.time() - t0, 3),
    }


def _action_generate(inp, job_id=""):
    max_chars = int(inp.get("max_chars", DEFAULT_MAX_CHARS))
    chunks = inp.get("chunks") or []
    if not chunks:
        # Testo grezzo: diviso qui. Vedi la nota su _split_text.
        chunks = _split_text(inp.get("text") or "", max_chars)
    chunks = [c for c in chunks if c and c.strip()]
    if not chunks:
        return {"error": "nessun testo da sintetizzare: fornire 'text' "
                         "oppure 'chunks'"}

    pool = _pool()
    try:
        ref_latents, prompt_latents, voice_cached = _resolve_voice(inp)
    except ValueError as e:
        return {"error": str(e)}

    prompt_text = inp.get("prompt_text") or ""
    cfg = float(inp.get("cfg", DEFAULT_CFG))
    concurrency = int(inp.get("concurrency", DEFAULT_CONCURRENCY))

    results = [np.zeros(0, dtype=np.float32) for _ in chunks]
    failed = []

    async def _one(i, text, sem):
        async with sem:
            try:
                pieces = []
                async for piece in pool.generate(
                        target_text=text,
                        prompt_latents=prompt_latents,
                        prompt_text=prompt_text,
                        ref_audio_latents=ref_latents,
                        cfg_value=cfg):
                    pieces.append(_to_array(piece))
                if not pieces:
                    raise RuntimeError("nessun campione audio restituito")
                results[i] = np.concatenate(pieces, axis=0)
            except Exception as e:  # noqa: BLE001
                # Silenzio al posto del chunk: il capitolo resta allineato e
                # il chiamante sa quali indici ritentare.
                print(f"[worker] chunk {i} FALLITO: {e}", flush=True)
                results[i] = np.zeros(_SAMPLE_RATE, dtype=np.float32)
                failed.append(i)

    async def _run():
        sem = asyncio.Semaphore(concurrency)
        await asyncio.gather(*(_one(i, c, sem) for i, c in enumerate(chunks)))

    t0 = time.time()
    _await(_run())
    tts_seconds = time.time() - t0

    pcm = _to_pcm16(np.concatenate(results, axis=0))
    audio_seconds = len(pcm) / 2.0 / _SAMPLE_RATE
    chars = sum(len(c) for c in chunks)

    out = {
        "sample_rate": _SAMPLE_RATE,
        "timesteps": TIMESTEPS,
        "concurrency": concurrency,
        "chunks": len(chunks),
        "max_chars": max_chars,
        "voice_from_cache": voice_cached,
        "failed_indices": sorted(failed),
        "chars": chars,
        "tts_seconds": round(tts_seconds, 2),
        "audio_seconds": round(audio_seconds, 2),
        # Throughput aggregato: e' cio' che si paga, non l'RTF per richiesta.
        "throughput_x_realtime": round(audio_seconds / tts_seconds, 2)
        if tts_seconds else 0.0,
    }

    fmt = (inp.get("output_format") or "wav").lower()
    if fmt == "wav":
        payload, ext, ctype = _to_wav(pcm, _SAMPLE_RATE), "wav", "audio/wav"
    else:
        payload, ext, ctype = pcm, "pcm", "application/octet-stream"
    out["format"] = fmt
    out["bytes"] = len(payload)

    _deliver_audio(out, payload, inp, job_id, ext, ctype)
    return out


def _stage(out, name, fn):
    """Esegue una sonda diagnostica isolata e ne registra l'esito.

    Ogni sonda e' cronometrata e racchiusa nel proprio try/except: una che
    fallisce non impedisce alle successive di girare, e stdout/stderr prodotti
    dalle librerie (che nella console RunPod non si vedono) finiscono nella
    risposta JSON. E' l'unico canale diagnostico affidabile che abbiamo.
    """
    buf = io.StringIO()
    t0 = time.time()
    entry = {}
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            entry["result"] = fn()
        entry["ok"] = True
    except BaseException as e:  # noqa: BLE001 — anche SystemExit va riportato
        entry["ok"] = False
        entry["error"] = f"{type(e).__name__}: {e}"
        entry["traceback"] = traceback.format_exc()[-4000:]
    entry["seconds"] = round(time.time() - t0, 2)
    captured = buf.getvalue()
    if captured:
        entry["stdout"] = captured[-4000:]
    out[name] = entry
    # Ristampato sul vero stdout: se il processo muore alla sonda successiva,
    # nei log resta almeno il punto raggiunto.
    print(f"[worker] diag/{name}: ok={entry['ok']} "
          f"{entry['seconds']}s", flush=True)
    return entry


def _probe_env():
    import platform
    du = shutil.disk_usage("/")
    mem = {}
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("MemTotal", "MemAvailable"):
                    mem[k] = v.strip()
    except OSError:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "disk_free_gb": round(du.free / 1e9, 2),
        "disk_total_gb": round(du.total / 1e9, 2),
        "meminfo": mem,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        # Solo ABM_VOXCPM_* e solo se non somigliano a un segreto: la risposta
        # viaggia fino al chiamante e finisce nei suoi log. Un "dammi tutte le
        # ABM_*" qui esporrebbe chiavi API e password SMTP.
        "abm_env": {k: v for k, v in sorted(os.environ.items())
                    if k.startswith("ABM_VOXCPM_")
                    and not re.search(r"KEY|SECRET|PASS|TOKEN", k)},
    }


def _probe_torch():
    import torch
    info = {
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "is_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if info["device_count"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        info["vram_free_gb"] = round(free / 1e9, 2)
        info["vram_total_gb"] = round(total / 1e9, 2)
        # Allocazione reale: is_available() puo' dire True e poi fallire
        # all'inizializzazione del contesto CUDA.
        t = torch.zeros(1024, 1024, device="cuda")
        info["alloc_test"] = float(t.sum().item())
    return info


def _probe_flash_attn():
    import flash_attn
    return {"version": getattr(flash_attn, "__version__", "?"),
            "file": getattr(flash_attn, "__file__", "?")}


def _probe_nanovllm():
    import nanovllm_voxcpm
    info = {
        "version": getattr(nanovllm_voxcpm, "__version__", "?"),
        "file": getattr(nanovllm_voxcpm, "__file__", "?"),
        "has_VoxCPM": hasattr(nanovllm_voxcpm, "VoxCPM"),
    }
    try:
        from nanovllm_voxcpm.models.voxcpm2 import server as _srv
        info["server_module"] = getattr(_srv, "__file__", "?")
        info["exports"] = [n for n in dir(_srv) if not n.startswith("_")]
    except Exception as e:  # noqa: BLE001
        info["server_module_error"] = f"{type(e).__name__}: {e}"
    return info


def _probe_model_path():
    """Risolve i pesi senza costruire nulla: e' l'errore che ha fermato il build."""
    d = _resolve_model_path(MODEL_PATH)
    return {"spec": MODEL_PATH, "dir": d, "is_dir": os.path.isdir(d),
            "has_config": os.path.isfile(os.path.join(d, "config.json")),
            "entries": sorted(os.listdir(d))[:40] if os.path.isdir(d) else []}


def _probe_weights():
    root = os.environ.get("HF_HOME", "/opt/hf")
    found = []
    for base, _sub, files in os.walk(root):
        for fn in files:
            p = os.path.join(base, fn)
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            if sz > 50_000_000:
                found.append({"path": p, "gb": round(sz / 1e9, 2)})
    return {"hf_home": root, "big_files": sorted(
        found, key=lambda d: -d["gb"])[:20]}


def _probe_s3(cfg):
    """Giro completo put/head/presign/delete su un oggetto minuscolo.

    Verificare le credenziali qui costa una frazione di secondo; scoprirle
    sbagliate dopo aver sintetizzato un capitolo costa minuti di GPU e l'audio,
    che a quel punto e' gia' stato prodotto e viene buttato.
    Non riporta MAI access_key/secret_key: la risposta torna al chiamante.
    """
    conf = _s3_conf(cfg)
    info = {"configured": _s3_ready(conf),
            "endpoint_url": conf["endpoint_url"],
            "bucket": conf["bucket"],
            "region": conf["region"],
            "key_prefix": conf["prefix"],
            "auto_prefix": S3_AUTO_PREFIX}
    if not info["configured"]:
        info["missing"] = _s3_missing(conf)
        return info

    client = _s3_client(conf)
    key = _s3_full_key(conf, f"{S3_AUTO_PREFIX}/_probe/{uuid.uuid4().hex}.txt")
    blob = b"abm-voxcpm probe"
    t0 = time.time()
    client.put_object(Bucket=conf["bucket"], Key=key, Body=blob,
                      ContentType="text/plain")
    info["put_seconds"] = round(time.time() - t0, 3)
    head = client.head_object(Bucket=conf["bucket"], Key=key)
    info["head_bytes"] = int(head["ContentLength"])
    info["size_ok"] = info["head_bytes"] == len(blob)
    url = client.generate_presigned_url(
        "get_object", Params={"Bucket": conf["bucket"], "Key": key},
        ExpiresIn=60)
    # Solo la forma dell'URL: il valore intero contiene la firma e non serve
    # a chi legge la diagnostica.
    info["presign_ok"] = url.startswith("http") and "X-Amz-Signature" in url
    try:
        client.delete_object(Bucket=conf["bucket"], Key=key)
        info["cleanup_ok"] = True
    except Exception as e:  # noqa: BLE001 — resta un oggetto da 16 byte
        info["cleanup_ok"] = False
        info["cleanup_error"] = f"{type(e).__name__}: {e}"
    return info


def _action_diag(inp):
    """Sonde a stadi. Diagnostica che torna al chiamante, non nei log.

    La console RunPod non mostra NULLA dell'applicazione: i tre health falliti
    hanno prodotto solo righe dell'infrastruttura. Questa azione porta la stessa
    informazione nel corpo della risposta HTTP, dove e' leggibile.
    """
    stages = {}
    _stage(stages, "env", _probe_env)
    _stage(stages, "torch", _probe_torch)
    _stage(stages, "flash_attn", _probe_flash_attn)
    _stage(stages, "nanovllm", _probe_nanovllm)
    _stage(stages, "model_path", _probe_model_path)
    if inp.get("weights"):
        _stage(stages, "weights", _probe_weights)
    # Esplicito e non automatico: scrive davvero sul bucket, e una diagnostica
    # non deve avere effetti collaterali che il chiamante non ha chiesto.
    if inp.get("s3_check"):
        _stage(stages, "s3", lambda: _probe_s3(inp.get("s3")))
    # Lo stadio caro e' esplicito: le sonde sopra costano secondi, questo
    # costa minuti di GPU e puo' far morire il processo.
    if inp.get("build"):
        _stage(stages, "build", lambda: {
            "sample_rate": (_pool(), _SAMPLE_RATE)[1]})
    return {"ok": all(s.get("ok") for s in stages.values()),
            "stages": stages}


def _handle(job):
    inp = job.get("input") or {}
    action = inp.get("action", "generate")
    print(f"[worker] job ricevuto action={action}", flush=True)
    try:
        if action == "encode_voice":
            return _action_encode_voice(inp)
        if action == "generate":
            # L'id del job serve a generare la chiave S3 quando il chiamante
            # non ne passa una: e' l'unico identificatore che lega il file al
            # lavoro che l'ha prodotto.
            return _action_generate(inp, str(job.get("id") or ""))
        if action == "diag":
            return _action_diag(inp)
        if action == "health":
            _pool()
            return {"ok": True, "sample_rate": _SAMPLE_RATE,
                    "timesteps": TIMESTEPS}
        return {"error": f"azione sconosciuta: {action}"}
    except Exception as e:  # noqa: BLE001 — l'errore deve tornare al chiamante
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-4000:]}


async def handler(job):
    """Sposta il lavoro bloccante fuori dal loop dell'SDK.

    _handle blocca (costruzione pool, attese sul loop dedicato). Se bloccasse
    il loop su cui gira il worker RunPod, l'SDK non potrebbe piu' rispondere
    alle proprie chiamate di controllo per tutta la durata del job.
    """
    return await asyncio.get_running_loop().run_in_executor(None, _handle, job)


# Il guard non e' una formalita': e' obbligatorio qui.
# AsyncVoxCPM2ServerPool avvia i propri server in processi separati con il
# metodo "spawn" (popen_spawn_posix). Spawn RE-IMPORTA il modulo __main__ nel
# figlio, e __main__ e' questo file (CMD python -u /app/handler.py). Senza
# guard il figlio rieseguiva runpod.serverless.start diventando un SECONDO
# worker RunPod nello stesso container: prendeva un job, tentava di costruire
# a sua volta il pool e multiprocessing sollevava _check_not_importing_main.
# Misurato: e' la causa delle morti silenziose del worker fra i 50 e i 90s.
# Nel figlio il modulo si chiama "__mp_main__", quindi il guard lo esclude.
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
