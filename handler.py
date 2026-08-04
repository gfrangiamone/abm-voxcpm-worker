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
MAX_INLINE_BYTES = int(os.environ.get("ABM_VOXCPM_MAX_INLINE_BYTES", "6000000"))
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


async def _build_pool():
    """Costruisce il pool DENTRO il loop dedicato.

    La costruzione avviene qui e non nel chiamante perche' il pool registra i
    propri task di background sul loop attivo al momento della creazione: se
    nascesse sul loop dell'handler, quei task morirebbero con la chiamata.
    """
    try:
        from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2ServerPool
        pool = AsyncVoxCPM2ServerPool(
            model_path=MODEL_PATH,
            inference_timesteps=TIMESTEPS,
            max_num_seqs=MAX_NUM_SEQS,
            gpu_memory_utilization=GPU_MEM_UTIL,
            devices=[0],
        )
    except ImportError:
        from nanovllm_voxcpm import VoxCPM
        pool = VoxCPM.from_pretrained(
            model=MODEL_PATH,
            inference_timesteps=TIMESTEPS,
            max_num_seqs=MAX_NUM_SEQS,
            gpu_memory_utilization=GPU_MEM_UTIL,
            devices=[0],
        )
    # Il ripiego VoxCPM.from_pretrained quasi certamente non espone
    # wait_for_ready: e' un'API del pool asincrono, non del wrapper sincrono.
    waiter = getattr(pool, "wait_for_ready", None)
    if waiter is not None:
        await waiter()
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


def _upload_s3(data, cfg):
    """Carica su S3/R2 e restituisce la chiave. cfg: bucket/key/endpoint_url."""
    import boto3
    client = boto3.client(
        "s3",
        endpoint_url=cfg.get("endpoint_url") or os.environ.get("ABM_S3_ENDPOINT"),
        aws_access_key_id=(cfg.get("access_key")
                           or os.environ.get("ABM_S3_ACCESS_KEY")),
        aws_secret_access_key=(cfg.get("secret_key")
                               or os.environ.get("ABM_S3_SECRET_KEY")),
        region_name=cfg.get("region") or os.environ.get("ABM_S3_REGION", "auto"),
    )
    bucket = cfg.get("bucket") or os.environ["ABM_S3_BUCKET"]
    key = cfg["key"]
    client.put_object(Bucket=bucket, Key=key, Body=data)
    return {"bucket": bucket, "key": key}


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


def _action_generate(inp):
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
    payload = _to_wav(pcm, _SAMPLE_RATE) if fmt == "wav" else pcm
    out["format"] = fmt

    s3cfg = inp.get("s3")
    if s3cfg and s3cfg.get("key"):
        out["s3"] = _upload_s3(payload, s3cfg)
    elif len(payload) <= MAX_INLINE_BYTES:
        out["audio_b64"] = base64.b64encode(payload).decode("ascii")
    else:
        out["error"] = (f"audio di {len(payload):,} byte oltre il limite inline "
                        f"di {MAX_INLINE_BYTES:,}: fornire 's3' con 'key'")
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
    if inp.get("weights"):
        _stage(stages, "weights", _probe_weights)
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
            return _action_generate(inp)
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
