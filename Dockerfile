# syntax=docker/dockerfile:1.7
#
# Worker RunPod Serverless per VoxCPM2 su motore nanovllm-voxcpm.
#
# BASE: immagine PyTorch ufficiale `runtime`, non `devel` e non runpod/pytorch.
#   - RunPod Serverless non impone una base propria: serve solo Python + il
#     pacchetto `runpod`. La base runpod/pytorch porta jupyter/ssh, inutili qui.
#   - `runtime` invece di `devel` perche' flash-attn arriva come wheel gia'
#     compilata: senza compilazione nvcc non serve, e si risparmiano ~4 GB.
#
# PESI COTTI NELL'IMMAGINE (~5 GB) invece che su network volume: l'uso e'
# sporadico (pochi libri al giorno), quindi ogni invocazione e' un cold start e
# il caricamento e' fatturato. Leggere da layer locale batte il volume di rete.
# Il prezzo e' il container disk (~0,10 USD/GB/mese).
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

# HF_HUB_ENABLE_HF_TRANSFER e' deprecato e non ha piu' effetto: huggingface_hub
# emette un FutureWarning a ogni import, che finiva nello stdout catturato dalle
# sonde diagnostiche. Rimosso perche' e' rumore, non funzionalita'.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# flash-attn e' un requisito a runtime di nanovllm-voxcpm (il pacchetto lo
# importa all'avvio). La wheel giusta dipende da TRE assi: versione torch,
# versione Python e ABI C++ con cui torch e' stato compilato. Sbagliarne uno
# non da' errore in build ma un ImportError a runtime, cioe' dentro un worker
# GPU fatturato. Per questo l'URL viene DERIVATO dall'immagine invece che
# scritto a mano: cambiando la base, la wheel si riallinea da sola.
ARG FLASH_ATTN_VERSION=2.8.3.post1
RUN python - "$FLASH_ATTN_VERSION" <<'PY' > /tmp/fa_url.txt
import sys, torch
fa = sys.argv[1]
torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
cp = f"cp{sys.version_info.major}{sys.version_info.minor}"
print(f"https://github.com/Dao-AILab/flash-attention/releases/download/"
      f"v{fa}/flash_attn-{fa}+cu12torch{torch_mm}cxx11abi{abi}"
      f"-{cp}-{cp}-linux_x86_64.whl", end="")
PY
RUN echo "flash-attn wheel: $(cat /tmp/fa_url.txt)" \
 && pip install --no-cache-dir "$(cat /tmp/fa_url.txt)" \
 && python -c "import flash_attn; print('flash-attn', flash_attn.__version__)"

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pesi nell'immagine: e' questo che rende il cold start accettabile.
ARG VOXCPM_MODEL=openbmb/VoxCPM2
# Retry con backoff: HuggingFace rate-limita i download anonimi (429 misurato
# dopo alcune build ravvicinate) e senza ritentativo l'intero workflow muore
# a valle di 5 minuti di lavoro gia' fatto. La cache dei layer in CI riduce la
# frequenza di questo step, non lo elimina.
RUN python - "$VOXCPM_MODEL" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
repo = sys.argv[1]
for i in range(7):
    try:
        print("pesi in", snapshot_download(repo), flush=True)
        break
    except Exception as e:
        if i == 6:
            raise
        # Tetto a 10 minuti: il rate limit HF si e' visto durare piu' di
        # qualche minuto, e il tempo di CI su repo pubblico non si paga.
        wait = min(600, 60 * 2 ** i)
        print(f"tentativo {i+1} fallito ({type(e).__name__}: {e}); "
              f"riprovo fra {wait}s", flush=True)
        time.sleep(wait)
PY
ENV ABM_VOXCPM_MODEL=${VOXCPM_MODEL}

# Un endpoint = una configurazione di qualita': i timesteps sono un parametro
# del costruttore del pool e non si possono variare per singola richiesta.
ENV ABM_VOXCPM_TIMESTEPS=10 \
    ABM_VOXCPM_CONCURRENCY=16 \
    ABM_VOXCPM_CFG=2.0 \
    ABM_VOXCPM_GPU_MEM_UTIL=0.9

# Compilatore C per Triton. Il motore usa torch.compile: Inductor genera
# kernel Triton e li COMPILA a runtime, al warmup del modello. Senza gcc/g++
# il pool muore con "InductorError: Failed to find C compiler" — misurato sul
# worker. La base e' l'immagine `runtime` (non `devel`) perche' flash-attn
# arriva come wheel gia' compilata, ma questo non copre la compilazione dei
# kernel a runtime.
#
# Volutamente DOPO il layer dei pesi: messo insieme all'apt-get iniziale
# invaliderebbe la cache di tutto cio' che segue, cioe' 5 GB di pesi
# riscaricati da HuggingFace a ogni modifica dell'immagine.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ \
 && rm -rf /var/lib/apt/lists/* \
 && gcc --version | head -1

# Triton non compila un .c qualunque: compila un modulo di estensione Python,
# quindi servono anche gli header (Python.h). Verificarlo qui costa nulla;
# scoprirlo a runtime costa un job GPU e un giro di release.
RUN python - <<'PY'
import subprocess, sysconfig, tempfile, os, sys
inc = sysconfig.get_paths()["include"]
src = "#include <Python.h>\nint main(void){ return 0; }\n"
with tempfile.TemporaryDirectory() as d:
    c = os.path.join(d, "t.c")
    open(c, "w").write(src)
    r = subprocess.run(["gcc", "-I", inc, c, "-o", os.path.join(d, "t")],
                       capture_output=True, text=True)
    print("include dir:", inc)
    if r.returncode:
        print(r.stderr)
        sys.exit("gcc non compila contro Python.h")
    print("gcc + Python.h ok")
PY

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
