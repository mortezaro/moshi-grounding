# Serving an emotion-aware Moshi live

The trained models are LoRA adapters + an emotion conditioner on top of base Moshi. `moshi.server` serves them (websocket + web mic UI); the emotion is set per-instance via `COND_EMOTION`.

## 1. Start a server on a GPU (clariden)

```bash
C=$SCR/moshi_runs/E_u8/checkpoints/checkpoint_000800/consolidated
export COND_EMOTION=neu            # or hap / sad
srun ... $MOSHI_PY -m moshi.server \
  --lora-weight $C/lora.safetensors --config-path $C/config.json \
  --host 0.0.0.0 --port 8901
```

Launcher examples: `launchers/serve_E_u8.sh` (+ `serve_E_long.sh`, `serve_E_u12.sh`), each on its own port (8901/8902/8903). The job holds one GPU and runs to the wall-clock limit. `moshi.server`'s stdout is buffered when redirected — the server is up once the port answers (`curl http://<node>:<port>/` → 200), even if the log looks quiet.

## 2. Reach it from your laptop (SSH port-forward)

The gradio `--gradio-tunnel` flag has **no aarch64 build** and the cluster **blocks UDP**, so public tunnels can't run cluster-side. Instead, forward the compute-node ports to your laptop. In a **local** terminal:

```bash
ssh -L 8901:<nodeE_u8>:8901 -L 8902:<nodeElong>:8902 -L 8903:<nodeEu12>:8903 clariden
```

The `<node>:port` is resolved on the login node (which can reach the compute nodes). Keep the terminal open, then open `http://localhost:8901` (etc.) in your browser, allow the mic, click connect, and talk. `localhost` is a secure context so the mic works.

## 3. Share a public link (cloudflared, from the laptop)

Your laptop has open internet, so run the tunnel there (not the cluster):

```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:8901     # one per port, in separate terminals
```

Each prints a `https://<random>.trycloudflare.com` URL (HTTPS → remote mics work, no account needed). Send the labeled URLs to testers. The link is alive only while the SSH tunnel **and** that cloudflared process stay running; the URL changes on restart.

**Caveats:** each server handles one conversation at a time; the links are open/unauthenticated (share deliberately); everything depends on your laptop staying connected. For always-on, laptop-independent hosting, deploy the adapter on a cloud GPU and run `moshi.server` + cloudflared there.

## The served models (from the overnight sweep)

Trained with **light LoRA + emotion conditioner on Expresso** (real, clean, expressive) — the recipe that preserves Moshi quality (see `docs/EXPERIMENTS.md`). The emotion strength is a dial (conditioner LR upweight):

| Run | Character | Quality (fluency, base ~0.40) |
|---|---|---|
| E_long | quality-first, subtlest emotion | 0.392 |
| E_u8 | balanced (recommended) | 0.380 |
| E_u12 | expressive, more emotion | 0.377 |

Behavior stays real-time full-duplex (RTF ~0.4) and conversational; emotion even modulates engagement (happy = more active). Configs: `finetune/example/E_*.yaml`; build the data with `scripts/build_expresso.py`.
