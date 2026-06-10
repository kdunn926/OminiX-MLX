#!/usr/bin/env bash
#
# Prefill/decode performance sweep for OminiX-MLX.
#
# Designed against the bench entry points verified in docs/perf-sweep-plan.md:
#   - generate     (qwen3-6-mlx)  -> TurboQuant / QUANTIZE_KV / TURBO_KV + SINK/SIMD/KV_MIN
#   - bench_mtplx  (mtplx-mlx)    -> MTPLX speculative, MTPLX_BLOCK_LEN, greedy/speculative
#   - bench_dflash (dflash-mlx)   -> DFlash speculative + DDTree, TURBO_KV
#
# SAFE BY DEFAULT: DRY_RUN=1 only prints the commands it would run. Model loads
# are expensive, so review the dry-run output before setting DRY_RUN=0.
#
# Usage:
#   DRY_RUN=1 SWEEP=quick scripts/bench_sweep.sh        # print quick plan
#   DRY_RUN=0 SWEEP=quick scripts/bench_sweep.sh        # execute quick sweep
#   DRY_RUN=0 SWEEP=full  scripts/bench_sweep.sh        # execute full sweep
#
# Env knobs:
#   DRY_RUN     (default 1)        1=print only, 0=execute
#   SWEEP       (default quick)    quick | full
#   SKIP_BUILD  (default 0)        1=skip cargo build of examples
#   REPS        (default 2)        repetitions per cell (keep last)
#   MODELS_DIR  (default ./models) where checkpoints live
#   OUT_DIR     (default ./results)
#   LONG_PROMPT (default built-in long paragraph)

set -uo pipefail

# --- resolve repo root (script lives in <root>/scripts) ---------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

DRY_RUN="${DRY_RUN:-1}"
SWEEP="${SWEEP:-quick}"
SKIP_BUILD="${SKIP_BUILD:-0}"
REPS="${REPS:-2}"
MODELS_DIR="${MODELS_DIR:-./models}"
OUT_DIR="${OUT_DIR:-./results}"
BIN_DIR="./target/release/examples"

CSV="${OUT_DIR}/bench_sweep.csv"
JSONL="${OUT_DIR}/bench_sweep.jsonl"

SHORT_PROMPT="The theory of general relativity"
# ~1-2k token long prompt (repeated paragraph) to stress prefill/TTFT.
DEFAULT_LONG="$(python3 - <<'PY'
print(("The quick brown fox jumps over the lazy dog. " * 220).strip())
PY
)"
LONG_PROMPT="${LONG_PROMPT:-$DEFAULT_LONG}"

mkdir -p "${OUT_DIR}"

CSV_HEADER="timestamp,sweep,model,harness,prompt,max_tokens,kv_mode,sink_tokens,simd_matmul,fused_kv_min,block_len,accept_mode,temp,ddtree,prefill_s,decode_tps,total_tokens,accept_rate,extra"
if [[ ! -f "${CSV}" ]]; then
  echo "${CSV_HEADER}" > "${CSV}"
fi

log()  { printf '[sweep] %s\n' "$*" >&2; }
die()  { printf '[sweep][err] %s\n' "$*" >&2; exit 1; }

# --- build examples ---------------------------------------------------------
build_examples() {
  [[ "${SKIP_BUILD}" == "1" ]] && { log "SKIP_BUILD=1, not building"; return; }
  log "building examples (cargo build --release)..."
  local cmds=(
    "cargo build --release -p qwen3-6-mlx --example generate"
    "cargo build --release -p mtplx-mlx   --example bench_mtplx"
    "cargo build --release -p dflash-mlx  --example bench_dflash"
    "cargo build --release -p gemma4-mlx  --example ar_bench"
  )
  for c in "${cmds[@]}"; do
    log "  $c"
    if [[ "${DRY_RUN}" == "0" ]]; then
      eval "$c" || die "build failed: $c"
    fi
  done
}

# --- helpers ----------------------------------------------------------------

model_path() { printf '%s/%s' "${MODELS_DIR}" "$1"; }

model_present() { [[ -d "$(model_path "$1")" ]]; }

# Build a stable key from the full param tuple to detect already-done cells.
cell_key() {
  # args: model harness prompt max_tokens kv_mode sink simd kvmin block accept temp ddtree
  printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s' "$@"
}

cell_done() {
  local key="$1"
  # Match on the descriptive columns (model..ddtree => CSV cols 3..14).
  # We rebuild the same pipe-joined signature from each CSV row and compare.
  [[ -f "${CSV}" ]] || return 1
  awk -F',' -v want="${key}" '
    NR>1 {
      sig=$3"|"$4"|"$5"|"$6"|"$7"|"$8"|"$9"|"$10"|"$11"|"$12"|"$13"|"$14
      if (sig==want) found=1
    }
    END { exit found?0:1 }
  ' "${CSV}"
}

# Parse a captured run log into "prefill_s decode_tps total_tokens accept_rate extra".
# Handles all three harnesses' metric formats.
parse_metrics() {
  local logfile="$1"
  python3 - "$logfile" <<'PY'
import re, sys
txt = open(sys.argv[1], errors="replace").read()

def grab(pat, default=""):
    m = re.search(pat, txt)
    return m.group(1) if m else default

prefill = grab(r"prefill_s=([0-9.]+)")
if not prefill:
    # generate.rs: "TTFT: 1.23s | ..."
    prefill = grab(r"TTFT:\s*([0-9.]+)s")

decode = grab(r"decode_tok_per_s=([0-9.]+)")          # bench_mtplx
if not decode:
    decode = grab(r"decode_tok_s=([0-9.]+)")          # bench_dflash
if not decode:
    decode = grab(r"\(([0-9.]+)\s*tok/s\)")           # generate.rs

total = grab(r"total_tokens=([0-9]+)")
if not total:
    total = grab(r"Total:\s*([0-9]+)\s*tok")          # generate.rs

accept = grab(r"acceptance_rate=([0-9.]+)")           # bench_mtplx
if not accept:
    accept = grab(r"acceptance_ratio=([0-9.]+)")      # bench_dflash

extra_bits = []
for k in ("avg_block_len", "peak_mem"):
    v = grab(rf"{k}=([0-9.]+)")
    if v:
        extra_bits.append(f"{k}={v}")
m = re.search(r"adaptive:\s*Large=(\d+)\s+Reduced=(\d+)\s+Probe=(\d+)", txt)
if m:
    extra_bits.append(f"adaptive={m.group(1)}/{m.group(2)}/{m.group(3)}")
m = re.search(r"Speedup ratio:\s*([0-9.]+)x", txt)
if m:
    extra_bits.append(f"speedup={m.group(1)}x")
m = re.search(r"kv_backend:\s*(\S+)", txt)
if m:
    extra_bits.append(f"kv_backend={m.group(1)}")
extra = ";".join(extra_bits)

print("\t".join([prefill, decode, total, accept, extra]))
PY
}

# Run one cell. Env-var assignments are passed as a single string ($1),
# the full command as the rest. Records a CSV + JSONL row.
#   run_cell <env_str> <model> <harness> <prompt_label> <max_tokens> \
#            <kv_mode> <sink> <simd> <kvmin> <block> <accept> <temp> <ddtree> -- <cmd...>
run_cell() {
  local env_str="$1" model="$2" harness="$3" prompt_label="$4" max_tokens="$5"
  local kv_mode="$6" sink="$7" simd="$8" kvmin="$9" block="${10}" accept="${11}" temp="${12}" ddtree="${13}"
  shift 13
  [[ "$1" == "--" ]] && shift
  local cmd=("$@")

  local key
  key="$(cell_key "$model" "$harness" "$prompt_label" "$max_tokens" "$kv_mode" "$sink" "$simd" "$kvmin" "$block" "$accept" "$temp" "$ddtree")"

  if cell_done "${key}"; then
    log "skip (already in CSV): ${key}"
    return
  fi

  local full="${env_str} ${cmd[*]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY-RUN: %s\n' "${full}"
    return
  fi

  log "RUN: ${full}"
  local prefill="" decode="" total="" arate="" extra=""
  local r tmplog
  for ((r=1; r<=REPS; r++)); do
    tmplog="$(mktemp)"
    # env_str holds KEY=VAL pairs; eval so they apply to this command only.
    eval "${env_str} \"\${cmd[@]}\"" >"${tmplog}" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
      log "  rep ${r} exited rc=${rc} (capturing partial output)"
    fi
    # keep the LAST rep's metrics (warm)
    IFS=$'\t' read -r prefill decode total arate extra < <(parse_metrics "${tmplog}")
    rm -f "${tmplog}"
  done

  local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${ts}" "${SWEEP}" "${model}" "${harness}" "${prompt_label}" "${max_tokens}" \
    "${kv_mode}" "${sink}" "${simd}" "${kvmin}" "${block}" "${accept}" "${temp}" "${ddtree}" \
    "${prefill}" "${decode}" "${total}" "${arate}" "${extra}" >> "${CSV}"
  printf '{"ts":"%s","sweep":"%s","model":"%s","harness":"%s","prompt":"%s","max_tokens":%s,"kv_mode":"%s","sink_tokens":"%s","simd_matmul":"%s","fused_kv_min":"%s","block_len":"%s","accept_mode":"%s","temp":"%s","ddtree":"%s","prefill_s":"%s","decode_tps":"%s","total_tokens":"%s","accept_rate":"%s","extra":"%s"}\n' \
    "${ts}" "${SWEEP}" "${model}" "${harness}" "${prompt_label}" "${max_tokens}" \
    "${kv_mode}" "${sink}" "${simd}" "${kvmin}" "${block}" "${accept}" "${temp}" "${ddtree}" \
    "${prefill}" "${decode}" "${total}" "${arate}" "${extra}" >> "${JSONL}"
  log "  -> prefill_s=${prefill} decode_tps=${decode} total=${total} accept=${arate} ${extra}"
}

prompt_text_for() { [[ "$1" == "long" ]] && printf '%s' "${LONG_PROMPT}" || printf '%s' "${SHORT_PROMPT}"; }

# ============================================================================
# Sweep definitions
# ============================================================================

# Per-sweep parameter sets.
if [[ "${SWEEP}" == "quick" ]]; then
  CELL_A_MODELS=("Qwen3.6-27B-4bit" "Qwen3.6-27B-UD-MLX-4bit")
  CELL_D_MODELS=("gemma-4-e4b-it-4bit")
  PROMPTS=("short")
  MAXTOKS=(64)
  # Include QUANTIZE_KV so the quick sweep measures the F9 fused-quantized-
  # attention path (which now skips the per-step full dequantize).
  KV_MODES=("unset" "QUANTIZE_KV" "PAGED" "TURBO_KV")
  SINKS=(0 4)
  SIMDS=("unset")
  KVMINS=("8192")          # plus 512 only for long prompt in full sweep
  MTPLX_BLOCKS=(1 4)
  MTPLX_ACCEPTS=("greedy@0.0")
  DFLASH_TEMPS=(0.0)
  DFLASH_DDTREE=("off")
elif [[ "${SWEEP}" == "full" ]]; then
  CELL_A_MODELS=("Qwen3.6-27B-4bit" "Qwen3.6-27B-UD-MLX-4bit" "Qwen3.6-35B-A3B-4bit")
  CELL_D_MODELS=("gemma-4-e4b-it-4bit" "gemma-4-26B-A4B-it")
  PROMPTS=("short" "long")
  MAXTOKS=(64 256)
  KV_MODES=("unset" "QUANTIZE_KV" "PAGED" "TURBO_KV")
  SINKS=(0 4)
  SIMDS=("unset" "1")
  KVMINS=("8192" "512")
  MTPLX_BLOCKS=(1 4 8)
  MTPLX_ACCEPTS=("greedy@0.0" "speculative@0.7")
  DFLASH_TEMPS=(0.0 0.7)
  DFLASH_DDTREE=("off" "on")
else
  die "unknown SWEEP='${SWEEP}' (expected quick|full)"
fi

# --- Cell A: TurboQuant KV + SDPA knobs (generate, qwen3.6) -----------------
cell_a() {
  local GEN="${BIN_DIR}/generate"
  for model in "${CELL_A_MODELS[@]}"; do
    if ! model_present "${model}"; then log "skip missing model ${model}"; continue; fi
    local mp; mp="$(model_path "${model}")"
    for prompt in "${PROMPTS[@]}"; do
      local ptext; ptext="$(prompt_text_for "${prompt}")"
      for mt in "${MAXTOKS[@]}"; do
        for kv in "${KV_MODES[@]}"; do
          if [[ "${kv}" == "TURBO_KV" ]]; then
            for sink in "${SINKS[@]}"; do
              for simd in "${SIMDS[@]}"; do
                # FUSED_KV_MIN=512 only meaningful on long prompts; on short keep 8192.
                local kvmins_use=("8192")
                [[ "${prompt}" == "long" ]] && kvmins_use=("${KVMINS[@]}")
                for kvmin in "${kvmins_use[@]}"; do
                  local env_str="TURBO_KV=1 TURBOQUANT_SINK_TOKENS=${sink} TURBOQUANT_FUSED_KV_MIN=${kvmin}"
                  [[ "${simd}" != "unset" ]] && env_str="${env_str} TURBOQUANT_SIMD_MATMUL=${simd}"
                  run_cell "${env_str}" "${model}" "generate" "${prompt}" "${mt}" \
                    "TURBO_KV" "${sink}" "${simd}" "${kvmin}" "-" "greedy" "0.0" "off" \
                    -- "${GEN}" "${mp}" "${ptext}" "${mt}"
                done
              done
            done
          else
            local env_str=""
            [[ "${kv}" == "QUANTIZE_KV" ]] && env_str="QUANTIZE_KV=1"
            [[ "${kv}" == "PAGED" ]] && env_str="PAGED_KV=1"
            run_cell "${env_str}" "${model}" "generate" "${prompt}" "${mt}" \
              "${kv}" "-" "-" "-" "-" "greedy" "0.0" "off" \
              -- "${GEN}" "${mp}" "${ptext}" "${mt}"
          fi
        done
      done
    done
  done
}

# --- Cell B: MTPLX speculative (bench_mtplx) --------------------------------
cell_b() {
  local BIN="${BIN_DIR}/bench_mtplx"
  local model="Qwen3.6-27B-MTPLX-Optimized-Speed"
  if ! model_present "${model}"; then log "skip missing model ${model}"; return; fi
  local mp; mp="$(model_path "${model}")"
  for prompt in "${PROMPTS[@]}"; do
    local ptext; ptext="$(prompt_text_for "${prompt}")"
    for mt in "${MAXTOKS[@]}"; do
      for block in "${MTPLX_BLOCKS[@]}"; do
        for acc in "${MTPLX_ACCEPTS[@]}"; do
          local mode="${acc%@*}" temp="${acc#*@}"
          local flag="--greedy"; [[ "${mode}" == "speculative" ]] && flag="--speculative"
          run_cell "MTPLX_BLOCK_LEN=${block}" "${model}" "bench_mtplx" "${prompt}" "${mt}" \
            "session-default" "-" "-" "-" "${block}" "${mode}" "${temp}" "off" \
            -- "${BIN}" --target "${mp}" --prompt "${ptext}" --max-tokens "${mt}" --temp "${temp}" "${flag}"
        done
      done
    done
  done
}

# --- Cell C: DFlash (bench_dflash) ------------------------------------------
# target -> draft pairs (draft optional; missing/unsupported falls back gracefully)
cell_c() {
  local BIN="${BIN_DIR}/bench_dflash"
  local pairs=(
    "Qwen3.6-35B-A3B-4bit:Qwen3.6-35B-A3B-DFlash"
    "gemma-4-26B-A4B-it:gemma-4-26B-A4B-it-DFlash"
  )
  for pair in "${pairs[@]}"; do
    local target="${pair%%:*}" draft="${pair##*:}"
    if ! model_present "${target}"; then log "skip missing target ${target}"; continue; fi
    local tp; tp="$(model_path "${target}")"
    local draft_args=()
    if model_present "${draft}"; then draft_args=(--draft "$(model_path "${draft}")"); else log "note: draft ${draft} missing — AR/Missing path"; fi
    local is_gemma="no"; [[ "${target}" == gemma-4* ]] && is_gemma="yes"
    for prompt in "${PROMPTS[@]}"; do
      local ptext; ptext="$(prompt_text_for "${prompt}")"
      for mt in "${MAXTOKS[@]}"; do
        for temp in "${DFLASH_TEMPS[@]}"; do
          for dd in "${DFLASH_DDTREE[@]}"; do
            # KV modes: only Gemma4 target reads TURBO_KV here; and TURBO_KV+ddtree is rejected.
            local kvopts=("unset")
            [[ "${is_gemma}" == "yes" && "${dd}" == "off" ]] && kvopts=("unset" "TURBO_KV")
            for kv in "${kvopts[@]}"; do
              local env_str=""; [[ "${kv}" == "TURBO_KV" ]] && env_str="TURBO_KV=1"
              local dd_args=(); [[ "${dd}" == "on" ]] && dd_args=(--ddtree --tree-budget 16 --tree-topk 4)
              run_cell "${env_str}" "${target}" "bench_dflash" "${prompt}" "${mt}" \
                "${kv}" "-" "-" "-" "draft" "rejection" "${temp}" "${dd}" \
                -- "${BIN}" --target "${tp}" "${draft_args[@]}" --prompt "${ptext}" --max-tokens "${mt}" --temp "${temp}" "${dd_args[@]}"
            done
          done
        done
      done
    done
  done
}

# --- Cell D: Gemma4 AR throughput (ar_bench) --------------------------------
# Standalone harness for Gemma4 variants — paged-vs-standard comparison on the
# mask-only sliding-window architecture. Covers gemma-4-e4b-it-4bit (the new
# 4-bit checkpoint) and the original gemma-4-26B-A4B-it baseline. ar_bench
# emits `prefill_s=` / `decode_tok_s=` in the same format bench_dflash does,
# so the existing parser captures it as-is.
cell_d() {
  local BIN="${BIN_DIR}/ar_bench"
  for model in "${CELL_D_MODELS[@]}"; do
    if ! model_present "${model}"; then log "skip missing model ${model}"; continue; fi
    local mp; mp="$(model_path "${model}")"
    # UD-MLX-4bit checkpoints (e.g. gemma-4-e4b-it-4bit) ship with the
    # `language_model.model.*` weight prefix — route ar_bench through the
    # UD loader to avoid `Weight not found: model.language_model.…`.
    local loader_env=""; [[ "${model}" == *4bit ]] && loader_env="LOADER=ud"
    for prompt in "${PROMPTS[@]}"; do
      local ptext; ptext="$(prompt_text_for "${prompt}")"
      for mt in "${MAXTOKS[@]}"; do
        # ar_bench KV modes: standard fp16 (unset) and PAGED (mixed cache —
        # pages only full-attn layers; sliding stay contiguous).
        for kv in "unset" "PAGED"; do
          local env_str="${loader_env}"
          [[ "${kv}" == "PAGED" ]] && env_str="${env_str:+${env_str} }PAGED_KV=1"
          run_cell "${env_str}" "${model}" "ar_bench" "${prompt}" "${mt}" \
            "${kv}" "-" "-" "-" "-" "greedy" "0.0" "off" \
            -- "${BIN}" "${mp}" "${mt}" "${ptext}"
        done
      done
    done
  done
}

# ============================================================================
main() {
  log "SWEEP=${SWEEP} DRY_RUN=${DRY_RUN} REPS=${REPS} MODELS_DIR=${MODELS_DIR}"
  log "CSV=${CSV} JSONL=${JSONL}"
  build_examples
  cell_a
  cell_b
  cell_c
  cell_d
  log "done. results in ${CSV}"
}

main "$@"
