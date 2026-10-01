#!/bin/bash
# r3v_sycl_run.sh — round3-final-restore SYCL battery on B50
# audio.cpp round3: yue2 XIELU/EXPM1/ROUND_BF16 + irodori cpy f32<->i32 restored
# usage: ./r3v_sycl_run.sh <case|all>
cd /e/b50-test/round3-verify/sycl
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export FATTN_BACKEND=4
unset UR_L0_USE_IMMEDIATE_COMMANDLISTS
T=900
M=E:/Models/sound/audiocpp
X=/e/b50-test/matrix-models
run() { # name task model out fam sess req text wav_in wav_ref [env...]
  local name=$1 task=$2 model=$3 out=$4 fam=$5 sess=$6 req=$7 text=$8 win=$9 wref=${10}; shift 10
  echo "== $name start $(date '+%T')"
  env "$@" timeout $T stdbuf -o0 -e0 ./fattn2_test.exe "$task" "$model" "$out" "$fam" "$sess" "$req" "$text" "$win" "$wref" > "r3v-$name.log" 2>&1
  local rc=$?
  echo "== $name rc=$rc $(date '+%T') $(grep -m1 -oE 'ggml_sycl: [0-9]+ = [^(]+' r3v-$name.log)"
  grep -E '\[stat' "r3v-$name.log" | tail -2 | cut -c1-170
  tail -2 "r3v-$name.log" | cut -c1-170
}
case "$1" in
  # --- round3 unblocked (key focus) ---
  yue2_en)      T=1200 run yue2_en gen $M/Yue2-3B-GGUF r3v-yue2-en.wav yue2 '@sess-empty.json' '@yue2-en-req.json' yue2-en-lyrics.txt dummy.txt '' ;;
  yue2_zh)      T=1200 run yue2_zh gen $M/Yue2-3B-GGUF r3v-yue2-zh.wav yue2 '@sess-empty.json' '@yue2-zh-req.json' yue2-zh-lyrics.txt dummy.txt '' ;;
  irodori)      run irodori     tts $M/irodori-tts-600m-v3-voicedesign-q8_0.gguf r3v-irodori.wav irodori_tts '' '@irodori-req.json' ja-text.txt '' '' ;;
  # --- round2 Pass re-verification (no regression) ---
  seed_vc)      run seed_vc     vc  $M/seed-vc-mlx-q8_0.gguf r3v-seedvc.wav seed_vc '' '{"num_inference_steps":20}' dummy.txt jfk.wav ref-vc.wav ;;
  rvc_fused)    run rvc_fused   vc  $M/rvc-f16.gguf r3v-rvc-fused.wav   rvc '' '{}' dummy.txt jfk.wav ref-vc.wav RVC_GRU_FUSED=1 ;;
  rvc_unrolled) run rvc_unrolled vc $M/rvc-f16.gguf r3v-rvc-unrolled.wav rvc '' '{}' dummy.txt jfk.wav ref-vc.wav RVC_GRU_FUSED=0 ;;
  htdemucs)     run htdemucs    seps $M/htdemucs-q8_0.gguf r3v-htdemucs htdemucs '' '{}' dummy.txt jfk44.wav '' ;;
  qwen3_q8)     run qwen3_q8    asr $M/qwen3-asr-0.6b-q8_0.gguf r3v-qwen3-q8.txt qwen3_asr '' '{}' dummy.txt jfk.wav '' ;;
  qwen3_f16)    run qwen3_f16   asr $M/qwen3-asr-0.6b-f16.gguf r3v-qwen3-f16.txt qwen3_asr '' '{}' dummy.txt jfk.wav '' ;;
  dramabox)     T=1500 run dramabox tts /e/b50-test/q4test/dramabox-q4_k.gguf r3v-dramabox.wav dramabox '{}' '{}' dummy.txt dummy.txt '' ;;
  ace_step)     run ace_step    gen $M/ace-step-1.5-turbo-q8_0.gguf r3v-ace_step.wav ace_step '@ace-session.json' '@ace-req.json' ace-prompt.txt dummy.txt '' ;;
  chatterbox)   run chatterbox  clon $M/chatterbox-q8_0.gguf r3v-chatterbox.wav chatterbox '' '{}' '' '' jfk.wav ;;
  music3)       T=1800 run music3 gen $M/MiniMax-Music3-GGUF r3v-music3.wav minimax_music3 '{}' '@music3-req.json' dummy.txt dummy.txt '' ;;
  heartmula)    T=1800 run heartmula gen $M/heartmula-q8_0.gguf r3v-heartmula.wav heartmula '{}' '@heartmula-req.json' heartmula-prompt.txt dummy.txt '' ;;
  # --- family expansion (20+ families) ---
  parakeet)     run parakeet    asr $M/parakeet-tdt-0.6b-v3-q8_0.gguf r3v-parakeet.txt parakeet_tdt '' '{}' dummy.txt jfk.wav '' ;;
  r2t2)         run r2t2        asr $X/Confucius4-R2T2-GGUF/r2t2-q8_0.gguf r3v-r2t2.txt confucius4_r2t2 '' '{}' dummy.txt jfk.wav '' ;;
  vibevoice)    T=1200 run vibevoice asr $M/vibevoice-asr-q8_0.gguf r3v-vibevoice.txt vibevoice_asr '' '{}' dummy.txt jfk.wav '' ;;
  dots_soar)    run dots_soar   tts $M/dots-tts-soar-q8_0.gguf r3v-dots-soar.wav dots_tts '' '{}' dummy.txt '' '' ;;
  index_tts2)   run index_tts2  tts $M/index-tts2-q8_0.gguf r3v-index-tts2.wav index_tts2 '' '{}' dummy.txt '' jfk.wav ;;
  miotts)       run miotts      tts $M/miotts-1.7b-q8_0.gguf r3v-miotts.wav miotts '@miotts-sess-remote.json' '{}' dummy.txt '' jfk.wav ;;
  glm_tts)      run glm_tts     tts $M/glm-tts-q8_0.gguf r3v-glm-tts.wav glm_tts '' '@glm-req2.json' dummy.txt '' jfk.wav ;;
  echo_tts)     run echo_tts    clon $X/Echo-TTS-GGUF r3v-echo-tts.wav echo_tts '' '@echo-req.json' dummy.txt '' jfk.wav ;;
  stable_audio) run stable_audio gen $M/stable-audio-3-small-music-q8_0.gguf r3v-stable-audio.wav stable_audio '{}' '@sa-req.json' sa-prompt.txt dummy.txt '' ;;
  confucius4_tts) run confucius4_tts clon $M/confucius4-tts-orig.gguf r3v-confucius4-tts.wav confucius4_tts '' '{}' '' '' jfk.wav ;;
  audio8_tts)   run audio8_tts  tts $X/Audio8-TTS-Preview-0.6B-GGUF r3v-audio8-tts.wav audio8_tts '' '{}' dummy.txt '' '' ;;
  bs_roformer)  run bs_roformer seps $M/bs-roformer-ep368-q8_0.gguf r3v-bs-roformer bs_roformer '' '{}' dummy.txt jfk44.wav '' ;;
  all)
    for t in yue2_en yue2_zh irodori seed_vc rvc_fused rvc_unrolled htdemucs qwen3_q8 qwen3_f16 dramabox ace_step chatterbox music3 heartmula parakeet r2t2 vibevoice dots_soar index_tts2 miotts glm_tts echo_tts stable_audio confucius4_tts audio8_tts bs_roformer; do "$0" "$t"; done
    echo "ALL DONE" ;;
esac
