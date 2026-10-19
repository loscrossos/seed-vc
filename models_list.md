# Automatically Downloaded Models in Seed-VC

This document lists all models and pretrained files used by the Seed-VC repository during inference, training, GUI execution, and evaluation. Every link below has been verified to exist (HTTP 200 OK).

## Local Checkpoints Directory Structure

By default, Seed-VC expects model files under the `checkpoints` directory in the repository root, organized by the repository name of the original repository:

```
checkpoints/
├── campplus/
│   └── campplus_cn_common.bin
├── VoiceConversionWebUI/
│   └── rmvpe.pt
├── CosyVoice-300M/
│   └── hift.pt
├── OpenVoiceV2/
│   └── converter/
│       ├── checkpoint.pth
│       └── config.json
├── Seed-VC/
│   ├── se_db.pt
│   ├── DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth
│   ├── DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth
│   ├── DiT_uvit_tat_xlsr_ema.pth
│   └── v2/
│       ├── cfm_small.pth
│       └── ar_base.pth
├── ASTRAL-quantization/
│   ├── bsq32/
│   │   └── bsq32_light.pth
│   └── bsq2048/
│       └── bsq2048_light.pth
├── whisper-small/
│   └── (Hugging Face / transformers files)
├── bigvgan_v2_22khz_80band_256x/
│   ├── config.json
│   └── bigvgan_generator.pt
└── bigvgan_v2_44khz_128band_512x/
    ├── config.json
    └── bigvgan_generator.pt
```

### Base Models Directory Argument
A custom base checkpoints directory can be passed to all entry scripts (`app.py`, `app_vc.py`, `app_svc.py`, `app_vc_v2.py`, `inference.py`, `inference_v2.py`, `eval.py`, `real-time-gui.py`, `train.py`, `train_v2.py`) using the `--checkpoint-dir` flag:

```bash
python inference.py --checkpoint-dir /path/to/my_checkpoints ...
python app.py --checkpoint-dir /path/to/my_checkpoints ...
```

Or via the `CHECKPOINT_DIR` / `SEED_VC_CHECKPOINT_DIR` environment variable.

If a model is found locally in the checkpoints directory, it is loaded immediately without connecting to Hugging Face. If it is not found, it is automatically downloaded from Hugging Face into the designated checkpoints directory.

---

## 1. Seed-VC Checkpoints & Configs (`Plachta/Seed-VC`)
*Repository: [https://huggingface.co/Plachta/Seed-VC](https://huggingface.co/Plachta/Seed-VC)*

### V1 Models
* **V1 Standard (Whisper-Small + WaveNet + BigVGAN)**
  * Checkpoint: [DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth](https://huggingface.co/Plachta/Seed-VC/resolve/main/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth)
  * Config: [config_dit_mel_seed_uvit_whisper_small_wavenet.yml](https://huggingface.co/Plachta/Seed-VC/resolve/main/config_dit_mel_seed_uvit_whisper_small_wavenet.yml)
* **V1 F0-conditioned / Singing Voice (44kHz BigVGAN)**
  * Checkpoint (v2): [DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth](https://huggingface.co/Plachta/Seed-VC/resolve/main/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth)
  * Checkpoint (v1): [DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth](https://huggingface.co/Plachta/Seed-VC/resolve/main/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth)
  * Config: [config_dit_mel_seed_uvit_whisper_base_f0_44k.yml](https://huggingface.co/Plachta/Seed-VC/resolve/main/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml)
* **V1 Tiny / XLSR (Fast / Real-time)**
  * Checkpoint: [DiT_uvit_tat_xlsr_ema.pth](https://huggingface.co/Plachta/Seed-VC/resolve/main/DiT_uvit_tat_xlsr_ema.pth)
  * Config: [config_dit_mel_seed_uvit_xlsr_tiny.yml](https://huggingface.co/Plachta/Seed-VC/resolve/main/config_dit_mel_seed_uvit_xlsr_tiny.yml)

### V2 Models
* **V2 CFM (Conditional Flow Matching)**
  * Checkpoint: [v2/cfm_small.pth](https://huggingface.co/Plachta/Seed-VC/resolve/main/v2/cfm_small.pth)
* **V2 AR (Autoregressive Style & Prosody Model)**
  * Checkpoint: [v2/ar_base.pth](https://huggingface.co/Plachta/Seed-VC/resolve/main/v2/ar_base.pth)

---

## 2. ASTRAL Quantization Encoders (`Plachta/ASTRAL-quantization`)
*Repository: [https://huggingface.co/Plachta/ASTRAL-quantization](https://huggingface.co/Plachta/ASTRAL-quantization)*

* **Narrow Content Extractor (BSQ 32)**
  * Checkpoint: [bsq32/bsq32_light.pth](https://huggingface.co/Plachta/ASTRAL-quantization/resolve/main/bsq32/bsq32_light.pth)
* **Wide Content Extractor (BSQ 2048)**
  * Checkpoint: [bsq2048/bsq2048_light.pth](https://huggingface.co/Plachta/ASTRAL-quantization/resolve/main/bsq2048/bsq2048_light.pth)

---

## 3. Speaker Style & Timbre Encoder (`funasr/campplus`)
*Repository: [https://huggingface.co/funasr/campplus](https://huggingface.co/funasr/campplus)*

* **CAMPPlus Speaker Verification Model**
  * Hugging Face File: [campplus_cn_common.bin](https://huggingface.co/funasr/campplus/blob/main/campplus_cn_common.bin)
  * Direct Weights Download: [campplus_cn_common.bin](https://huggingface.co/funasr/campplus/resolve/main/campplus_cn_common.bin)
  * Expected Local Path: `checkpoints/campplus/campplus_cn_common.bin`

---

## 4. Pitch / F0 Extractor (`lj1995/VoiceConversionWebUI`)
*Repository: [https://huggingface.co/lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI)*

* **RMVPE Model (Pitch Extraction for F0 Conditioning)**
  * Weights: [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt)

---

## 5. Vocoders

### NVIDIA BigVGAN
* **BigVGAN v2 (22kHz, 80-band, 256x hop size)**
  * Repository: [https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x)
  * Model Weights: [bigvgan_generator.pt](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x/resolve/main/bigvgan_generator.pt)
  * Config: [config.json](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x/resolve/main/config.json)
* **BigVGAN v2 (44kHz, 128-band, 512x hop size)**
  * Repository: [https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x)
  * Model Weights: [bigvgan_generator.pt](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x/resolve/main/bigvgan_generator.pt)
  * Config: [config.json](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x/resolve/main/config.json)

### FunAudioLLM CosyVoice HiFT (HiFi-GAN)
* **HiFT Vocoder**
  * Repository: [https://huggingface.co/FunAudioLLM/CosyVoice-300M](https://huggingface.co/FunAudioLLM/CosyVoice-300M)
  * Weights: [hift.pt](https://huggingface.co/FunAudioLLM/CosyVoice-300M/resolve/main/hift.pt)

---

## 6. Speech Tokenizers & Pretrained Encoders

### OpenAI Whisper Small
* Repository: [https://huggingface.co/openai/whisper-small](https://huggingface.co/openai/whisper-small)
* Weights: [model.safetensors](https://huggingface.co/openai/whisper-small/resolve/main/model.safetensors)
* Model Config: [config.json](https://huggingface.co/openai/whisper-small/resolve/main/config.json)
* Preprocessor Config: [preprocessor_config.json](https://huggingface.co/openai/whisper-small/resolve/main/preprocessor_config.json)

### Meta HuBERT Large (LL60k)
* Repository: [https://huggingface.co/facebook/hubert-large-ll60k](https://huggingface.co/facebook/hubert-large-ll60k)
* Weights: [pytorch_model.bin](https://huggingface.co/facebook/hubert-large-ll60k/resolve/main/pytorch_model.bin)
* Model Config: [config.json](https://huggingface.co/facebook/hubert-large-ll60k/resolve/main/config.json)
* Preprocessor Config: [preprocessor_config.json](https://huggingface.co/facebook/hubert-large-ll60k/resolve/main/preprocessor_config.json)

### Meta Wav2Vec2 XLS-R (300M)
* Repository: [https://huggingface.co/facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
* Weights: [pytorch_model.bin](https://huggingface.co/facebook/wav2vec2-xls-r-300m/resolve/main/pytorch_model.bin)
* Model Config: [config.json](https://huggingface.co/facebook/wav2vec2-xls-r-300m/resolve/main/config.json)
* Preprocessor Config: [preprocessor_config.json](https://huggingface.co/facebook/wav2vec2-xls-r-300m/resolve/main/preprocessor_config.json)

---

## 7. Real-Time VAD (Voice Activity Detection)
*Repository: [https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)*

* **FSMN-VAD (`real-time-gui.py`)**
  * Model Weights: [model.pt](https://modelscope.cn/api/v1/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/repo?Revision=master&FilePath=model.pt)
  * Config: [configuration.json](https://modelscope.cn/api/v1/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/repo?Revision=master&FilePath=configuration.json)

---

## 8. Evaluation Benchmark Models (`eval.py`)

### Meta HuBERT Large ASR (LS960-ft)
* Repository: [https://huggingface.co/facebook/hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft)
* Weights: [pytorch_model.bin](https://huggingface.co/facebook/hubert-large-ls960-ft/resolve/main/pytorch_model.bin)
* Model Config: [config.json](https://huggingface.co/facebook/hubert-large-ls960-ft/resolve/main/config.json)

### Microsoft WavLM Base Plus SV
* Repository: [https://huggingface.co/microsoft/wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv)
* Weights: [pytorch_model.bin](https://huggingface.co/microsoft/wavlm-base-plus-sv/resolve/main/pytorch_model.bin)
* Model Config: [config.json](https://huggingface.co/microsoft/wavlm-base-plus-sv/resolve/main/config.json)
