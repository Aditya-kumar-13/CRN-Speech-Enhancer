# Real-Time Voice Isolation

A single-channel speech-enhancement system for preserving a close, on-axis speaker
while suppressing distant speech and background noise. It includes reproducible
training and evaluation, offline file enhancement, blockwise streaming, a small web
interface, and Windows virtual-microphone routing.

## Features

- Dynamic target/interferer mixing at controlled signal-to-interference ratios.
- Speaker-disjoint speech splits and room-disjoint impulse-response splits.
- A 2.4 million-parameter convolutional recurrent mask estimator.
- Near-field simulation with a direct target and reverberant distant speakers.
- SI-SDR and SNR-improvement evaluation at fixed interference levels.
- Offline WAV enhancement and M4A/MP3 conversion through FFmpeg.
- Live 40 ms microphone blocks with retained past context and no future audio.
- Windows virtual-microphone routing through VB-CABLE.

## Signal path

```text
close speech + distant reverberant speech
                  |
                  v
             mono mixture
                  |
                  v
       STFT -> CRN mask -> inverse STFT
                  |
                  v
            enhanced speech
```

The close speaker is defined by acoustic proximity, not voice identity. No enrollment
recording is required. With a single input channel, the system cannot perform true
directional beamforming; that requires a microphone array.

## Evaluation

The near-field checkpoint was evaluated on 100 held-out mixtures per condition. Test
speakers and room-response groups were excluded from training.

| Target/interferer SNR | SI-SDR improvement | SNR improvement |
| ---: | ---: | ---: |
| -5 dB | +0.02 dB | +1.77 dB |
| 0 dB | +1.47 dB | +2.21 dB |
| 3 dB | +2.10 dB | +2.59 dB |
| 6 dB | +2.48 dB | +2.81 dB |
| 10 dB | +2.21 dB | +2.40 dB |

The model was trained from 0 to 15 dB. The -5 dB row is an out-of-range stress test,
where the distant speaker is louder than the target. Full precision results are in
[`docs/results/near_field_metrics.json`](docs/results/near_field_metrics.json).

In a local Windows loopback test, 40 ms blocks had 19.3 ms p95 CPU compute time and
no queue drops; two startup output underflows were observed. Audio-device buffering
adds further end-to-end latency, and results vary by hardware and driver backend.

## Repository layout

```text
configs/                       Training configurations
docs/results/                  Versioned evaluation summaries
src/voice_isolation/           Package source
src/voice_isolation/models/    CRN architecture
tests/                         Unit and pipeline tests
app.py                         Streamlit file-enhancement interface
```

Datasets, checkpoints, recordings, and generated outputs are intentionally ignored by
Git.

## Installation

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,demo,live]"
python -m pytest -q
```

Install FFmpeg separately if M4A or MP3 input is needed.

## Data preparation

The development configuration uses:

- [Mini LibriSpeech (OpenSLR SLR31)](https://www.openslr.org/31/), CC BY 4.0.
- [Room Impulse Response and Noise Database (OpenSLR SLR28)](https://www.openslr.org/28/),
  Apache 2.0.

Place extracted files under `data/raw/`, then create the manifests:

```powershell
python -m voice_isolation.prepare_minilibri `
  --train-root data/raw/LibriSpeech/train-clean-5 `
  --test-root data/raw/LibriSpeech/dev-clean-2 `
  --output-dir data/manifests

voice-prepare-rirs `
  --rir-root data/raw/RIRS_NOISES/simulated_rirs `
  --output-dir data/manifests
```

Manifest entries use absolute audio paths and optional speaker identifiers:

```json
{"path":"C:/datasets/LibriSpeech/train-clean-5/19/198/example.flac","speaker_id":"19"}
```

## Training

Run the short pipeline check before a full training job:

```powershell
voice-train --config configs/near_field_smoke.yaml
voice-train --config configs/near_field_baseline.yaml
```

The best checkpoint is written to `artifacts/near_field_baseline/best.pt`. Checkpoints
are not committed to the repository.

An environmental-noise-only configuration is also available:

```powershell
voice-train --config configs/noise_baseline.yaml
```

## Evaluation

```powershell
voice-evaluate `
  --config configs/near_field_baseline.yaml `
  --checkpoint artifacts/near_field_baseline/best.pt `
  --split test `
  --snr-db -5 0 3 6 10 `
  --max-examples 100 `
  --save-examples 2
```

## Offline enhancement

```powershell
voice-enhance recording.wav `
  --checkpoint artifacts/near_field_baseline/best.pt `
  --output artifacts/demo/recording_enhanced.wav
```

Add `--reference clean.wav` to calculate per-file SI-SDR and SNR improvements.

## Blockwise streaming simulation

```powershell
voice-stream-simulate recording.wav `
  --checkpoint artifacts/near_field_baseline/best.pt `
  --output artifacts/streaming/recording_enhanced.wav `
  --chunk-ms 100 `
  --context-ms 200
```

This command processes a file as sequential blocks and reports mean, p95, and maximum
compute latency. It does not access the microphone.

## Live virtual microphone on Windows

Install [VB-CABLE](https://vb-audio.com/Cable/) and restart Windows. List audio
devices, then start the live processor with a physical microphone ID:

```powershell
voice-live --list-devices
voice-live --input-device 10 --chunk-ms 40 --context-ms 200
```

When possible, the command selects a VB-CABLE playback endpoint from the same Windows
audio backend as the input device. Use `--output-device ID` to override it. In the
calling application, select `CABLE Output` as the microphone. Press Ctrl+C to stop.

Audio device IDs are assigned by Windows and can change after hardware or driver
updates.

## Web interface

```powershell
python -m streamlit run app.py
```

The interface accepts WAV, FLAC, M4A, and MP3 recordings and saves enhanced WAV
output under `artifacts/demo/`.

## Limitations

- Single-channel input provides no directional information.
- A louder distant speaker can dominate outside the trained SNR range.
- Blockwise processing is not stateful at the STFT or GRU level; past waveform context
  is reprocessed for each block.
- Mini LibriSpeech is intended for pipeline development. Larger and more varied speech
  data is needed for broad deployment.
- The virtual-microphone driver is a separate system dependency.

## License

Source code is released under the [MIT License](LICENSE). Dataset licenses apply
separately and their files are not included in this repository.
