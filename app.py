from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

from voice_isolation.inference import enhance_file

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "artifacts" / "near_field_baseline" / "best.pt"
OUTPUT_DIR = ROOT / "artifacts" / "demo"


def find_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    package_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    matches = list(package_root.glob("Gyan.FFmpeg_*/**/ffmpeg.exe"))
    return str(matches[0]) if matches else None


def convert_to_wav(source: Path, destination: Path) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found. Restart the terminal after installing it.")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(destination),
        ],
        check=True,
    )


def main() -> None:
    st.set_page_config(page_title="Voice Isolation Demo", page_icon="🎙️")
    st.title("Real-Time Voice Isolation")
    st.caption(
        "Universal near-field enhancement · 2.4M parameters · no voice enrollment"
    )

    uploaded = st.file_uploader(
        "Upload a noisy recording",
        type=["wav", "m4a", "mp3", "flac"],
    )
    if uploaded is None:
        st.info("Upload a recording to compare the original and enhanced audio.")
        return

    st.subheader("Original")
    st.audio(uploaded.getvalue(), format=uploaded.type)

    if not st.button("Enhance voice", type="primary"):
        return

    if not CHECKPOINT.exists():
        st.error(f"Checkpoint not found: {CHECKPOINT}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{Path(uploaded.name).stem}_enhanced.wav"

    try:
        with (
            st.spinner("Removing background noise…"),
            tempfile.TemporaryDirectory(prefix="voice-isolation-") as temp_dir,
        ):
            temp = Path(temp_dir)
            source = temp / Path(uploaded.name).name
            source.write_bytes(uploaded.getvalue())
            model_input = temp / "input.wav"
            convert_to_wav(source, model_input)
            result = enhance_file(
                input_path=model_input,
                checkpoint_path=CHECKPOINT,
                output_path=output_path,
                device_name="auto",
            )
    except (RuntimeError, subprocess.CalledProcessError) as error:
        st.error(str(error))
        return

    enhanced_bytes = output_path.read_bytes()
    st.success("Enhancement complete")
    st.subheader("Enhanced")
    st.audio(enhanced_bytes, format="audio/wav")

    columns = st.columns(3)
    columns[0].metric("Audio length", f"{result['audio_seconds']:.1f} s")
    columns[1].metric("Inference", f"{result['inference_ms']:.0f} ms")
    speed = 1.0 / result["real_time_factor"]
    columns[2].metric("Processing speed", f"{speed:.1f}× real time")

    st.download_button(
        "Download enhanced WAV",
        data=enhanced_bytes,
        file_name=output_path.name,
        mime="audio/wav",
    )


if __name__ == "__main__":
    main()
