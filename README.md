# Pseudo-Continuous Audio Transcription

This is an attempt at making Whisper transcribe audio on the fly. It uses `webrtcvad` to detect and segmentize audio that contains speech.

## Installation

Use the `./install` script to automatically install ffmpeg and portaudio-dev, set up a virtual python environment, and install the requirements openai-whisper, pyaudio, webrtcvad, and pyAudioAnalysis.

## Run

Running the `./realtime` bash script will launch the transcription. This uses the system's default audio device and will attempt to use `/dev/shm` for temporary storage. Output is dumped to the console.

