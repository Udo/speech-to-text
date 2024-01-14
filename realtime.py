import pyaudio
import wave
import whisper
import threading
import os
import collections
import webrtcvad
import time
import sys
import torch
from pydub import AudioSegment

CHANNELS = 1
CHUNK_DURATION_MS = 30
FORMAT = pyaudio.paInt16
MAX_SILENCE_DURATION_MS = 500
MIN_SILENCE_DURATION_MS = 10
MAX_SPEECH_DURATION_MS = 10000
MIN_SPEECH_DURATION_MS = 1000
RATE = 16000
ONE_GB = 1024*1024*1024

MIN_SPEECH_CHUNKS = int(MIN_SPEECH_DURATION_MS / CHUNK_DURATION_MS)
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)

#MODEL_NAME = "base"
#MODEL_NAME = "tiny"
#MODEL_NAME = "small"
#MODEL_NAME = "medium"
#MODEL_NAME = "medium.en"
MODEL_NAME = "large-v3"

# Initialize VAD
vad = webrtcvad.Vad(1)  # Level of aggressiveness from 0 to 3
temp_dir = "/tmp"

print("# loading model...")
model = whisper.load_model(MODEL_NAME)

if torch.cuda.is_available():
	print(f"# GPU/CUDA memory alloc {(torch.cuda.memory_allocated()/ONE_GB):.1f}GB, reserved {(torch.cuda.memory_reserved()/ONE_GB):.1f}GB")

import ctypes
from contextlib import contextmanager

# Define a context manager to suppress C-level stderr output
@contextmanager
def suppress_c_stderr():
	original_stderr_fd = os.dup(2)  # Duplicate the file descriptor for stderr
	try:
		devnull = os.open(os.devnull, os.O_WRONLY)
		os.dup2(devnull, 2)  # Replace stderr with devnull
		yield
	finally:
		os.dup2(original_stderr_fd, 2)  # Restore the original stderr
		os.close(devnull)
		os.close(original_stderr_fd)

with suppress_c_stderr():
	audio = pyaudio.PyAudio()

def list_audio_devices():
	info = audio.get_host_api_info_by_index(0)
	num_devices = info.get('deviceCount')

	# Scan for available devices and print them
	for i in range(0, num_devices):
		if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
			device_name = audio.get_device_info_by_host_api_device_index(0, i).get('name')
			print(f"# Audio device id {i} - {device_name}")

	# Get default input device information
	default_device_index = audio.get_default_input_device_info().get('index')
	default_device_name = audio.get_device_info_by_host_api_device_index(0, default_device_index).get('name')
	print(f"# default: {default_device_name} (id {default_device_index})")

list_audio_devices()

def record_audio():
	stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

	print("# listening...")

	while True:
		speech_detected = False
		frames = []
		silence_counter = 0
		speech_duration = 0
		current_silence_duration_ms = MAX_SILENCE_DURATION_MS

		while True:
			frame = stream.read(CHUNK_SIZE)
			is_speech = vad.is_speech(frame, RATE)

			if is_speech:
				silence_counter = 0
				speech_duration += 1
				total_speech_duration_ms = speech_duration * CHUNK_DURATION_MS
				if not speech_detected:
					print("segment start")
					speech_detected = True
				if total_speech_duration_ms > MAX_SPEECH_DURATION_MS:
					current_silence_duration_ms = max(MIN_SILENCE_DURATION_MS, current_silence_duration_ms - 1)

			else:
				silence_counter += 1

			if speech_detected:
				frames.append(frame)

			if speech_detected and silence_counter >= current_silence_duration_ms / CHUNK_DURATION_MS:
				break

		duration_in_seconds = float(speech_duration * CHUNK_DURATION_MS / 1000)

		if speech_duration >= MIN_SPEECH_CHUNKS:
			print(f"segment end, {float(duration_in_seconds):.1f}s, silence_threshold {int(current_silence_duration_ms)} ms")
			filename = f"{temp_dir}/temp_audio_{int(time.time())}.wav"
			with wave.open(filename, 'wb') as wf:
				wf.setnchannels(CHANNELS)
				wf.setsampwidth(audio.get_sample_size(FORMAT))
				wf.setframerate(RATE)
				wf.writeframes(b''.join(frames))
			threading.Thread(target=transcribe_audio, args=(filename,duration_in_seconds,)).start()
		else:
			print(f"segment rejected, {float(duration_in_seconds)}s")

def average_segment_metrics(result):
	segments = result.get('segments', [])
	if not segments:
		return None

	total_avg_logprob = 0.0
	total_no_speech_prob = 0.0

	for segment in segments:
		avg_logprob = segment.get('avg_logprob', 0.0)
		no_speech_prob = segment.get('no_speech_prob', 0.0)

		total_avg_logprob += avg_logprob
		total_no_speech_prob += no_speech_prob

	num_segments = len(segments)
	if num_segments > 0:
		result["avg_logprob"] = total_avg_logprob / num_segments
		result["no_speech_prob"] = total_no_speech_prob / num_segments

	return result

def transcribe_audio(filename, duration):
	try:
		if os.path.exists(filename):
			start_time = time.time()
			result = model.transcribe(filename)
			end_time = time.time()
			result["duration"] = f"{duration:.1f}"
			result["trn_time"] = f"{end_time-start_time:.1f}"
			result["speed"] = f"{duration/(end_time-start_time):.1f}"
			print(f"transcript", average_segment_metrics(result))
	except Exception as e:
		print(f"! Error during transcription: {e}")
	finally:
		try:
			if os.path.exists(filename):
				os.remove(filename)
		except Exception as e:
			print(f"! Error removing file {filename}: {e}")

if __name__ == "__main__":
	temp_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
	recording_thread = threading.Thread(target=record_audio, daemon=True)
	recording_thread.start()
	recording_thread.join()
