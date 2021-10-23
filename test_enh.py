import librosa
import pysndfile
from espnet2.bin.enh_inference import SeparateSpeech

fs = 8000

separate_speech = SeparateSpeech.from_pretrained(
    "Chenda Li/wsj0_2mix_enh_train_enh_conv_tasnet_raw_valid.si_snr.ave",
    # for segment-wise process on long speech
    segment_size=2.4,
    hop_size=0.8,
    normalize_segment_scale=False,
    show_progressbar=True,
    ref_channel=None,
    normalize_output_wav=True,
    device="cuda:0",
)
# Confirm the sampling rate is equal to that of the training corpus.
# If not, you need to resample the audio data before inputting to speech2text
speech, rate = librosa.load("/home/adrien/Documents/test_voicecomms_kcorp.mp3", sr=fs)
waves = separate_speech(speech[None, ...], fs=rate)

for i, wave in enumerate(waves):
    pysndfile.sndio.write(f"audio_sse_{i}.wav", wave.squeeze(), rate=rate, format='wav', enc='pcm16')
