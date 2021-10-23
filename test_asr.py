import librosa
import pysndfile
from espnet2.bin.asr_inference import Speech2Text

fs = 16000

speech2text = Speech2Text.from_pretrained(
    "Shinji Watanabe/spgispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_unnorm_bpe5000_valid.acc.ave",
    # Decoding parameters are not included in the model file
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=20,
    ctc_weight=0.3,
    lm_weight=0.5,
    penalty=0.0,
    nbest=1,
    device="cuda:0",
)

speech, rate = librosa.load("/home/adrien/Documents/test_voicecomms_kcorp.mp3", sr=fs)

pysndfile.sndio.write('audio_asr.wav', speech, rate=rate, format='wav', enc='pcm16')

print(f"File rate : {rate}")

nbests = speech2text(speech)
text, *_ = nbests[0]

print(f"Detected text :\n{text}")
