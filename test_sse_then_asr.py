import librosa
import pysndfile
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.enh_inference import SeparateSpeech

def init_asr():
    rate = 16000
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
    return speech2text, rate

def init_enh():
    rate = 8000
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
    return separate_speech, rate


speech2text, asr_rate = init_asr()
separate_speech, enh_rate = init_enh()
speech, rate = librosa.load("/home/adrien/Documents/test_voicecomms_kcorp.mp3", sr=enh_rate)

pysndfile.sndio.write('base_audio.wav', speech, rate=rate, format='wav', enc='pcm16')

# SEPARATE SPEECH
waves = separate_speech(speech[None, ...], fs=rate)

print(waves)

# DO TEXT RECOGNITION ON SEPARATED SPEECH
for i, wave in enumerate(waves):
    pysndfile.sndio.write(f"audio_wave_{i}.wav", wave.squeeze(), rate=rate, format='wav', enc='pcm16')
    speech, _ = librosa.load(f"audio_wave_{i}.wav", sr=asr_rate)
    
    nbests = speech2text(speech)
    text, *_ = nbests[0]

    print(f"Wave {i} detected text :\n{text}")
