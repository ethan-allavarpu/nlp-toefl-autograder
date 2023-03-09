from datasets import load_dataset, Dataset, Audio
import os
import pandas as pd
# setting path
import sys
sys.path.append('../../')
from espnet2.bin.tts_inference import Text2Speech


audio_files = [
    os.path.join(dp, f)
    for dp, dn, fn in os.walk(os.path.expanduser("data/speechocean/WAVE"))
    for f in fn
    if f.endswith(".WAV")
]


scores = pd.read_json(path_or_buf="data/speechocean/SCORES.json").T
scores = scores.reset_index().rename(columns={"index": "file_id"})

# get dict of speaker, file_name to file_id
speaker_dict = {
    int(x.split("/")[-1][:-4]): int(x.split("/")[-2].replace("SPEAKER", ""))
    for x in audio_files
}
audio_file_dict = {int(x.split("/")[-1][:-4]): x for x in audio_files}
# dicts to df
speaker_df = (
    pd.DataFrame.from_dict(speaker_dict, orient="index")
    .reset_index()
    .rename(columns={"index": "file_id", 0: "speaker_id"})
)
audio_file_df = (
    pd.DataFrame.from_dict(audio_file_dict, orient="index")
    .reset_index()
    .rename(columns={"index": "file_id", 0: "file_name"})
)

scores = scores.merge(speaker_df, on="file_id", how="inner")
scores = scores.merge(audio_file_df, on="file_id", how="inner")
scores.to_csv("data/speechocean/scores.csv", index=False)

words_series = scores.apply(
    lambda x: pd.DataFrame(
        x["words"], index=[x["file_id"] for i in range(len(x["words"]))]
    ),
    axis=1,
)
words_df = (
    pd.concat(words_series.values).reset_index().rename(columns={"index": "file_id"})
)
words_df["speaker_id"] = words_df["file_id"].map(speaker_dict)
words_df.to_csv("data/speechocean/words.csv", index=False)

# push huggingface dataset
audio_files = [f for f in audio_files if f in scores["file_name"].values]
ds = Dataset.from_dict(
    {**{"audio": scores["file_name"].values.tolist()}, **scores.to_dict("list")}
).cast_column("audio", Audio())

model = Text2Speech.from_pretrained("espnet/kan-bayashi_ljspeech_vits", "parallel_wavegan/csmsc_multi_band_melgan.v2",
    threshold=0.5,
    # Only for Tacotron 2
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    # Only for FastSpeech & FastSpeech2 & VITS
    speed_control_alpha=1.0,
    # Only for VITS
    noise_scale=0.333,
    noise_scale_dur=0.333)
# model.tts.fs = 16000
correct_speech = [model(t.lower())['wav'] for t in ds['text']]
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
ds = ds.add_column("correct_speech", [c.numpy() for c in correct_speech])
ds.push_to_hub("siegels/speechocean", token=os.environ.get("HUGGINGFACE_TOKEN"))
ds = ds.remove_columns("correct_speech")


from espnet2.bin.tts_inference import Text2Speech
from scipy.io.wavfile import write
resampled = [librosa.resample(j.numpy().T, orig_sr=22050, target_sr = 16000) for j in correct_speech]

model = Text2Speech.from_pretrained(
    model_tag="espnet/english_male_ryanspeech_tacotron",
        threshold=0.5,
    # Only for Tacotron 2
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    speed_control_alpha=0.5,
)
correct_speech = [model(t.lower())['wav'] for t in ds['text']]
librosa.resample(correct_speech[0].numpy().T, orig_sr=22050, target_sr = 16000)
write("test.wav", 22050, output['wav'].numpy())

