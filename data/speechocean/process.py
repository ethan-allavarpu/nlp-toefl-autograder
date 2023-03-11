from datasets import load_dataset, Dataset, Audio
import os
import pandas as pd
# setting path
import sys
sys.path.append('../../')
from espnet2.bin.tts_inference import Text2Speech
import json
from espnet2.bin.tts_inference import Text2Speech
import torch
from scipy.io.wavfile import write

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
#ds = ds.cast_column("audio", Audio(sampling_rate=16000))
# ds = ds.add_column("correct_speech", [c.numpy() for c in correct_speech])
ds.push_to_hub("siegels/speechocean", token=os.environ.get("HUGGINGFACE_TOKEN"))
ds = ds.remove_columns("correct_speech")

model = Text2Speech.from_pretrained(
    model_tag="espnet/english_male_ryanspeech_tacotron",
    vocoder_file='/Users/spencersiegel/Documents/Stanford/Winter 2023/nlp-toefl-autograder/data/speechocean/train_nodev_parallel_wavegan.v1.long/checkpoint-1000000steps.pkl'
    )

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
with torch.no_grad():
    correct_speech = [model(t.lower())['wav'] for t in ds['text']]
resampled = [librosa.resample(c.numpy().T, orig_sr=22050, target_sr = 16000) for c in correct_speech]
# output_dict = {ds['file_name'][idx]:r.tolist() for idx, r in enumerate(resampled)}
# json_object = json.dumps(output_dict)
 
# # Writing to json
# with open("correct_speech.json", "w") as outfile:
#     outfile.write(json_object)

data = np.array([np.pad(row, (0, 100000-len(row))) for row in resampled])
pd.DataFrame(data).to_csv('correct_speech.csv', index=False)

# write("test.wav", 22050, output['wav'].numpy())

