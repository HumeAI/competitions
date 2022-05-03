"""
Copyright (c) 2022 Marco Jiralerspong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

(MIT License (https://opensource.org/licenses/MIT) )
"""

import pandas
import os
import soundfile
import torch.utils.data as tdata

class ICMLExVo(tdata.Dataset):
    def __init__(self, wav_path, csv_path, emotion="All", country="All", sample_rate=16000, transforms=None, check_data=False, query=None):
        super(ICMLExVo, self).__init__()

        self.wav_path = wav_path
        self.csv_path = csv_path
        self.emotion = emotion
        self.country = country
        self.wav_df = self.create_wav_dataframe()

        if query:
            self.wav_df = self.wav_df.query(query)
        self.sample_rate = sample_rate
        self.transforms = transforms
        self.check_data = check_data

    def create_wav_dataframe(self):
        df = pandas.read_csv(self.csv_path)
        if self.emotion != "All":
            if self.emotion not in df.columns:
                raise Exception("Invalid emotion")
            df = df[df[self.emotion] == 1]
        print(f"Found {len(df)} samples for emotion {self.emotion}.")

        if self.country != "All":
            df = df[df["Country"] == self.country]
            if df.empty:
                raise Exception("Invalid country")
        print(f"Of those, found {len(df)} samples from {self.country}.")

        # Make sure file_ids match file names by padding with 0s
        def parse_file_id(file_id):
            file_id = file_id.replace("[", "")
            file_id = file_id.replace("]", "")
            return file_id

        df["File_ID"] = df["File_ID"].apply(parse_file_id)
        df = df.reset_index()
        print(f"Training on {len(df)} samples.")
        return df

    def __getitem__(self, idx):
        file_path = os.path.join(self.wav_path, self.wav_df.iloc[idx]["File_ID"] + ".wav")
        data, sr = soundfile.read(file_path, dtype='float32')

        # Stereo to mono
        if data.ndim == 2:
            data = data.mean(axis=1)

        return data, self.sample_rate

    def __len__(self):
        return len(self.wav_df)