import numpy as np
import argparse
import sys

from pathlib import Path
from moviepy.editor import AudioFileClip
from collections import defaultdict


parser = argparse.ArgumentParser(description="Covert A-VB Labels to End2You format.")
parser.add_argument("--avb_path", type=str, help="Path to A-VB label file.")
parser.add_argument(
    "--task",
    type=str,
    default="low",
    help='A-VB Task. One of ["high", "two", "culture", "type"].',
)
parser.add_argument(
    "--save_path",
    type=str,
    default="./e2u_files_low",
    help="Path to save End2You conversion files.",
)

SR = 16000
WIN_LEN = 0.1
AUDIO_CHUNK = int(WIN_LEN * SR)


def convert_files(avb_path, task, save_path):
    label_path = avb_path / "labels" / f"{task}_info.csv"
    audio_files_path = avb_path / "audio" / "wav"

    data_info = np.loadtxt(str(label_path), dtype="<U200", delimiter=",")

    files, split, labels = data_info[1:, 0], data_info[1:, 1], data_info[1:, 2:]

    if task == "type":
        for i, l in enumerate(np.unique(labels)):
            idx = np.where(labels == l)
            labels[idx] = i - 1

    LABEL_CSV_HEADER = ",".join(["Timestamp"] + data_info[0][2:].tolist())

    Path(f"{save_path}/labels").mkdir(exist_ok=True, parents=True)

    input_files = defaultdict(list)
    for f, fsplit, l in zip(*[files, split, labels]):

        f = f.split("[")[1].split("]")[0]
        audio_file = audio_files_path / (f + ".wav")
        if not audio_file.exists():
            continue

        print(f"Writing csv file for: [{f}]")
        clip = AudioFileClip(str(audio_file), fps=SR)
        num_samples = int(SR * clip.duration // AUDIO_CHUNK)

        file_data = []
        for i in range(num_samples + 1):
            fl = ",".join(l) if "Test" not in fsplit else "-1"
            file_data.append([round(i * WIN_LEN, 2), fl])
        data_array = np.array(file_data)

        save_label_file_path = save_path / "labels" / (f + ".csv")
        np.savetxt(
            str(save_label_file_path),
            data_array,
            delimiter=",",
            header=LABEL_CSV_HEADER,
            fmt="%s",
        )

        input_files[fsplit.lower()].append(
            [str(audio_file), str(save_path / "labels" / (f + ".csv"))]
        )

    for partition, data in input_files.items():
        input_file_path = save_path / "labels" / f"{partition}_input_file.csv"
        np.savetxt(
            str(input_file_path),
            np.array(data),
            delimiter=",",
            fmt="%s",
            header="Filename,Label_File",
        )


if __name__ == "__main__":
    flags = sys.argv[1:]
    flags = vars(parser.parse_args(flags))
    avb_path, save_path = Path(flags["avb_path"]), Path(flags["save_path"])
    task = flags["task"]

    convert_files(avb_path, task, save_path)
