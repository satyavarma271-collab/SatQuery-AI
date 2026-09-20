import json
import os


def load_rsvqa(json_file):

    if not os.path.exists(json_file):
        raise FileNotFoundError(
            f"Dataset not found: {json_file}"
        )

    with open(
        json_file,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    return data


def get_sample(data, index=0):

    if len(data) == 0:
        return None

    return data[index]
