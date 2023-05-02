import sys
from pathlib import Path
import numpy as np
import pandas as pd
import unicodedata
from time import time
from subprocess import run

WIDTH = 2048
HEIGHT = 128
FONT_SIZE = 32
DATA_DIR = "./out"
GRAVITY = "West"
TEXT_FILE = "data/data_wikitext_103.txt"

fonts = {
    "architects_daughther": ["data/fonts/Architects_Daughter/ArchitectsDaughter-Regular.ttf"],
    "caveat": ["data/fonts/Caveat/Caveat-Regular.ttf"],
    "dancing_script": ["data/fonts/Dancing_Script/DancingScript-Regular.ttf"],
    "indie_flower": ["data/fonts/Indie_Flower/IndieFlower-Regular.ttf"],
    "lobster": ["data/fonts/Lobster/Lobster-Regular.ttf"],
    "open_sans": ["data/fonts/Open_Sans/OpenSans-Regular.ttf", "data/fonts/Open_Sans/OpenSans-Italic.ttf"],
    "roboto_mono": ["data/fonts/Roboto_Mono/RobotoMono-Regular.ttf", "data/fonts/Roboto_Mono/RobotoMono-Italic.ttf"],
    "sacramento": ["data/fonts/Sacramento/Sacramento-Regular.ttf"],
    "shadows_into_light": ["data/fonts/Shadows_Into_Light/ShadowsIntoLight-Regular.ttf"],
    "times_new_roman": ["data/fonts/Times_New_Roman/times_new_roman_regular.ttf", "data/fonts/Times_New_Roman/times_new_roman_italic.ttf"],
}

font_types = list(fonts.keys())


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def split_lines(text, max_chars=128):
    counter = 0
    last_occurrence = text.rfind(' ')
    next_counter = text[0:max_chars].rfind(' ')

    parts = [text[counter:next_counter]]
    counter = next_counter
    while True:
        if counter >= last_occurrence:
            break
        next_counter = text[counter:max_chars+counter].rfind(' ') + counter
        parts.append(text[counter:next_counter])
        counter = next_counter

    return parts


def generate(line, out_dir, max_chars=128):
    max_chars = max_chars - 2  # To accommodate SOS_token and EOS_token
    line = unicodeToAscii(line)
    which_font = font_types[np.random.randint(len(fonts))]

    if which_font == "roboto_mono":
        font_size_l = 25
    else:
        font_size_l = FONT_SIZE

    if len(fonts[which_font]) == 2:
        font = fonts[which_font][np.random.randint(2)]
    else:
        font = fonts[which_font][0]

    prefix = font.split("/")[1]

    data = []
    parts = split_lines(line, max_chars)
    for index, p in enumerate(parts):
        p = p.replace('"', r'\"')
        print(p)
        filename = str(int(time())) + "_" + str(index) + ".png"
        full_filename = f"{prefix}_{filename}"
        data.append((full_filename, which_font, p))
        p = "\ " + p
        command = f'magick -gravity West -size {WIDTH}x{HEIGHT} -pointsize {font_size_l} -font "{font}" label:"{p}" -channel Black -bordercolor White {out_dir}/{prefix}_{filename}'
        run(command, shell=True)

    return data


def main(out_folder):
    out_dir = Path(DATA_DIR)/out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    f = open(TEXT_FILE, 'r')
    data = f.readlines()
    result = []

    for line in data[:]:
        if line.strip() == "" or line.strip().startswith("="):
            continue
        if not all(ord(char) < 128 for char in line):
            continue
        image_info = generate(line, out_dir)
        if image_info is not None:
            result.extend(image_info)

    df = pd.DataFrame(result, columns=["image", "font_type", "line"])
    df.to_csv(out_dir/"dataset.csv", index=False)

    f.close()


if __name__ == "__main__":
    out_folder = sys.argv[1]
    main(out_folder)
