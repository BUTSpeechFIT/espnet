#!/usr/bin/env python3
# author: Santosh Kesiraju


import os
import sys
import argparse
import glob


def main(args):
    """main method"""

    lang = args.lang

    data_dir = os.path.realpath(args.data_dir)

    for set_name in args.set_names:

        os.makedirs(args.data_dir + f"/{set_name}", exist_ok=True)

        wav_files = glob.glob(args.orig_data_dir + f"{set_name}/wav/*.wav")
        wav_dir = os.path.realpath(args.orig_data_dir + f"{set_name}/")

        seg_file = os.path.join(args.orig_data_dir, f"{set_name}/stamped.tsv")
        txt_file = os.path.join(args.orig_data_dir, f"{set_name}/txt/{set_name}.{lang}")

        utt_ids = []
        segments = []
        wav_scps = []
        utt2spk = []
        utt2langs = []
        with open(seg_file, "r", encoding="utf-8") as fpr:
            for line in fpr:
                parts = line.strip().split()
                utt_id = parts[0].split("/")[1].split(".")[0]
                utt_ids.append(utt_id)
                segments.append(f"{utt_id} {utt_id} {parts[1]} {parts[2]}")
                utt2spk.append(f"{utt_id} {utt_id}")
                utt2langs.append(f"{utt_id} {lang}")

                wav_file = os.path.join(wav_dir, f"{parts[0]}")
                if os.path.exists(wav_file):
                    ffmpeg_cmd = f"ffmpeg -i {wav_file} -f wav -ar 16000 -ab 16 -ac 1 - | "
                    wav_scps.append(f"{utt_id} {ffmpeg_cmd}")
                else:
                    print(wav_file, "not found")
                    sys.exit()

        text = []
        i = 0
        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as fpr:
                for line in fpr:
                    line = line.strip()
                    text.append(f"{utt_ids[i]} {line}")
                    i += 1

        print(set_name, len(wav_files), len(utt_ids), 'text:', len(text))

        out_seg_file = os.path.join(args.data_dir, f"{set_name}/segments")
        assert not os.path.exists(out_seg_file), f"{out_seg_file} already exists"
        with open(out_seg_file, "w", encoding="utf-8") as fpw:
            fpw.write("\n".join(segments) + "\n")

        if len(text):
            out_txt_file = os.path.join(data_dir, f"{set_name}/text.{lang}")
            assert not os.path.exists(out_txt_file), f"{out_txt_file} already exists"
            ln_txt_file = os.path.join(data_dir, f"{set_name}/text")
            with open(out_txt_file, "w", encoding="utf-8") as fpw:
                fpw.write("\n".join(text) + "\n")
            os.system(f"ln -sv {out_txt_file} {ln_txt_file}")

        out_wav_scp = os.path.join(args.data_dir, f"{set_name}/wav.scp")
        assert not os.path.exists(out_wav_scp), f"{out_wav_scp} already exists"
        with open(out_wav_scp, "w", encoding="utf-8") as fpw:
            fpw.write("\n".join(wav_scps) + "\n")

        utt2spk_file = os.path.join(data_dir, f"{set_name}/utt2spk")
        with open(utt2spk_file, "w") as fpw:
            fpw.write("\n".join(utt2spk) + "\n")

        utt2lang_file = os.path.join(data_dir, f"{set_name}/utt2lang")
        lid_scp_file = os.path.join(data_dir, f"{set_name}/lid.scp")
        utt2cat_file = os.path.join(data_dir, f"{set_name}/utt2category")
        with open(utt2lang_file, "w") as fpw:
            fpw.write("\n".join(utt2langs) + "\n")
        os.system(f"ln -sv {utt2lang_file} {lid_scp_file}")
        os.system(f"ln -sv {utt2lang_file} {utt2cat_file}")

    print("utils/fix_data_dir.sh ", args.data_dir, args.set_names)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("orig_data_dir", help="path to original data dir from IWSLT'23")
    parser.add_argument("data_dir", help="path to out data dir (eg: data/)")
    parser.add_argument("--lang", type=str, required=True, help="langauge code")
    parser.add_argument(
        "--set_names", nargs="+", default=[], required=True, help="set names to process (eg: train, dev, test)"
    )
    args = parser.parse_args()

    main(args)
