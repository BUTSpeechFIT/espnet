import argparse
import os
import yaml
from sacremoses import MosesPunctNormalizer, MosesTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_set", type=str, default="train", help="train set directory"
    )
    parser.add_argument(
        "--test_set", type=str, default="test", help="test set directory"
    )
    parser.add_argument(
        "--valid_set", type=str, default="valid", help="valid set directory"
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="data",
        help="path to where prepared data will be stored",
    )
    parser.add_argument(
        "--dset_path",
        type=str,
        default=None,
        required=True,
        help="path to original raw data directory (eg: IWSLT2022_Tamasheq_data/taq_fra_clean/)",
    )

    return parser.parse_args()


# prepare files in data dir
def prepare_files_for_dataset(dset, args):
    if not os.path.exists(os.path.join(args.datadir, dset)):
        print("creating directories: ", os.path.join(args.datadir, dset))
        os.makedirs(os.path.join(args.datadir, dset))

    wavs_path = os.path.join(args.dset_path, dset, "wav")
    # create files from .yaml

    utt_ids = []
    yaml_data = []
    utt_id_to_dict = {}

    text_lines = []
    with open(
        os.path.join(args.dset_path, dset, "txt", dset + ".fra"), "r"
    ) as fpr:
        for line in fpr:
            text_lines.append(line.strip())

    with open(
        os.path.join(args.dset_path, dset, "txt", dset + ".yaml"), "r"
    ) as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)

    assert len(yaml_data) == len(text_lines), "Number of rows {:d} in yaml file do not match number of lines ({:d}) in text".format(len(yaml_data), len(text_lines))

    for i, line_dict in enumerate(yaml_data):

        spk_id = line_dict['speaker_id']
        wav_id = line_dict['wav']
        offset = line_dict['offset']
        dur = line_dict['duration']

        utt_text = text_lines[i]

        utt_ids.append(wav_id)
        utt_id_to_dict[wav_id] = {
            'spk_id': spk_id,
            'offset': offset,
            'duration': dur,
            'seg': f"{wav_id} {wav_id} {offset} {dur}",
            "wav_scp": f"{wav_id} " + os.path.join(wavs_path, f"{wav_id}.wav"),
            "utt2spk": f"{wav_id} {wav_id}",
            "text": f"{wav_id} {utt_text}"
        }

    sorted_utt_ids = utt_ids
    # print("  sorting utt_ids")

    with open(
        os.path.join(args.datadir, dset, "segments"), "w"
    ) as segment_file, open(
        os.path.join(args.datadir, dset, "utt2spk"), "w"
    ) as utt2spk_file, open(
        os.path.join(args.datadir, dset, "wav.scp"), "w"
    ) as wav_scp_file, open(
        os.path.join(args.datadir, dset, "text"), "w", encoding="utf-8"
    ) as text_file:

        for utt_id in sorted_utt_ids:
            segment_file.write(utt_id_to_dict[utt_id]['seg'] + "\n")
            utt2spk_file.write(utt_id_to_dict[utt_id]['utt2spk'] + "\n")
            wav_scp_file.write(utt_id_to_dict[utt_id]['wav_scp'] + "\n")
            text_file.write(utt_id_to_dict[utt_id]['text'] + "\n")


def prepare_text_data(dset, args):
    """
    # normalize text
    # 1. normalize punctuation
    # 2. tokenizer
    """

    src_path = os.path.join(args.datadir, dset, "text")
    dst_path = os.path.join(args.datadir, dset)

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    pure_txt = []
    utt_ids = []

    with open(src_path) as data_text:
        for line in data_text.readlines():
            utt_id, utt_pure_txt = line.strip("\n").split(" ", maxsplit=1)
            utt_ids.append(utt_id)
            pure_txt.append(utt_pure_txt)

    # 1. pure text files
    print("  Normalizing text data...")
    # pure text
    with open(os.path.join(dst_path, "pure_text"), "w") as f:
        f.write("\n".join(pure_txt) + "\n")

    # normalize punctuation
    mpn = MosesPunctNormalizer(lang="fr")
    txt_norm = []
    for text in pure_txt:
        txt_norm.append(mpn.normalize(text))
    assert len(pure_txt) == len(txt_norm)
    with open(os.path.join(dst_path, "fra.norm"), "w") as f:
        f.write("\n".join(txt_norm) + "\n")

    # tokenized
    mt = MosesTokenizer(lang="fr")
    txt_norm_tc_tok = []
    for line in txt_norm:
        txt_norm_tc_tok.append(mt.tokenize(line, return_str=True))

    with open(os.path.join(dst_path, "fra.norm.tc.tok"), "w") as f:
        f.write("\n".join(txt_norm_tc_tok) + "\n")

    # 2. "utt_id sentence" files
    with open(os.path.join(dst_path, "text.tc.fra"), "w") as f:
        for utt_id, text in zip(utt_ids, pure_txt):
            f.write(f"{utt_id} {text}\n")

    with open(os.path.join(dst_path, "text.tc.fra.tok"), "w") as f:
        for utt_id, text in zip(utt_ids, txt_norm_tc_tok):
            f.write(f"{utt_id} {text}\n")


if __name__ == "__main__":
    args = parse_args()
    print(
        "Set paths for data preparation",
        args.train_set,
        args.test_set,
        args.valid_set,
        args.datadir,
    )

    for dset in [args.train_set, args.test_set, args.valid_set]:
        print(f"Preparing {dset} set...")
        prepare_files_for_dataset(dset, args)
        prepare_text_data(dset, args)
