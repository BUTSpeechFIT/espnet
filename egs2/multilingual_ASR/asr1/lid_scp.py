import argparse
import subprocess
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dump_dirs",'--names-list', nargs='+', default=[], help="Should be paths to train and validation dset directories")
parser.add_argument("--bpedir", type=str, required=True)

args = parser.parse_args()

# read ID to language mapping into dictionary
with open(f"{args.bpedir}/id2lang.txt", "r") as f:
    mappings = f.readlines()
lang2id = {}
for mapping in mappings:
    id, lang = mapping.replace("\n", "").split(" ")
    lang2id[lang] = id


for dir in args.dump_dirs:
    print(f"Creating language ID .scp file for {dir}")
    # copy file containing id to language mapping into given selected dump directories
    with open(f"{dir}/id2lang.txt", "w") as f:
        f.writelines(mappings)

    # read feats.scp to parse and create lid.scp
    with open(f"{dir}/feats.scp", "r") as f:
        lines = f.readlines()

    # create .scp file containing data ID and language ID
    with open(f"{dir}/lid.scp", "w") as f:
        for line in tqdm.tqdm(lines):
            id, _ = line.split(" ")
            lang = id.split('common_voice_')[1].split("_")[0] # parse language from ark path
            f.write(f"{id} {lang2id[lang]}\n")


