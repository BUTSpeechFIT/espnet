import argparse
import subprocess

H2S = 3600

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--hours', type=int, required=True)

def fix_feats_file(args):
    limit = args.hours*H2S
    sec_tmp = 0.0
    tmp_file = f"{args.datadir}/utt_{args.hours}"
    string_tmp = ""

    with open(args.datadir+"/utt2dur", "r") as f:
        lines = f.readlines() 
    
    for line in lines:
        splitted = line.split(" ")
        if len(splitted) == 2:
            id, time = line.split(" ")
            time = time.replace("\n", "")
        else:
            id, time, _ = line.split(" ")

        sec_tmp += float(time)
        string_tmp += id+"\n"
        if sec_tmp > limit:
            print(sec_tmp/H2S)
            break
        
    with open(args.datadir+"/utt_"+str(args.hours), "w") as f:
        f.write(string_tmp)
    #feats_file = f"{args.datadir}/feats.scp" 

    #subprocess.run(f"grep -F -f {tmp_file} {feats_file} > {feats_file}", capture_output=True, shell=True)

if __name__=='__main__':
    args = parser.parse_args()
    if args.hours == 0:
        exit(0)
    fix_feats_file(args)




