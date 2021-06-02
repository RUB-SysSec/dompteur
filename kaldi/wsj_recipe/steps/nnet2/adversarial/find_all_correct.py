import sys
import os
from pathlib import Path
import json

def main():

    if len(sys.argv) < 5:
        print("Wrong or not enough arguments")
        sys.exit()

    data_name = sys.argv[1]
    target_dir = sys.argv[2]
    itr = int(sys.argv[3])
    dir_name = sys.argv[4]

    # num utterance
    print("defined for experiments: " + data_name)
    print("defined for model: " + dir_name)

    root_dir = "./"
    # result dir
    result_dir = root_dir + "exp/" + dir_name +"/decode_" + data_name + "/scoring_kaldi/wer_details/per_utt"
    itr_dir = root_dir + "exp/" + dir_name + "/adversarial_" + target_dir + "/scoring_kaldi/wer_details/utt_itr"

    utt = []
    if os.path.isfile(itr_dir):
        with open(itr_dir) as f:
            for line in f:
                line = line.split()
                utt.append(line[0])


    with open(itr_dir, "a") as write_f:

        with open(result_dir) as f:
            for line in f:
                line = line.split()

                if line[1] == "#csid":
                    if int(line[3]) == 0 and int(line[4]) == 0 and int(line[5]) == 0:
                        if not line[0] in utt:
                            write_f.write("{} {}\n".format(line[0], str(itr)))

    best_wer_file = Path(root_dir, "exp", dir_name, "decode_" + data_name, "scoring_kaldi", "best_wer").as_posix()
    with open(best_wer_file) as f:
        best_wer = f.readline().strip()
        print("[+] %s" % best_wer)

    results_file = Path(root_dir, "exp", dir_name, data_name + '_wer.json').as_posix()
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f) 
    else: 
        results = []
    results.append(best_wer)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()