import os
import json
import argparse

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", type=float, default=0.01, help="the lowest face quality")
    parser.add_argument("--lsec", type=float, default=0.01, help="the lowest sync confidence")
    parser.add_argument("--lsed", type=float, default=20.0, help="the highest sync distance")
    parser.add_argument("--cred", type=float, default=0.0, help="the lowest emotion credibility")
    args = parser.parse_args()
    json_dir = "filters/q_0_0_c_0_0_d_20_0_e_0_0"
    save_dir = f"filters/q_{'_'.join(str(args.quality).split('.'))}_c_{'_'.join(str(args.lsec).split('.'))}_d_{'_'.join(str(args.lsed).split('.'))}_e_{'_'.join(str(args.cred).split('.'))}"
    languages = os.listdir(json_dir)
    for language in tqdm(languages):
        os.makedirs(f"{save_dir}/{language}",exist_ok=True)
        jsonnames = os.listdir(f"{json_dir}/{language}")
        for jsonname in tqdm(jsonnames):
            with open(f"{json_dir}/{language}/{jsonname}") as fp:
                total_dic = json.load(fp)
            new_dic = {}
            for vid_id, vid_meta in total_dic.items():
                if vid_meta["quality"] >= args.quality and vid_meta["LSE-C"] >= args.lsec and vid_meta["LSE-D"] <= args.lsed and vid_meta["credibility"] >= args.cred and not vid_meta["noface_frame"]:
                    new_dic[vid_id] = vid_meta
            if not new_dic:
                continue
            with open(f"{save_dir}/{language}/{jsonname}","w") as fp:
                json.dump(new_dic,fp,indent=4)