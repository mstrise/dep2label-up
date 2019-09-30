'''

'''

from argparse import ArgumentParser
import codecs
import os
import time
import sys
import uuid
from dep2label import decode_dependencies as decode_dependencies
from utils.data import Data
from importlib import reload

STATUS_TEST = "test"
STATUS_TRAIN = "train"

if __name__ == '__main__':

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--test", dest="test", help="Path to the input test file as sequences", required=True)
    arg_parser.add_argument("--gold", dest="gold_dependency",
                            help="Path to the gold file in CoNNL-X format with dependency trees")
    arg_parser.add_argument("--model", dest="model", help="Path to the model", required=True)
    arg_parser.add_argument("--status", dest="status", help="[train|test]", required=True)
    arg_parser.add_argument("--gpu", dest="gpu", help="[True|False]", default="False", required=False)
    arg_parser.add_argument("--output", help="output path for parsed dependency tree", dest="output_dependency")
    arg_parser.add_argument("--ncrfpp", dest="ncrfpp", help="Path to the NCRFpp repository", required=True)
    arg_parser.add_argument("--multitask", dest="multitask", default=False, action="store_true")

    args = arg_parser.parse_args()
    reload(sys)
    path_raw_dir = args.test
    path_name = args.model
    path_output = "/tmp/" + path_name.split("/")[-1] + ".output"
    path_tagger_log = "/tmp/" + path_name.split("/")[-1] + ".tagger.log"
    path_dset = path_name + ".dset"
    path_model = path_name + ".model"
    data = Data()
    data.load(path_dset)

    conf_str = """
    ### Decode ###
    status=decode
    """
    conf_str += "raw_dir=" + path_raw_dir + "\n"
    conf_str += "decode_dir=" + path_output + "\n"
    conf_str += "dset_dir=" + path_dset + "\n"
    conf_str += "load_model_dir=" + path_model + "\n"
    conf_str += "gpu=" + args.gpu + "\n"

    decode_fid = str(uuid.uuid4())
    decode_conf_file = codecs.open("/tmp/" + decode_fid, "w")
    decode_conf_file.write(conf_str)

    os.system("python " + args.ncrfpp + "/main.py --config " + decode_conf_file.name + " > " + path_tagger_log)
    log_lines = codecs.open(path_tagger_log).readlines()

    time_prediction = float([l for l in log_lines
                             if l.startswith("raw: time:")][0].split(",")[0].replace("raw: time:", "").replace("s", ""))

    output_content = codecs.open(path_output)
    start = time.time()
    decode_dependencies.decode(output_content, args.output_dependency)
    time_dependency = time.time() - start
    decode_dependencies.evaluate_dependencies(args.gold_dependency, args.output_dependency)
    gold_depen = codecs.open(args.gold_dependency, "r")
    total_nb = 0
    for line in gold_depen:
        if line == "\n":
            total_nb += 1

    total = time_dependency + time_prediction
    print("Sent/sec " + repr(round(total_nb / total, 2)))
    print("TOTAL TIME " + repr(round(total, 2)))
