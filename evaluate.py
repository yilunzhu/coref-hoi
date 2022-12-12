import os.path

from run import Runner
import sys
from argparse import ArgumentParser

def evaluate(config_name, gpu_id, saved_suffix, dataset, conll_test_path):
    runner = Runner(config_name, gpu_id, dataset)
    model = runner.initialize_model(saved_suffix)

    _, _, examples_test = runner.data.get_tensor_examples()
    stored_info = runner.data.get_stored_info()

    # runner.evaluate(model, examples_dev, stored_info, 0, official=True, conll_path=runner.config['conll_eval_path'])  # Eval dev
    # print('=================================')
    runner.evaluate(model, examples_test, stored_info, 0, official=True, conll_path=conll_test_path)  # Eval test


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", default="train_mtl")
    parser.add_argument("--checkpoint")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--dataset", default="ontonotes", help="Select from ['ontonotes', 'ontogum', 'wikicoref']")
    # parser.add_argument("--conll_path", default="./data/ontogum/test.gum.english.v4_gold_conll")
    args = parser.parse_args()

    if args.dataset == "ontogum":
        conll_path = "./data/ontogum/test.gum.english.v4_gold_conll"
    elif args.dataset == "ontonotes":
        conll_path = "./data/test.english.v4_gold_conll"
    elif args.dataset == "wikicoref":
        conll_path = "./data/wikicoref/wikicoref.v4_gold_conll"
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}")

    evaluate(args.config, args.gpu, args.checkpoint, args.dataset, conll_path)
