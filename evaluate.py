from run import Runner
import sys
from argparse import ArgumentParser

def evaluate(config_name, gpu_id, saved_suffix, dataset):
    runner = Runner(config_name, gpu_id, dataset)
    model = runner.initialize_model(saved_suffix)

    examples_train, examples_dev, examples_test = runner.data.get_tensor_examples()
    stored_info = runner.data.get_stored_info()

    # runner.evaluate(model, examples_dev, stored_info, 0, official=True, conll_path=runner.config['conll_eval_path'])  # Eval dev
    # print('=================================')
    runner.evaluate(model, examples_test, stored_info, 0, official=True, conll_path=runner.config['conll_test_path'])  # Eval test


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", default="train_spanbert_large_ml0_d2")
    parser.add_argument("--checkpoint")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--dataset", default="ontonotes", help="Select from ['ontonotes', 'ontogum']")
    args = parser.parse_args()

    evaluate(args.config, args.gpu, args.checkpoint, args.dataset)
