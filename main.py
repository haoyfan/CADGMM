#! python3
import os

import argparse
import importlib
import logging

# Logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s @%(name)s] %(message)s', datefmt='%H:%M:%S'))
# logging.basicConfig(level=logging.DEBUG, handlers=[console])
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logger = logging.getLogger("AnomalyDetection")


def run(args):

    has_effect = False

    if args.model and args.dataset and args.split:
        try:

            mod_name = "{}.{}".format(args.model, args.split)

            logger.info("Running script at {}".format(mod_name))

            mod = importlib.import_module(mod_name)

            mod.run(args)

        except Exception as e:
            logger.exception(e)
            logger.error("Uhoh, the script halted with an error.")
    else:
        if not has_effect:
            logger.error("Script halted without any effect. To run code, use command:\npython3 main.py <example name> {train, test}")

def path(d):
    try:
        assert os.path.isdir(d)
        return d
    except Exception as e:
        raise argparse.ArgumentTypeError("Example {} cannot be located.".format(d))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run examples from the DL 2.0 Anomadaly Detector.')
    parser.add_argument('--model', nargs="?", default='cadgmm', type=path,
                        help='the folder name of the example you want to run e.g gan or bigan')
    parser.add_argument('--dataset', nargs="?", default='arrhythmia',
                        choices=['kdd', 'arrhythmia''satellite'],
                        help='the name of the dataset you want to run the exp eriments on')
    parser.add_argument('--split', nargs="?", default='run', choices=['run'])
    parser.add_argument('--gpu', nargs="?", type=int, default=0, help='GPU device id')
    parser.add_argument('--v', nargs="?", type=float, default=0.00, help='0=<v and v<=1, 1%~5% ')

    # validation
    parser.add_argument('--it_e_val', nargs="?", default=4, type=int, help='interval of epochs to test')

    #anomaly
    parser.add_argument('--label', nargs="?", type=int, default=0, help='anomalous label for the experiment')
    parser.add_argument('--d', nargs="?", type=int, default=1, help='degree for the L norm')
    parser.add_argument('--rd', nargs="?", type=int, default=0,  help='random_seed')
    parser.add_argument('--enable_early_stop', action='store_true', help='enable early_stopping')
    parser.add_argument('--w', nargs="?", type=float, default=0.1, help='weight for AnoGAN')

    # args for dagmm
    parser.add_argument('--nb_epochs', nargs="?", default=-1, type=int, help='number of epochs you want to train the dataset on')
    parser.add_argument('--K', nargs="?", type=float, default=-1, help='number of mixtures in GMM')
    parser.add_argument('--KNN', nargs="?", type=int, default=-1, help='K value for K-NN')
    parser.add_argument('--l1', nargs="?", type=float, default=-1, help='weight of the energy in DAGMM')
    parser.add_argument('--l2', nargs="?", type=float, default=-1, help='weight of the penalty of diag term in DAGMM')
    parser.add_argument('--l3', nargs="?", type=float, default=-1, help='weight of the penalty of diag term in DAGMM')
    parser.add_argument('--a', nargs="?", type=float, default=-1, help='weight of the penalty of diag term in DAGMM')
    parser.add_argument('--b', nargs="?", type=float, default=-1, help='weight of the penalty of diag term in DAGMM')


    run(parser.parse_args())
