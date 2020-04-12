#!/usr/bin/env python

import os
import argparse
from active_learning import *
from results import Results

def argument_parser(script_name, model_choices, dataset_choices, method_choices):
    """
    A parser to allow user to easily experiment different models along with datasets
    and differents parameters
    Args:
        script_name: string that will be used as script name
        model_choices: list of the possible models to use
        dataset_choices: list of the possible datasets to use
        method_choices: list of the possible selection methods to use
    Returns:
       The parsed arguments
    """
    parser = argparse.ArgumentParser(
                prog=script_name,
                description="This program allows to train different models of classification on"
                            " different datasets, with different active learning methods")
    parser.add_argument('--models',type=str, default="VggNet",
            help="set the list of models (ex: 'VggNet,ResNet'). "
                "Possible models: "+str(model_choices))
    parser.add_argument('--datasets', type=str, default="cifar10",
            help="set the list of datasets (ex: 'cifar10,cifar100')."
                "Possible datasets: "+str(dataset_choices))
    parser.add_argument('--pool-length', type=int, default=None,
            help='set the length of the pool set (train + validation)')
    parser.add_argument('--methods', type=str, default="least_confidence",
            help="set the list of active learning selection methods "
                "(ex: 'random,margin')."
                " Possible methods: "+str(method_choices))
    parser.add_argument('--Ks', type=str, default="200",
            help="set the list of number of samples selected and added "
                "to the training set (ex: '20,40,200').")
    parser.add_argument('--order', type=str, default="model,dataset,method,k",
            help="set the order for results plot (ex: 'model,dataset,method,k')")
    parser.add_argument('--split-level', type=int, default=0,
            help="set the level where to split the results in multiple plots "
                "(ex: 1 will show the results, given the above order, per model)")
    parser.add_argument('--num-trainings', type=int, default=5,
            help='set the number of active learning loops')
    parser.add_argument('--batch-size', type=int, default=20,
            help='set the size of the training batch')
    parser.add_argument('--num-epochs', type=int, default=10,
            help='set the number of epochs per training')
    parser.add_argument('--lr', type=float, default=0.001,
            help='Learning rate')
    parser.add_argument('--use-cuda', action='store_true',
            help='try to use the gpu for trainings')
    parser.add_argument('--skip-train', action='store_true',
            help="skip training and load previous results")
    return parser.parse_args()

def check_list_arg(arg, choices=None):
    """
    Function that split the string containing arguments
    and if choices are given, raise an error if an argument 
    is not in the choice list
    Args:
        arg: string containing arguments (ex: "cifar10,cifar100")
        choices: list of possible choices
    Returns:
        List of arguments (ex: ['cifar10','cifar100'])
    """
    arg_list = [item for item in arg.split(',')]
    if choices is not None:
        args_in_choices = all(item in choices for item in arg_list)
        if not args_in_choices:
            raise ValueError("The list '"+arg+"' must contain "
                            "the possible choices: "+str(choices))
    return arg_list

def check_full_list_arg(arg, choices=None):
    """
    Function that split the string containing arguments
    and if choices are given, raise an error if arg does
    not contain every possible choice
    Args:
        arg: string containing arguments (ex: "k,model,dataset,method")
        choices: list of choices that must be present in arg
    Returns:
        List of arguments (ex: ['k','model','dataset','method'])
    """
    arg_list = [item for item in arg.split(',')]
    if choices is not None:
        all_choices_in_list = all(item in arg_list for item in choices)
        if not all_choices_in_list :
            raise ValueError("The list '"+arg+"' must contain "
                            "all the possible choices: "+str(choices))
    return arg_list

if __name__ == "__main__":
    """
    Main program that launch an active learning experiment
    based on arguments passed by user
    """
    model_choices = ['VggNet','ResNet','AlexNet']
    dataset_choices = ['cifar10','cifar100','urbansound']
    methods_choices = ['random','least_confidence','margin', 'entropy']
    order_choices = ['model','dataset','method', 'k']

    # Get arguments from user
    args = argument_parser(os.path.basename(__file__), \
            model_choices, dataset_choices, methods_choices)

    # Check if the arguments are ok and set corresponding variables
    models = check_list_arg(args.models,model_choices)
    datasets = check_list_arg(args.datasets,dataset_choices)
    methods = check_list_arg(args.methods,methods_choices)
    Ks = check_list_arg(args.Ks)
    order = check_full_list_arg(args.order,order_choices)

    pool_length = args.pool_length
    split_level = args.split_level
    batch_size = args.batch_size
    num_trainings = args.num_trainings
    num_epochs = args.num_epochs
    learning_rate = args.lr
    use_cuda = args.use_cuda
    skip_train = args.skip_train

    if not skip_train:
       # Launch the experiment and save each result in a dictionary
        dic_results = {}
        for model in models:
            if model not in dic_results:
                dic_results[model] = {}
            for dataset in datasets:
                if dataset not in dic_results[model]:
                    dic_results[model][dataset] = {}
                for method in methods:
                    if method not in dic_results[model][dataset]:
                        dic_results[model][dataset][method] = {}
                    for k in Ks:
                        if k not in dic_results[model][dataset][method]:
                            dic_results[model][dataset][method][k]  = {}
    
                        accuracies = active_learning(model, dataset, pool_length,
                                method, k, num_trainings, batch_size,
                                num_epochs, learning_rate, use_cuda)
                        
                        dic_results[model][dataset][method][k] = accuracies

        # Save the results
        res = Results(dic_results)
        res.save_results()

    else:
        # Load the results
        res = Results()
        res.load_results()

    res.plot_results(order, split_level)
