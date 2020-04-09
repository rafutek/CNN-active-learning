#!/usr/bin/env python

import os
import argparse
from active_learning import *
from results import Results

def argument_parser(script_name, model_choices, dataset_choices, method_choices):
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(
                prog=script_name,
                description="This program allows to train different models of classification on"
                            " different datasets, with different active learning methods.")
    parser.add_argument('--models',type=str, default="VggNet",
            help="set the list of models (ex: 'VggNet,ResNeXt'). "
                "Possible models: "+str(model_choices))
    parser.add_argument('--datasets', type=str, default="cifar10",
            help="set the list of datasets (ex: 'cifar10,cifar100')."
                "Possible datasets: "+str(dataset_choices))
    parser.add_argument('--methods', type=str, default="uncertainty_sampling",
            help="set the list of active learning selection methods "
                "(ex: 'random,uncertainty_sampling')."
                " Possible methods: "+str(method_choices))
    parser.add_argument('--Ks', type=str, default="200",
            help="set the list of number of samples selected and added "
                "to the training set (ex: '20,40,200').")
    parser.add_argument('--order', type=str, default="model,dataset,method,k",
            help="set the order for results plot (ex: 'model,dataset,method,k')")
    parser.add_argument('--split-level', type=int, default=0,
            help="set the level where to split the results in multiple plots "
                "(ex: 1 will show the results, given the above order, of each model)")
    parser.add_argument('--num-trainings', type=int, default=5,
            help='set the number of active learning loops')
    parser.add_argument('--batch-size', type=int, default=20,
            help='set the size of the training batch')
    parser.add_argument('--num-epochs', type=int, default=10,
            help='set the number of epochs per training')
    parser.add_argument('--lr', type=float, default=0.001,
            help='Learning rate')
    return parser.parse_args()

def check_list_arg(arg, choices=None):
    arg_list = [item for item in arg.split(',')]
    if choices is not None:
        args_in_choices = all(item in choices for item in arg_list)
        if not args_in_choices:
            raise ValueError("The list '"+arg+"' must contain "
                            "the possible choices: "+str(choices))
    return arg_list

def check_full_list_arg(arg, choices=None):
    arg_list = [item for item in arg.split(',')]
    if choices is not None:
        all_choices_in_list = all(item in arg_list for item in choices)
        if not all_choices_in_list :
            raise ValueError("The list '"+arg+"' must contain "
                            "all the possible choices: "+str(choices))
    return arg_list

if __name__ == "__main__":
    model_choices = ['VggNet','ResNeXt','SENet']
    dataset_choices = ['cifar10','cifar100','audioset']
    methods_choices = ['random','uncertainty_sampling','margin_sampling']
    order_choices = ['model','dataset','method', 'k']

    args = argument_parser(os.path.basename(__file__), \
            model_choices, dataset_choices, methods_choices)
    models = check_list_arg(args.models,model_choices)
    datasets = check_list_arg(args.datasets,dataset_choices)
    methods = check_list_arg(args.methods,methods_choices)
    Ks = check_list_arg(args.Ks)
    order = check_full_list_arg(args.order,order_choices)

    split_level = args.split_level
    batch_size = args.batch_size
    num_trainings = args.num_trainings
    num_epochs = args.num_epochs
    learning_rate = args.lr

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

                    accuracies = active_learning(model, dataset,
                            method, k, num_trainings,
                            batch_size, num_epochs, learning_rate)
                    
                    dic_results[model][dataset][method][k] = accuracies

    res = Results(dic_results, models, datasets, methods, Ks)
    res.plot_results(order, split_level)
    quit()
