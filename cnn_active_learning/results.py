import matplotlib.pyplot as plt
import pickle

class Results(object):
    """
    Class to show the experiment result depending
    on user settings
    """
    results_filename = "./data/results.pkl"
    
    def __init__(self, results:dict=None):
        """
        Constructor that set experiment variables
        Args:
            results: dictonary containing the results
        """
        self.results = results

    def plot_results(self, order:list, split_level:int=0):
        """
        Function that reorder the result dictionnary
        and split them in several plots.
        Args:
            order: list that set the new order of the result dictionary
            split_level: number that set where to split the result
                        in several plots
        Example:
            If order is ['model','dataset','method','k']
            and split_level is 0, there will be a plot containg the result
            of each active_learning.
            If split_level is 1, it corresponds to 'model' splitting,
            so, there will be a plot per model.
            If split_level is 2, there will be a plot per model and dataset.
        
        """
        self.check_splitlevel(order, split_level)
        ordered_results = self.rearrange_results(order)

        if split_level == 0:
            plt.figure()
        for level1 in ordered_results.keys():
            if split_level == 1:
                plt.figure()
            for level2 in ordered_results[level1].keys():
                if split_level == 2:
                    plt.figure()
                for level3 in ordered_results[level1][level2].keys():
                    if split_level == 3:
                        plt.figure()
                    for level4 in ordered_results[level1][level2][level3].keys():
                        if split_level == 4:
                            plt.figure()
                        label, title = self.create_label_and_title(level1,level2, \
                                        level3,level4,split_level)
                        self.set_plot(title,label,ordered_results[level1][level2][level3][level4]) 
                        
        plt.show()

    def check_splitlevel(self, order:list, split_level:int):
        """
        Function that raise an error if the split level
        is bigger than order length
        Args:
            order: list of the dictionary order
            split_level: split position of the dictionary order 
        """
        if split_level > len(order):
            raise ValueError("Split level "+str(split_level)+" must "
                            "be less than "+str(len(order)))

    def rearrange_results(self, order:list):
        """
        Function that returns a new ordered dictionary
        Args:
            order: list of the new dictionary order
        Returns:
            An ordered result dictionary
        """
        new_results = {}
        for model in self.results.keys():
            for dataset in self.results[model].keys():
                for method in self.results[model][dataset].keys():
                    for k in self.results[model][dataset][method].keys():
                        if eval(order[0]) not in new_results:
                            new_results[eval(order[0])] = {}

                        if eval(order[1]) not in new_results[eval(order[0])]:
                            new_results[eval(order[0])][eval(order[1])] = {}

                        if eval(order[2]) not in new_results[eval(order[0])][eval(order[1])]:
                            new_results[eval(order[0])][eval(order[1])][eval(order[2])] = {}

                        if eval(order[3]) not in new_results[eval(order[0])][eval(order[1])][eval(order[2])]:
                            new_results[eval(order[0])][eval(order[1])][eval(order[2])][eval(order[3])] = {}

                        new_results[eval(order[0])][eval(order[1])][eval(order[2])][eval(order[3])] \
                                = self.results[model][dataset][method][k]

        return new_results

    def create_label_and_title(self,level1,level2,level3,level4,split_level):
        """
        Function that return label and title for plot
        depending on result variables and split level
        Args:
            level1: First result variable
            level2: Second result variable
            level3: Third result variable
            level4: Fourth result variable
            split_level: position where to split
                        result description in title and label
        Returns:
            title: the title of the plot
            label: the label of the plot
        """
        label = ""
        title = ""
        if split_level == 0:
            label = level1+" - "+level2+" - "+level3+" - "+level4
            title = "All experiment"
        if split_level == 1:
            label = level2+" - "+level3+" - "+level4
            title = level1
        if split_level == 2:
            label = level3+" - "+level4
            title = level1+" - "+level2
        if split_level == 3:
            label = level4
            title = level1+" - "+level2+" - "+level3
        if split_level == 4:
            label = None
            title = level1+" - "+level2+" - "+level3+" - "+level4

        return label, title

    def set_plot(self, title, label, result):
        """
        Function used to set the plot title,
        label if not None, axes and curve values
        Args:
            title: plot title
            label: plot label
            result: curve values
        """
        if label is None:
            plt.plot(result)
        else:
            plt.plot(result,label=label)
            plt.legend()

        plt.axis([None, None, 0, 100])
        plt.title(title)
        plt.xlabel("training")
        plt.ylabel("accuracy (%)")

    def save_results(self):
        """
        Function to save the results
        """
        with open(self.results_filename, 'wb') as handle:
            pickle.dump(self.results, handle)

    def load_results(self):
        """
        Function to load the saved results
        """
        with open(self.results_filename, 'rb') as handle:
            self.results = pickle.load(handle)

