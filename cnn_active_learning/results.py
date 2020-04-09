import matplotlib.pyplot as plt

class Results(object):
    def __init__(self, results:dict, models, datasets, methods, Ks):
        self.default_order = ["model","dataset","method","k"]
        self.results = results
        self.models = models
        self.datasets = datasets
        self.methods = methods
        self.Ks = Ks

    def plot_results(self, order:list=None, split_level:int=0):
        if order is None:
            order = self.default_order
        self.check_order(order)
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


    def check_order(self, order:list):
        if not all(item in order for item in self.default_order):
            raise ValueError("Order "+str(order)+" must contain "
                            "values of "+str(self.default_order))


    def check_splitlevel(self, order:list, split_level:int):
        if split_level > len(order):
            raise ValueError("Split level "+str(split_level)+" must "
                            "be less than "+str(len(order)))

    def rearrange_results(self, order:list):
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
        if label is None:
            plt.plot(result)
        else:
            plt.plot(result,label=label)
            plt.legend()

        plt.axis([None, None, 0, 100])
        plt.title(title)
        plt.xlabel("training")
        plt.ylabel("accuracy (%)")

