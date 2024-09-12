from TravelSalesmanGeneticAlgorithm import TravelSalesmanGeneticAlgorithm
import pandas as pd
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_excel("JoseSalvadorGonzalezBlanco_GeneticAlgorithm\Code\Ciudades.xlsx")
    cities = df['Ciudad'].to_list()
    xs = df['x'].to_list()
    ys = df['y'].to_list()
    zipped = zip(cities, xs, ys)
    test_dic = {city: [x, y] for (city, x, y) in zipped}

    ga = TravelSalesmanGeneticAlgorithm(test_dic)

    ga.find_best_path(initial_pop=1000, crossover_point=0.3, crossover_prob=0.9, mutation_prob=0.6, num_mut_gen=3, iter=350)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
