import random
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker


class Chromosome:
    def __init__(self, actions):
        self.fitness = 0
        self.actions = actions
        self.reproduction_rate = 0


class GeneticAlgorithm:
    def __init__(self, level, population_size, calc_win_score):
        self.level = level
        self.chromosomes = []
        self.population_size = population_size
        self.calc_win_score = calc_win_score
        self.sum_fitness = 0
        self.chance_of_mutation = 0.2
        self.new_population = []
        self.generations = []
        self.generations_mean_fitness = []

    def initial_population(self):
        actions = ['0', '0', '0', '0', '0', '1', '2']
        for i in range(self.population_size):
            rand_str = ''.join(random.choice(actions) for i in range(len(self.level)))
            # print("Random string of length", len(self.level), "is:", rand_str)
            new_chromosome = Chromosome(rand_str)
            self.chromosomes.append(new_chromosome)

        # for i in self.chromosomes:
        #     print(i.actions)

    def evaluation(self, actions):
        current_level = self.level
        steps = 0
        score = 0
        extra_score = 0
        not_possible = False
        checked = []
        checked.extend(range(0, len(actions)))

        # Check the actions taken before each object
        # if it is correct earn a point
        for i in range(1, len(current_level) - 1):
            if current_level[i] == 'L':
                checked.remove(i - 1)
                if actions[i - 1] == '2':
                    score += 1
                if actions[i - 1] == '1' or actions[i] == '1':
                    score -= 1
            elif current_level[i] == 'G':
                checked.remove(i - 1)
                if actions[i - 1] == '1':
                    score += 1
                    # After jumping it can only move to the right
                    if actions[i] != '0':
                        not_possible = True
                elif i > 2:
                    if actions[i - 2] == '1':
                        score += 3
                if actions[i - 1] == '2' or actions[i] == '2':
                    score -= 1
            elif current_level[i] == 'M':
                # checked.remove(i)
                if actions[i - 1] != '1':
                    extra_score += 1

        # If other actions are zero, increase score
        # else it's not efficient so, lose score
        for i in checked:
            if actions[i] == '0':
                extra_score += 1
            else:
                extra_score -= 0.5

        # Jump at the end
        if actions[len(current_level) - 1] == '1':
            extra_score += 2

        # Check if it reaches the goal
        if self.calc_win_score:
            if score >= len(current_level) - 1:
                score += 5
                steps = len(current_level) - 1

        score += extra_score
        # Make sure that score is not negative
        if score < 0 or not_possible:
            score = 0
        return steps == len(current_level) - 1, score

    # def evaluation(self, actions):
    #     # Get an action sequence and determine the steps taken/score
    #     # Return a tuple, the first one indicates if these actions result in victory
    #     # and the second one shows the steps taken
    #     current_level = self.level
    #     steps = 0
    #     score = 0
    #     for i in range(len(current_level) - 1):
    #         current_step = current_level[i]
    #         if current_step == '_':
    #             steps += 1
    #         elif current_step == 'G' and actions[i - 1] == '1':
    #             steps += 1
    #         elif current_step == 'L' and actions[i - 1] == '2':
    #             steps += 1
    #         else:
    #             break
    #     if self.calc_win_score:
    #         if steps == len(current_level) - 1:
    #             score = steps + 5
    #         else:
    #             score = steps
    #     else:
    #         score = steps
    #     return steps == len(current_level) - 1, score

    def evaluate_all(self):
        self.sum_fitness = 0
        for i in self.chromosomes:
            win, score = self.evaluation(i.actions)
            self.sum_fitness += score
            i.fitness = score
        self.chromosomes.sort(reverse=True, key=lambda x: x.fitness)
        self.generations_mean_fitness.append(self.sum_fitness / self.population_size)

    def selection(self, choose_bests_only):
        if choose_bests_only:
            selected = self.chromosomes[0:int(self.population_size / 2)]
            selected.extend(selected)
        else:
            weights = []
            for i in self.chromosomes:
                weights.append(i.fitness / self.sum_fitness)
            selected = random.choices(self.chromosomes, weights=weights, k=int(self.population_size))
        # for i in selected:
        #     print(i.fitness)

        return selected

    def crossover(self, selected, one_point):
        new_population = []
        for i, x in enumerate(selected[::2]):
            y = selected[i + 1]
            string1_list = list(x.actions)
            string2_list = list(y.actions)
            point1 = random.randint(0, len(x.actions) - 2)
            if one_point:
                child_str1 = "".join(string1_list[:point1] + string2_list[point1:])
                child_str2 = "".join(string2_list[:point1] + string1_list[point1:])
            else:
                point2 = random.randint(0, len(x.actions) - 2)
                while point2 == point1:
                    point2 = random.randint(0, len(x.actions) - 2)
                child_str1 = "".join(string1_list[:point1] + string2_list[point1:point2] + string1_list[point2:])
                child_str2 = "".join(string2_list[:point1] + string1_list[point1:point2] + string2_list[point2:])
            # print("parent1: " + x.actions + " parent2: " + y.actions +"   "+ child_str1 + "  " + child_str2)
            child1 = Chromosome(child_str1)
            child2 = Chromosome(child_str2)
            new_population.append(child1)
            new_population.append(child2)
        return new_population

    def mutation(self, mutation_probability):
        # for i in self.new_population:
        #     print(i.actions, end=" ")
        # print()
        for i in self.new_population:
            if random.random() < mutation_probability:
                new_value = random.choices(['0', '1', '2'], weights=[0.4, 0.2, 0.2], k=1)
                selected_char = random.randint(0, len(i.actions) - 1)
                string_list = list(i.actions)
                while string_list[selected_char] == new_value[0]:
                    selected_char = random.randint(0, len(i.actions) - 1)

                string_list[selected_char] = new_value[0]
                newStr = "".join(string_list)
                print(newStr)
                i.actions = newStr
            else:
                continue

        # print("------")
        # for i in self.new_population:
        #     print(i.actions, end=" ")

    def mean_fitness_difference(self, num_generations, epsilon):
        if num_generations > len(self.generations):
            return False
        curr_mean = self.sum_fitness / self.population_size
        latest_generations = self.generations[-num_generations:]
        # print(latest_generations)
        if len(self.generations) == 1:
            return False
        for generation in reversed(latest_generations):
            sumFitness = 0
            for chromosome in generation:
                # print(chromosome.actions)
                sumFitness += chromosome.fitness
            g_mean = sumFitness / self.population_size

            # print(curr_mean)
            # print(g_mean)
            # print(curr_mean - g_mean)
            diff = curr_mean - g_mean
            if diff < 0:
                return False
            if diff > epsilon:
                return False
        return True

    def run_algorithm(self, choose_bests_only, one_point_crossover, epsilon, mutation_probability):
        self.initial_population()
        self.generations.append(self.chromosomes)
        loop_range = 40
        print("Initial population: ", end=" ")
        for i in self.chromosomes:
            print(i.actions, end=" ")
        print()
        self.evaluate_all()
        for i in range(loop_range):
            if self.mean_fitness_difference(15, epsilon):
                return self.generations, self.generations_mean_fitness
            selected = self.selection(choose_bests_only)
            self.chromosomes = self.crossover(selected, one_point_crossover)
            self.mutation(mutation_probability)
            self.evaluate_all()
            self.generations.append(self.chromosomes)
            print(str(i + 1) + "th generation population: ", end=" ")
            for i in self.chromosomes:
                print(i.actions, end=" ")
            print()
        return self.generations, self.generations_mean_fitness

class Game:
    def __init__(self, levels):
        # Get a list of strings as levels
        # Store level length to determine if a sequence of action passes all the steps
        self.levels = levels
        self.current_level_index = 0
        self.current_level_len = len(self.levels[self.current_level_index])

    def load_next_level(self):
        self.current_level_index += 1
        self.current_level_len = len(self.levels[self.current_level_index])


def plot_generations(generations1, generations_mean_fitness1, generations2, generations_mean_fitness2):
    generations_num = list(range(0, len(generations1)))
    mins = []
    maxs = []
    mean = generations_mean_fitness1
    for i in generations1:
        mins.append(i[-1].fitness)
        maxs.append(i[0].fitness)

        # print(generations_num)
        # print(mean)
        # print(mins)
        # print(maxs)

        # fig, axs = plt.subplots(1, figsize=(12, 5))
    fig, axs = plt.subplots(2, gridspec_kw={'hspace': 0.5, 'wspace': 10}, figsize=(15,15))
    # fig = plt.figure()
    # axs[0] = fig.add_subplot(1, 1, 1)
    axs[0].plot(generations_num, mean, 'b+', alpha=0.7, label='mean')
    axs[0].plot(generations_num, mins, 'rx', alpha=0.7, label='min')
    axs[0].plot(generations_num, maxs, 'go', alpha=0.4, label='max')
    axs[0].set_title('Result of the algorithm for easy level')
    axs[0].legend(loc=4)
    axs[0].set_ylabel('fitness score')
    axs[0].set_xlabel('Generation number')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))
    generations_num = list(range(0, len(generations2)))
    mins = []
    maxs = []
    mean = generations_mean_fitness2
    for i in generations2:
        mins.append(i[-1].fitness)
        maxs.append(i[0].fitness)

    axs[1].plot(generations_num, mean, 'b+', alpha=0.7, label='mean')
    axs[1].plot(generations_num, mins, 'rx', alpha=0.7, label='min')
    axs[1].plot(generations_num, maxs, 'go', alpha=0.4, label='max')
    axs[1].set_title('Result of the algorithm for hard level')
    axs[1].legend(loc=4)
    axs[1].set_ylabel('fitness score')
    axs[1].set_xlabel('Generation number')

    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))
    plt.show()


def plot_all_levels(g):
    ai_agent = GeneticAlgorithm(g.levels[g.current_level_index], population_size, calc_win_score)
    ai_agent.run_algorithm(choose_bests_only, one_point_crossover, diff_epsilon, mutation_probability)
    plot_generations(ai_agent)


if __name__ == '__main__':
    # Get input
    levels = []
    # num = input("Please enter number of levels:\n")
    # for i in range(num):
    #     each_level = [input("Please your level:\n")]
    #     levels.append(each_level)
    #
    # g = Game(levels)
    # for i in range(levels):
    #     g.load_next_level()
    #     plot_all_levels(g)

    g = Game(["____G_ML__G_", "___M____MGM________M_M______M____L___G____M____L__G__GM__L____ML__G___G___L___G__G___M__L___G____M__"])
    # g.load_next_level()
    # print(g.current_level_len)
    # print(g.levels[g.current_level_index])

    # Set these values differently to see different outcomes
    population_size = 50
    calc_win_score = True
    choose_bests_only = True
    one_point_crossover = True
    diff_epsilon = 0.001
    mutation_probability = 0.1

    ai_agent = GeneticAlgorithm(g.levels[g.current_level_index], population_size, calc_win_score)
    generations1, generations1mean = ai_agent.run_algorithm(choose_bests_only, one_point_crossover, diff_epsilon, mutation_probability)

    # ai_agent = GeneticAlgorithm(g.levels[g.current_level_index], population_size, calc_win_score)
    # ai_agent.run_algorithm(choose_bests_only, one_point_crossover, diff_epsilon, mutation_probability)
    # plot_generations(ai_agent)

    g.load_next_level()
    ai_agent = GeneticAlgorithm(g.levels[g.current_level_index], population_size, calc_win_score)
    generations2, generations2mean = ai_agent.run_algorithm(choose_bests_only, one_point_crossover, diff_epsilon, mutation_probability)
    plot_generations(generations1, generations1mean, generations2, generations2mean)
