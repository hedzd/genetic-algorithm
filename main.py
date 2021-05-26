import random


class Chromosome:
    def __init__(self, actions):
        self.fitness = 0
        self.actions = actions
        self.reproduction_rate = 0


class GeneticAlgorithm:
    def __init__(self, level, population_size):
        self.level = level
        self.chromosomes = []
        self.population_size = population_size
        self.sum_fitness = 0
        self.chance_of_mutation = 0.2
        self.new_population = []
        self.generations = []

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
        # Get an action sequence and determine the steps taken/score
        # Return a tuple, the first one indicates if these actions result in victory
        # and the second one shows the steps taken
        current_level = self.level
        steps = 0
        score = 0
        for i in range(len(current_level) - 1):
            current_step = current_level[i]
            if current_step == '_':
                steps += 1
            elif current_step == 'G' and actions[i - 1] == '1':
                steps += 1
            elif current_step == 'L' and actions[i - 1] == '2':
                steps += 1
            else:
                break
        if steps == len(current_level) - 1:
            score = steps + 5
        else:
            score = steps
        return steps == len(current_level) - 1, score

    def evaluate_all(self):
        for i in self.chromosomes:
            win, score = self.evaluation(i.actions)
            self.sum_fitness += score
            i.fitness = score
        self.chromosomes.sort(reverse=True, key=lambda x: x.fitness)

    def selection(self, select_randomly):
        if select_randomly:
            weights = []
            for i in self.chromosomes:
                weights.append(i.fitness / self.sum_fitness)
            selected = random.choices(self.chromosomes, weights=weights, k=int(self.population_size))
        else:
            selected = self.chromosomes[0:int(self.population_size / 2)]
            selected.extend(selected)
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

    def mutation(self):
        # for i in self.new_population:
        #     print(i.actions, end=" ")
        # print()
        for i in self.new_population:
            if random.random() < 0.2:
                new_value = random.choices(['0', '1', '2'], weights=[0.8, 0.1, 0.1], k=1)
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
        curr_mean = self.sum_fitness/self.population_size
        total_num = len(self.generations)
        for g in reversed(self.generations[total_num-num_generations:-1]):
            sum = 0
            for chromosome in g:
                print(chromosome.actions)
                sum += chromosome.fitness
            g_mean = sum/self.population_size
            print(curr_mean)
            print(g_mean)
            print(curr_mean - g_mean)
            if curr_mean - g_mean > epsilon:
                return False
        return True

    def run_algorithm(self, select_randomly, one_point_crossover):
        self.initial_population()
        self.generations.append(self.chromosomes)
        print("Initial population: ", end=" ")
        for i in self.chromosomes:
            print(i.actions, end=" ")
        print()
        for i in range(50):
            self.evaluate_all()
            if self.mean_fitness_difference(5, 0.000001):
                break
            selected = self.selection(select_randomly)
            self.chromosomes = self.crossover(selected, one_point_crossover)
            self.mutation()
            self.generations.append(self.chromosomes)

            print(str(i + 1) + "th iteration population: ", end=" ")
            for i in self.chromosomes:
                print(i.actions, end=" ")
            print()




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


if __name__ == '__main__':
    g = Game(["____G__L__", "___G_M___L_"])
    # g.load_next_level()
    # print(g.current_level_len)
    # print(g.levels[g.current_level_index])

    ai_agent = GeneticAlgorithm(g.levels[g.current_level_index], 5)
    ai_agent.run_algorithm(False, True)
