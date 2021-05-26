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
                weights.append(i.fitness/self.sum_fitness)
            selected = random.choices(self.chromosomes, weights=weights, k=int(self.population_size/2))
        else:
            selected = self.chromosomes[0:int(self.population_size / 2)]

        # for i in selected:
        #     print(i.fitness)

        return selected

    def crossover(self):
        return

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

    def run_algorithm(self):
        self.initial_population()
        self.evaluate_all()
        self.mutation()


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
    # This outputs (False, 4)
    # print(g.current_level_len)
    # print(g.levels[g.current_level_index])
    # print(g.get_score("0000000000"))
    ai_agent = GeneticAlgorithm(g.levels[g.current_level_index], 20)
    ai_agent.initial_population()
    ai_agent.selection(False)
    ai_agent.mutation()