import csv
import neat

class CSVReporter(neat.reporting.BaseReporter):
    def __init__(self, filename):
        self.filename = filename
        self.generation = 0

        # Initialize the CSV file and write the header
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Generation', 'Best Fitness', 'Average Fitness', 'Worst Fitness'])

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        best_fitness = best_genome.fitness
        fitnesses = [genome.fitness for genome in population.values()]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        worst_fitness = min(fitnesses)

        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.generation, best_fitness, avg_fitness, worst_fitness])
