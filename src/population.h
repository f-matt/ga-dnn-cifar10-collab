#ifndef POPULATION_H_
#define POPULATION_H_

#include "ann-config.h"
#include "solution.h"
#include "fitness-regressor.h"
#include "cifar10-data-wrapper.h"

#include <iomanip>
#include <fstream>

class Population {
public:
	Population(unsigned int size,
			float pc,
			float pm,
			float elitism,
			bool  use_pareto);

	virtual ~Population();

	void evolve();

	void print_stats();

	void update_stats();

	void save_stats(int generation);

private:
	void evaluate_solution(Solution& solution);

	void train_fitness_regressor();

	unsigned int size;

	float max_fitness;

	float min_fitness;

	float avg_fitness;

	vector<Solution> solutions;

	// GA parameters
	float pc;

	float pm;

	float elitism;

	bool  use_pareto;

	boost::shared_ptr<FitnessRegressor> fitness_regressor;

	boost::shared_ptr<Cifar10DataWrapper> data_wrapper;

	// Collab flag
	int evaluated_solutions;

};

#endif
