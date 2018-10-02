#include "population.h"

#include "selector.h"


void print_individual(int idx, Solution& solution) {
	cout << boost::posix_time::to_simple_string(boost::posix_time::second_clock::local_time())
			<< " Solution " << setw(2) << right << setfill('0') << idx << ": ";
	cout << setw(80) << left << setfill(' ')
			<< solution.get_descriptor().to_string() << " | " << fixed
			<< setprecision(6) << solution.get_train_acc() << " | "
			<< solution.get_test_acc() << " | "
			<< solution.get_weights() << endl;
}


Population::Population(unsigned int size,
		float pc,
		float pm,
		float elitism,
		bool  use_pareto) :
				size(size),
				max_fitness(0),
				min_fitness(INTMAX_MAX),
				avg_fitness(0),
				pc(pc),
				pm(pm),
				elitism(elitism),
				use_pareto(use_pareto),
				data_wrapper(new Cifar10DataWrapper()),
				evaluated_solutions(0) {

	FitnessDataWrapper fitness_data_wrapper;

	train_fitness_regressor();

	for (unsigned int i = 0; i < size; ++i) {

		Solution s(data_wrapper);

		evaluate_solution(s);

		print_individual(i + 1, s);

		solutions.push_back(s);
	}

	update_stats();

}


Population::~Population() {

}


void Population::train_fitness_regressor() {
	string descriptor_str = "FR;64;0.4-FR;64;0.4-FR;64;0.4";

	TopologyDescriptor descriptor(descriptor_str);

	fitness_regressor.reset(new FitnessRegressor(descriptor,
			1, // output size
			1, // batch size
			100, // max epochs
			10)); // patience

	fitness_regressor->train();

}


void Population::evaluate_solution(Solution& solution) {

	long weights = solution.get_weights();

	// Use predictor to estimate solution fitness
	FitnessPattern pattern;
	pattern.descriptor = solution.get_descriptor().to_string();
	pattern.input = solution.get_output_vector();
	pattern.n_weights = weights;

	// Fitness data wrapper is used to test dominant and dominated solutions
	FitnessDataWrapper fitness_data_wrapper;

	if (use_pareto && fitness_regressor->is_ready()) {
		fitness_regressor->regress(pattern);
		float predicted_test_acc = pattern.test_acc;
		float predicted_train_acc = pattern.train_acc;

		cout << "[DBG] Predicted train acc: " << predicted_train_acc << endl;
		cout << "[DBG] Predicted test acc: " << predicted_test_acc << endl;

		// Dominated solution: 10% of chance to be evaluated
		// Non dominated solution: 90% of chance to be evaluated
		float threshold = 0.9;

		if (fitness_data_wrapper.is_dominated(pattern))
			threshold = 0.1;

		float eval_prob = (float) rand() / RAND_MAX;

		if (eval_prob < threshold) {
			pair<float, float> train_test_acc = fitness_data_wrapper.get_train_test_acc(solution.get_descriptor().to_string());

			if (train_test_acc.first >= 0) {
				solution.set_train_acc(train_test_acc.first);
				solution.set_test_acc(train_test_acc.second);
				solution.set_predicted(false);
			} else {
				solution.evaluate();

				// ### COLLAB BEGIN ###
				if (++evaluated_solutions >= 5) {
					cout << "Reached 5 evaluated solutions. Aborting." << endl;
					exit(0);
				}
				// ### COLLAB END ###

				pattern.train_acc = solution.get_train_acc();
				pattern.test_acc = solution.get_test_acc();
				fitness_data_wrapper.append_pattern(pattern);
			}
		} else {
			solution.set_train_acc(predicted_train_acc);
			solution.set_test_acc(predicted_test_acc);
			solution.set_predicted(true);
		}

		if (fitness_data_wrapper.has_new_data()) {
				train_fitness_regressor();
				fitness_data_wrapper.set_new_data(false);
		}
	} else {
		pair<float, float> train_test_acc = fitness_data_wrapper.get_train_test_acc(solution.get_descriptor().to_string());

		solution.set_predicted(false);

		if (train_test_acc.first >= 0) {
			solution.set_train_acc(train_test_acc.first);
			solution.set_test_acc(train_test_acc.second);
		} else {
			solution.evaluate();

			// ### COLLAB BEGIN ###
			if (++evaluated_solutions >= 5) {
				cout << "Reached 5 evaluated solutions. Aborting." << endl;
				exit(0);
			}
			// ### COLLAB END ###

			pattern.train_acc = solution.get_train_acc();
			pattern.test_acc = solution.get_test_acc();
			fitness_data_wrapper.append_pattern(pattern);
		}
	}

}


void Population::update_stats() {
	float total_fitness = 0;
	max_fitness = 0;
	min_fitness = INTMAX_MAX;

	for (size_t i = 0; i < solutions.size(); ++i) {

		float f = solutions[i].get_test_acc();

		if (f > max_fitness)
			max_fitness = f;

		if (f < min_fitness)
			min_fitness = f;

		total_fitness += f;
	}

	avg_fitness = total_fitness / solutions.size();

}


void Population::print_stats() {
	cout << "MIN: " << min_fitness << " || " << "AVG: " << avg_fitness << " || "
			<< "MAX: " << max_fitness << endl;
}


void Population::evolve() {

	Selector selector;

	sort(solutions.begin(), solutions.end());

	// Elitism
	int elite_size = (int) round(solutions.size() * elitism);

	vector<Solution> elite;
	vector<Solution> generation;

	for (int i = 0; i < elite_size; ++i) {
		elite.push_back(*(solutions.rbegin() + i));
	}

	FitnessDataWrapper fitness_data_wrapper;

	// Predicted elite solutions are always evaluated.
	for (size_t i = 0; i < elite.size(); ++i) {
		if (elite[i].is_predicted()) {
			pair<float, float> train_test_acc = fitness_data_wrapper.get_train_test_acc(elite[i].get_descriptor().to_string());

			if (train_test_acc.first >= 0) {
				elite[i].set_train_acc(train_test_acc.first);
				elite[i].set_test_acc(train_test_acc.second);
				elite[i].set_predicted(false);
			} else {
				cout << "[Population] Evaluating elite solution: " << elite[i].get_descriptor().to_string() << endl;
				cout << "[Population] Predicted test acc: " << elite[i].get_test_acc() << endl;

				elite[i].evaluate();

				// ### COLLAB BEGIN ###
				if (++evaluated_solutions >= 5) {
					cout << "Reached 5 evaluated solutions. Aborting." << endl;
					exit(0);
				}
				// ### COLLAB END ###

				cout << "[Population] Real fitness: " << elite[i].get_test_acc() << endl;

				// Save pattern
				FitnessPattern pattern;
				pattern.descriptor = elite[i].get_descriptor().to_string();
				pattern.input = elite[i].get_output_vector();
				pattern.n_weights = elite[i].get_weights();

				pattern.train_acc = elite[i].get_train_acc();
				pattern.test_acc = elite[i].get_test_acc();
				fitness_data_wrapper.append_pattern(pattern);
			}
		}
	}

	int i = 1;
	for (Solution& s : elite) {
		generation.push_back(s);
		print_individual(i++, s);
	}

	while (generation.size() < size) {

		Solution s1 = selector.roulette_select(solutions);

		if (generation.size() < (size - 2)) {
			float p = (float) rand() / RAND_MAX;

			if (p < pc) {
				//idx = rand() % generation.size();
				Solution s2 = selector.roulette_select(solutions);

				s1.crossover_conv(s2);

				p = (float) rand() / RAND_MAX;
				if (p < pc) {
					s1.crossover_fc(s2);
				}

				p = (float) rand() / RAND_MAX;
				if (p < pm) {
					s1.mutate_conv();
				}

				p = (float) rand() / RAND_MAX;
				if (p < pm) {
					s2.mutate_conv();
				}

				p = (float) rand() / RAND_MAX;
				if (p < pm) {
					s1.mutate_fc();
				}

				p = (float) rand() / RAND_MAX;
				if (p < pm) {
					s2.mutate_fc();
				}

				evaluate_solution(s1);

				print_individual(generation.size() + 1, s1);
				cout << "Weights: " << s1.get_weights() << endl;
				generation.push_back(s1);

				evaluate_solution(s2);


				print_individual(generation.size() + 1, s2);
				cout << "Weights: " << s2.get_weights() << endl;
				generation.push_back(s2);

			} else {

				p = (float) rand() / RAND_MAX;
				if (p < pm) {
					s1.mutate_conv();
				}

				p = (float) rand() / RAND_MAX;
				if (p < pm) {
					s1.mutate_fc();
				}

				evaluate_solution(s1);

				print_individual(generation.size() + 1, s1);

				generation.push_back(s1);

			}
		} else {

			float p = (float) rand() / RAND_MAX;
			if (p < pm) {
				s1.mutate_conv();
			}

			p = (float) rand() / RAND_MAX;
			if (p < pm) {
				s1.mutate_fc();
			}
			evaluate_solution(s1);

			print_individual(generation.size() + 1, s1);

			generation.push_back(s1);

		}

	}

	// Update population with current generation
	solutions = generation;

	update_stats();

}


void Population::save_stats(int generation) {

	DbHandler db_handler;

	for (size_t i = 0; i < solutions.size(); ++i) {
		db_handler.save_ga_log(
				generation,
				solutions[i].get_train_acc(),
				solutions[i].get_test_acc(),
				solutions[i].get_weights(),
				solutions[i].is_predicted()
		);
	}

}
