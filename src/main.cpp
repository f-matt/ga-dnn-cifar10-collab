#include "descriptors.h"
#include "fitness-regressor.h"
#include "fitness-data-wrapper.h"
#include "cifar10-data-wrapper.h"
#include "cifar10-classifier.h"
#include "population.h"
#include "db-handler.h"

#include <iostream>
#include <fstream>

using namespace std;

const int POPULATION_SIZE = 200;
const int N_GENERATIONS = 20;

const int MAX_EVALUATIONS_SKIPPED = 19;

struct training_result {
	string topology;
	long weights;
	float train_acc;
	float test_acc;
};


training_result train(const TopologyDescriptor& descriptor) {

	boost::shared_ptr<Cifar10DataWrapper> data_wrapper(new Cifar10DataWrapper());

	boost::shared_ptr<Cifar10Classifier> classifier(new Cifar10Classifier(descriptor,
			data_wrapper,
			N_CLASSES, // output size
			BATCH_SIZE, // batch size
			MAX_EPOCHS, // max epochs
			N_TRAIN_PATTERNS, // n training patterns
			N_TEST_PATTERNS, // n test patterns
			PATIENCE)); // patience

	classifier->train();

	training_result tr;
	tr.topology = descriptor.to_string();
	tr.weights = classifier->get_n_weights();
	tr.train_acc = classifier->get_train_acc();
	tr.test_acc = classifier->get_test_acc();

	return tr;

}


void create_training_pattern(const string& descriptor) {

	TopologyDescriptor topology_descriptor(descriptor);

	training_result tr = train(topology_descriptor);

	std::ofstream train_file;
	train_file.open ("training.csv", ios::app);

	train_file << fixed << setprecision(2);

	vector<float> input_pattern = topology_descriptor.get_output_vector();

	for (unsigned int i = 0; i < input_pattern.size() - 1; ++i) {
		train_file << input_pattern[i] << ",";
	}

	// Input features are separated by comma. Semicolons separate input features from outputs.
	train_file << *(input_pattern.rbegin()) << ";";

	train_file << tr.test_acc << ";" << tr.weights << endl;

	train_file.close();

}



void debug_training(const string& descriptor) {

	TopologyDescriptor topology_descriptor(descriptor);

	training_result tr = train(topology_descriptor);

	cout << "Descriptor: " << descriptor << endl;
	cout << "Train acc: " << tr.train_acc << endl;
	cout << "Test acc: " << tr.test_acc << endl;
	cout << "Weights: " << tr.weights << endl;

}


/*
 * Evaluate 1000 random generated unique solutions
 */
void run_random_experiment() {

	DbHandler db_handler;

	vector<Solution> solutions;

	boost::shared_ptr<Cifar10DataWrapper> cifar10_data_wrapper(new Cifar10DataWrapper);

	int evaluations_skipped = 0;

	while (solutions.size() < 4000) {

		Solution solution(cifar10_data_wrapper);

		// Check if solution already exists
		bool found = false;
		for (Solution s : solutions) {
			if (s.get_descriptor().to_string() == solution.get_descriptor().to_string()) {
				found = true;
				break;
			}
		}

		// If solution already exists, continue
		if (found) {
			continue;
		}

		cout << "Inserting solution " << solutions.size() + 1 << " " << solution.get_descriptor().to_string() << endl;

		solutions.push_back(solution);
	}

	// Evaluate a meximum of 5 models for each run to prevent long running machine pause
	int evaluated_models = 0;

	for (Solution &solution : solutions) {

		if (evaluated_models == 5) {
			exit(0);
		}

		// Use predictor to estimate solution fitness
		FitnessPattern pattern;
		pattern.descriptor = solution.get_descriptor().to_string();
		pattern.input = solution.get_output_vector();
		pattern.n_weights = solution.get_weights();

		// Fitness data wrapper is used to test dominant and dominated solutions
		FitnessDataWrapper fitness_data_wrapper;

		pair<float, float> train_test_acc = fitness_data_wrapper.get_train_test_acc(solution.get_descriptor().to_string());

		if (train_test_acc.first >= 0) {
			solution.set_train_acc(train_test_acc.first);
			solution.set_test_acc(train_test_acc.second);
			solution.set_predicted(false);
		} else {
			if (evaluations_skipped == MAX_EVALUATIONS_SKIPPED) {
				evaluations_skipped = 0;
				evaluated_models++;
				cout << "Evaluating solution " << solution.get_descriptor().to_string() << endl;
				solution.evaluate();
				pattern.train_acc = solution.get_train_acc();
				pattern.test_acc = solution.get_test_acc();
				fitness_data_wrapper.append_pattern(pattern);
			} else {
				evaluations_skipped++;
				continue;
			}
		}

	}

}


/*
 * Generate valid conv descriptors
 */
vector<string> generate_conv_descriptors() {
	vector<string> descriptors;

	int conv_filters[] = {16, 32};
	int conv_kernel_size[] =  {3, 5};

	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			stringstream ss;
			ss << "CR;" << conv_filters[i] << ";" << conv_kernel_size[j] << ";1;2";
			descriptors.push_back(ss.str());
		}
	}

	return descriptors;

}


/*
 * Generate valid fc descriptors
 */
vector<string> generate_fc_descriptors() {
	vector<string> descriptors;

	string activation_functions[] = {"S", "T", "R"};
	int fc_neurons[] = {32, 64};

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 2; ++j) {
			stringstream ss;
			ss << "F" << activation_functions[i] << ";" << fc_neurons[j] << ";0.4";
			descriptors.push_back(ss.str());
		}
	}

	return descriptors;

}


/*
 * Evaluate solutions using grid search
 */
void run_grid_experiment() {

	DbHandler db_handler;

	vector<Solution> solutions;

	boost::shared_ptr<Cifar10DataWrapper> cifar10_data_wrapper(new Cifar10DataWrapper);

	// create_training_pattern("CR;8;9;2;2-CR;8;9;2;2-FR;8;0.2-FR;8;0.2");
	// create_training_pattern("CR;8;13;3;2-FR;8;0.2");
	// create_training_pattern("FR;8;0.2-FR;8;0.2-FR;8;0.2");

	vector<string> descriptors;

	vector<string> conv_descriptors = generate_conv_descriptors();
	vector<string> fc_descriptors = generate_fc_descriptors();

	// 1 conv layer
	for (string d1 : conv_descriptors) {
		descriptors.push_back(d1);
	}

	// 2 conv layers
	for (string d1 : conv_descriptors) {
		for (string d2 : conv_descriptors) {
			descriptors.push_back(d1 + "-" + d2);
		}
	}

	// 3 conv layers
	for (string d1 : conv_descriptors) {
		for (string d2 : conv_descriptors) {
			for (string d3 : conv_descriptors) {
				descriptors.push_back(d1 + "-" + d2 + "-" + d3);
			}
		}
	}

	// 1 fc layer
	for (string d1 : fc_descriptors) {
		descriptors.push_back(d1);
	}

	// 2 fc layers
	for (string d1 : fc_descriptors) {
		for (string d2 : fc_descriptors) {
			descriptors.push_back(d1 + "-" + d2);
		}
	}

	// 1 conv + 1 fc layer
	for (string d1 : conv_descriptors) {
		for (string d2 : fc_descriptors) {
			descriptors.push_back(d1 + "-" + d2);
		}
	}

	// 1 conv + 2 fc layers
	for (string d1 : conv_descriptors) {
		for (string d2 : fc_descriptors) {
			for (string d3 : fc_descriptors) {
				descriptors.push_back(d1 + "-" + d2 + "-" + d3);
			}
		}
	}

	// 2 conv + 1 fc layer
	for (string d1 : conv_descriptors) {
		for (string d2 : conv_descriptors) {
			for (string d3 : fc_descriptors) {
				descriptors.push_back(d1 + "-" + d2 + "-" + d3);
			}
		}
	}

	// 2 conv + 2 fc layers
	for (string d1 : conv_descriptors) {
		for (string d2 : conv_descriptors) {
			for (string d3 : fc_descriptors) {
				for (string d4 : fc_descriptors) {
					descriptors.push_back(d1 + "-" + d2 + "-" + d3 + "-" + d4);
				}
			}
		}
	}

	// 3 conv + 1 fc layer
	for (string d1 : conv_descriptors) {
		for (string d2 : conv_descriptors) {
			for (string d3 : conv_descriptors) {
				for (string d4 : fc_descriptors) {
					descriptors.push_back(d1 + "-" + d2 + "-" + d3 + "-" + d4);
				}
			}
		}
	}

	// 3 conv + 2 fc layer
	for (string d1 : conv_descriptors) {
		for (string d2 : conv_descriptors) {
			for (string d3 : conv_descriptors) {
				for (string d4 : fc_descriptors) {
					for (string d5 : fc_descriptors) {
						descriptors.push_back(d1 + "-" + d2 + "-" + d3 + "-" + d4 + "-" + d5);
					}
				}
			}
		}
	}

	int evaluations_skipped = 0;

	// Evaluate a meximum of 5 models for each run to prevent long running machine pause
	int evaluated_models = 0;

	for (string descriptor : descriptors) {

		if (evaluated_models == 5) {
			exit(0);
		}

		TopologyDescriptor topology_descriptor(descriptor);
	
		if (!topology_descriptor.is_valid(cifar10_data_wrapper->get_width(), cifar10_data_wrapper->get_height())) {
			cout << "Invalid descriptor: " << topology_descriptor.to_string() << ". Skipping" << endl;
			continue;
		}

		Solution solution(cifar10_data_wrapper, descriptor);

		// Use predictor to estimate solution fitness
		FitnessPattern pattern;
		pattern.descriptor = solution.get_descriptor().to_string();
		pattern.input = solution.get_output_vector();
		pattern.n_weights = solution.get_weights();

		// Fitness data wrapper is used to test dominant and dominated solutions
		FitnessDataWrapper fitness_data_wrapper;

		pair<float, float> train_test_acc = fitness_data_wrapper.get_train_test_acc(solution.get_descriptor().to_string());

		if (train_test_acc.first >= 0) {
			solution.set_train_acc(train_test_acc.first);
			solution.set_test_acc(train_test_acc.second);
			solution.set_predicted(false);
		} else {
			if (evaluations_skipped == MAX_EVALUATIONS_SKIPPED) {
				evaluations_skipped = 0;
				evaluated_models++;
				cout << "Evaluating solution " << solution.get_descriptor().to_string() << endl;
				solution.evaluate();
				pattern.train_acc = solution.get_train_acc();
				pattern.test_acc = solution.get_test_acc();
				fitness_data_wrapper.append_pattern(pattern);
			} else {
				evaluations_skipped++;
				continue;
			}
		}

		cout << "Inserting solution " << solutions.size() + 1 << " " << solution.get_descriptor().to_string() << endl;

		solutions.push_back(solution);
	}

	// Update training log
	for (size_t i = 0; i < solutions.size(); ++i) {
		db_handler.save_grid_log(
				solutions[i].get_train_acc(),
				solutions[i].get_test_acc(),
				solutions[i].get_weights(),
				solutions[i].is_predicted()
		);
	}

}


void run_ga_experiment() {

	FitnessDataWrapper fitness_data_wrapper;

	fitness_data_wrapper.clear();

	Population population(POPULATION_SIZE, 0.7, 0.2, 0.1, false);

	population.print_stats();
	population.save_stats(0);

	for (int i = 0; i < N_GENERATIONS; ++i) {
		population.evolve();
		cout << "Generation " << i + 1 << ":" << endl;
		population.print_stats();
		population.save_stats(i+1);
	}

}


void run_ga_pareto_experiment() {

	FitnessDataWrapper fitness_data_wrapper;

	fitness_data_wrapper.clear();

	Population population(POPULATION_SIZE, 0.7, 0.2, 0.1, true);

	population.print_stats();
	population.save_stats(0);

	for (int i = 0; i < N_GENERATIONS; ++i) {
		population.evolve();
		cout << "Generation " << i + 1 << ":" << endl;
		population.print_stats();
		population.save_stats(i+1);
	}

}


void sync_rest_with_database() {

	DbHandler db;
	RestManager rest;

	vector<FitnessRecord> records = db.get_all();

	for (FitnessRecord fr : records) {

		cout << "Testing " << fr.descriptor << endl;

		FitnessRecord fr_rest = rest.get(fr.descriptor);

		if (fr_rest.descriptor.empty()) {
			cout << "Not found in REST: " << fr.descriptor << endl;
			rest.insert(fr);

			exit(EXIT_SUCCESS);

		} else {

			cout << "DBG: " << fr_rest.descriptor << endl;

			if (fr.train_acc != fr_rest.train_acc) {
				cout << "Different fitness: Rest: " << fr_rest.train_acc << " Db: " << fr.train_acc << endl;
			}

			if (fr.test_acc != fr_rest.test_acc) {
				cout << "Different fitness: Rest: " << fr_rest.test_acc << " Db: " << fr.test_acc << endl;
			}


			if (fr.n_weights != fr_rest.n_weights) {
				cout << "Different n_weights: Rest: " << fr_rest.n_weights << " Db: " << fr.n_weights << endl;
				// TODO: update rest record
			}

		}

	}

}


void create_solution(const string& descriptor) {

	boost::shared_ptr<Cifar10DataWrapper> data_wrapper(new Cifar10DataWrapper());

	Solution solution(data_wrapper, descriptor);

	solution.evaluate();

	cout << "\nTrain acc: " << solution.get_train_acc() << endl;
	cout << "\nTest acc: " << solution.get_test_acc() << endl;
	cout << "Weights: " << solution.get_weights() << endl;

}






int main(int argc, char *argv[]) {

	google::InitGoogleLogging("GA-DL");
	google::SetCommandLineOption("GLOG_minloglevel", "1");

	srand(20180928);

	// Cifar10DataWrapper data_wrapper;
	// data_wrapper.create_train_test_files();
	// data_wrapper.load_train_test_data(20, 10);

	// debug_training("CR;8;13;3;2-FR;8;0.2");
	// debug_training("CR;8;9;2;2-CR;8;9;2;2-FR;8;0.2-FR;8;0.2");
	// debug_training("FR;8;0.2-FR;8;0.2-FR;8;0.2");

	// run_random_experiment();
	run_grid_experiment();
	// run_ga_experiment();
	// run_ga_pareto_experiment();

	// sync_rest_with_database();

	// create_solution("CR;32;5;2;4");

	return EXIT_SUCCESS;

}
