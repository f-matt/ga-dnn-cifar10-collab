#include "selector.h"

Selector::Selector() {

}

Selector::~Selector() {

}


Solution Selector::roulette_select(vector<Solution>& population) {

	vector<float> prob;

	float total_prob = 0;
	float total_fitness = 0;

	for (vector<Solution>::iterator it = population.begin();
				it != population.end();
				++it) {
		total_fitness += it->get_test_acc();
	}

	for (vector<Solution>::iterator it = population.begin();
			it != population.end();
			++it) {

		float this_prob = it->get_test_acc() / total_fitness;

		prob.push_back(total_prob + this_prob);

		total_prob += this_prob;

	}

	float p = (float) rand() / RAND_MAX;

	for (size_t j = 0; j < prob.size(); ++j) {
		if (p < prob[j]) {
			return population[j];
		}
	}

	return *population.rbegin();

}


Solution Selector::inverted_roulette_select(vector<Solution>& population) {

	vector<float> prob;

	float total_prob = 0;
	float total_fitness = 0;

	for (vector<Solution>::iterator it = population.begin();
				it != population.end();
				++it) {
		total_fitness += (1.0 / it->get_test_acc());
	}

	for (vector<Solution>::iterator it = population.begin();
			it != population.end();
			++it) {
		float this_prob = (1.0 / it->get_test_acc()) / total_fitness;

		prob.push_back(total_prob + this_prob);

		total_prob += this_prob;
	}

	float p = (float) rand() / RAND_MAX;

	for (size_t j = 0; j < prob.size(); ++j) {
		if (p < prob[j]) {
			return population[j];
		}
	}

	return *population.rbegin();

}

