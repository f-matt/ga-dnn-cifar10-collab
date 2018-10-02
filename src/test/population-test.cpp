#include "../population.h"
#include "../utils.h"

#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace std;


/*
 * Test the memory consumption progress during population creation end evolution
 */
void memory_consumption_test() {

	const int POPULATION_SIZE = 5;
	const int GENERATIONS = 2;

	cout << "Before creation: " << get_memory_usage() << endl;

	Population population(POPULATION_SIZE, 0.5, 0.1, 0.1, true);

	cout << "After creation: " << get_memory_usage() << endl;

	population.print_stats();

	for (int i = 0; i < GENERATIONS; ++i) {
		cout << "Before evolve: " << get_memory_usage() << endl;
		population.evolve();
		cout << "Generation " << i + 1 << ":" << endl;
		population.print_stats();
		cout << "After evolve: " << get_memory_usage() << endl;
	}

}

