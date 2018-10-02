#ifndef FITNESS_DATA_WRAPPER_H_
#define FITNESS_DATA_WRAPPER_H_

#include "ann-config.h"
#include "data-wrapper.h"
#include "descriptors.h"
#include "rest-manager.h"
#include "db-handler.h"

#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iomanip>

using namespace std;

class FitnessDataWrapper {
public:
	FitnessDataWrapper();

	virtual ~FitnessDataWrapper();

	pair<vector<FitnessPattern>, vector<FitnessPattern>> load_train_test_data(unsigned int n_train_patterns, unsigned int n_test_patterns);

	pair<vector<FitnessPattern>, vector<FitnessPattern>> load_train_test_data();

	void append_pattern(const FitnessPattern &pattern);

	void update_pareto_front(const FitnessPattern &pattern);

	vector<FitnessPattern> get_patterns();

	float get_test_acc(const string& descriptor);

	pair<float, float> get_train_test_acc(const string& descriptor);

	bool is_dominated(const FitnessPattern& pattern);

	bool is_empty();

	int available_patterns();

	bool has_new_data();

	void set_new_data(bool new_data);

	void clear();

private:

	vector<FitnessPattern> load_patterns();

	vector<FitnessPattern> load_pareto_front();

	RestManager rest_manager;

	bool new_data;

};

#endif

