#include "fitness-data-wrapper.h"


const string TRAINING_FILE_PATH = "training.csv";


FitnessDataWrapper::FitnessDataWrapper() : new_data(false) {

}


FitnessDataWrapper::~FitnessDataWrapper() {

}


pair<vector<FitnessPattern>, vector<FitnessPattern>> FitnessDataWrapper::load_train_test_data(unsigned int n_train_patterns, unsigned int n_test_patterns) {

	pair<vector<FitnessPattern>, vector<FitnessPattern>> train_test_patterns;

	vector<FitnessPattern> patterns = load_patterns();

	if ( (n_train_patterns + n_test_patterns) != patterns.size()) {
		cerr << "Error: incorrect number of patterns specified." << endl;
		cerr << "Train: " << n_train_patterns << "; " << "Test: " << n_test_patterns << "; Available: " << patterns.size() << endl;
		exit(-1);
	}

	vector<FitnessPattern> train_patterns(n_train_patterns);
	vector<FitnessPattern> test_patterns(n_test_patterns);

	for (unsigned int i = 0; i < n_train_patterns; ++i) {
		train_patterns[i] = patterns[i];
	}

	for (unsigned int i = 0; i < n_test_patterns; ++i) {
		test_patterns[i] = *(patterns.rbegin() + i);
	}

	train_test_patterns.first = train_patterns;
	train_test_patterns.second = test_patterns;

	return train_test_patterns;

}


pair<vector<FitnessPattern>, vector<FitnessPattern>> FitnessDataWrapper::load_train_test_data() {

	vector<FitnessPattern> patterns = load_patterns();

	pair<vector<FitnessPattern>, vector<FitnessPattern>> train_test_patterns;

	int n_test_patterns = floor(0.2 * patterns.size());
	int n_train_patterns = patterns.size() - n_test_patterns;

	vector<FitnessPattern> train_patterns(n_train_patterns);
	vector<FitnessPattern> test_patterns(n_test_patterns);

	for (int i = 0; i < n_train_patterns; ++i) {
		train_patterns[i] = patterns[i];
	}

	for (int i = 0; i < n_test_patterns; ++i) {
		test_patterns[i] = *(patterns.rbegin() + i);
	}

	train_test_patterns.first = train_patterns;
	train_test_patterns.second = test_patterns;

	return train_test_patterns;

}


void FitnessDataWrapper::append_pattern(const FitnessPattern& pattern) {

	DbHandler db_handler;

	stringstream ss;
	ss << fixed << setprecision(2);

	for (unsigned int i = 0; i < pattern.input.size() - 1; ++i) {
		ss << pattern.input[i] << ";";
	}

	ss << *(pattern.input.rbegin());

	FitnessRecord fr;
	fr.descriptor 		= pattern.descriptor;
	fr.train_acc 		= pattern.train_acc;
	fr.test_acc 		= pattern.test_acc;
	fr.n_weights 		= pattern.n_weights;
	fr.training_input	= ss.str();

	rest_manager.insert(fr);

	db_handler.insert(fr);

	new_data = true;

	update_pareto_front(pattern);

}


void FitnessDataWrapper::update_pareto_front(const FitnessPattern &pattern) {

	DbHandler db_handler;

	vector<FitnessPattern> pareto_front = load_pareto_front();

	bool is_dominated = false;

	for (FitnessPattern fp : pareto_front) {
		// Check if dominant solution
		if ((pattern.n_weights < fp.n_weights) && (pattern.test_acc > fp.test_acc)) {
			db_handler.clear_pareto_flag(fp.id);
			// Check if dominated solution
		} else if ((pattern.n_weights > fp.n_weights) && (pattern.test_acc < fp.test_acc)) {
			is_dominated = true;
			break;
		}
	}

	if (!is_dominated) {
		db_handler.set_pareto_flag(pattern.id);
	}

}


float FitnessDataWrapper::get_test_acc(const string& descriptor) {

	DbHandler db_handler;

	FitnessRecord fr = db_handler.get_by_descriptor(descriptor);

	if (!fr.descriptor.empty()) {
		return fr.test_acc;
	}

	// Search in web service
	fr = rest_manager.get_by_descriptor(descriptor);

	if (!fr.descriptor.empty()) {
		db_handler.insert(fr);
		new_data = true;
		return fr.test_acc;
	}

	return -1;

}


pair<float, float> FitnessDataWrapper::get_train_test_acc(const string& descriptor) {

	DbHandler db_handler;

	FitnessRecord fr = db_handler.get_by_descriptor(descriptor);

	if (!fr.descriptor.empty()) {
		return pair<float, float>(fr.train_acc, fr.test_acc);
	}

	// Search in web service
	fr = rest_manager.get_by_descriptor(descriptor);

	if (!fr.descriptor.empty()) {
		db_handler.insert(fr);
		new_data = true;
		return pair<float, float>(fr.train_acc, fr.test_acc);
	}

	return pair<float, float>(-1, -1);

}




bool FitnessDataWrapper::is_dominated(const FitnessPattern& pattern) {

	DbHandler db_handler;

	vector<FitnessPattern> pareto_front = load_pareto_front();

	for (FitnessPattern fp : pareto_front) {
		if ((fp.n_weights < pattern.n_weights) && (fp.test_acc > pattern.test_acc)) {
			return true;
		}
	}

	return false;

}


bool FitnessDataWrapper::is_empty() {
	DbHandler db_handler;

	vector<FitnessRecord> records = db_handler.get_all();

	return records.empty();
}


int FitnessDataWrapper::available_patterns() {
	DbHandler db_handler;

	vector<FitnessRecord> records = db_handler.get_all();

	return records.size();
}


bool FitnessDataWrapper::has_new_data() {
	return new_data;
}


void FitnessDataWrapper::set_new_data(bool new_data) {
	this->new_data = new_data;
}


vector<FitnessPattern> FitnessDataWrapper::load_patterns() {

	vector<FitnessPattern> patterns;

	DbHandler db_handler;

	vector<FitnessRecord> records = db_handler.get_all();

	for (FitnessRecord fr : records) {

		FitnessPattern pattern;

		pattern.id = fr.id;
		pattern.descriptor = fr.descriptor;
		pattern.test_acc = fr.test_acc;
		pattern.n_weights = fr.n_weights;

		vector<string> strs;

		boost::split(strs, fr.training_input, boost::is_any_of(";"));

		for (unsigned int j = 0; j < strs.size(); ++j) {
			pattern.input.push_back(stof(strs[j]));
		}

		patterns.push_back(pattern);

	}

	return patterns;

}


vector<FitnessPattern> FitnessDataWrapper::load_pareto_front() {

	vector<FitnessPattern> pareto_front;

	DbHandler db_handler;

	vector<FitnessRecord> records = db_handler.get_pareto_front();

	for (FitnessRecord fr : records) {

		FitnessPattern pattern;

		pattern.id			= fr.id;
		pattern.descriptor  = fr.descriptor;
		pattern.train_acc 	= fr.train_acc;
		pattern.test_acc 	= fr.test_acc;
		pattern.n_weights 	= fr.n_weights;

		vector<string> strs;

		boost::split(strs, fr.training_input, boost::is_any_of(";"));

		for (unsigned int j = 0; j < strs.size(); ++j) {
			pattern.input.push_back(stof(strs[j]));
		}

		pareto_front.push_back(pattern);

	}

	return pareto_front;

}


/*
 * Clear db data from previous experiments
 */
void FitnessDataWrapper::clear() {
	DbHandler db_handler;
	db_handler.clear();

	new_data = false;
}
