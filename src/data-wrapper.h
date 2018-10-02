#ifndef DATA_WRAPPER_H_
#define DATA_WRAPPER_H_

#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>
#include "ann-config.h"

template<class T>
class DataWrapper {
public:
	DataWrapper(const string& train_file, const string& test_file) :
		train_file(train_file),
		test_file(test_file) {

	}


	DataWrapper(const string& train_file) : train_file(train_file) {

	}

	virtual ~DataWrapper() {

	}

	virtual pair<vector<T>, vector<T>> load_train_test_data(unsigned int n_train_patterns, unsigned int n_test_patterns) = 0;

	virtual pair<vector<T>, vector<T>> load_train_test_data() = 0;

protected:
	string train_file;

	string test_file;

};

#endif
