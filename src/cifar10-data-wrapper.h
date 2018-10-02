#ifndef CIFAR10_DATA_WRAPPER_H_
#define CIFAR10_DATA_WRAPPER_H_

#include "data-wrapper.h"


class Cifar10DataWrapper {
public:
	Cifar10DataWrapper();

	virtual ~Cifar10DataWrapper();

	pair<vector<Pattern>, vector<Pattern>> load_train_test_data(unsigned int n_train_patterns, unsigned int n_test_patterns);

	const int& get_height() const;

	const int& get_width() const;

	const int& get_channels() const;

	const int& get_type() const;

	void create_train_test_files() const;

private:
	vector<Pattern> load_patterns(const string& filename, int n_patterns) const;

	string train_file;

	string test_file;

	int height;

	int width;

	int channels;

	int type;

};

#endif
