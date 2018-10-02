#ifndef CIFAR10_CLASSIFIER_H_
#define CIFAR10_CLASSIFIER_H_

#include "ann-config.h"
#include "classifier.h"
#include "descriptors.h"
#include "cifar10-data-wrapper.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/timer.hpp>
#include <caffe/caffe.hpp>

using namespace caffe;
using namespace boost::filesystem;


class Cifar10Classifier : public Classifier<Pattern> {

public:
	Cifar10Classifier(const TopologyDescriptor &descriptor,
			boost::shared_ptr<Cifar10DataWrapper> &data_wrapper,
			int output_size,
			int batch_size,
			int max_epochs,
			int n_training_patterns,
			int n_test_patterns,
			int patience);

	Cifar10Classifier(boost::shared_ptr<Cifar10DataWrapper> &data_wrapper,
				int output_size,
				int batch_size,
				int max_epochs,
				int n_training_patterns,
				int n_test_patterns,
				int patience);

	Cifar10Classifier();

	virtual ~Cifar10Classifier();

	// Train the tracker
	void train();

	float train_with_batch(const vector<Pattern>& batch);

	void classify(const Pattern& pattern, int& pred);

	void classify(Pattern& pattern);

private:
	void add_input_layer();

	void add_softmax_loss_layer();

	boost::shared_ptr<Cifar10DataWrapper> data_wrapper;

};

#endif
