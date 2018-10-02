#include "cifar10-classifier.h"

#include <caffe/blob.hpp>
#include <caffe/net.hpp>
#include <caffe/proto/caffe.pb.h>

#include <opencv2/core/core.hpp>

#include <stddef.h>
#include <cassert>
#include <string>
#include <vector>


Cifar10Classifier::Cifar10Classifier(const TopologyDescriptor &descriptor,
		boost::shared_ptr<Cifar10DataWrapper> &data_wrapper,
		int output_size,
		int batch_size,
		int max_epochs,
		int n_training_patterns,
		int n_test_patterns,
		int patience) :
		Classifier(descriptor,
				output_size,
				batch_size,
				max_epochs,
				n_training_patterns,
				n_test_patterns,
				patience),
		data_wrapper(data_wrapper) {

	Classifier::init();

}


Cifar10Classifier::Cifar10Classifier(boost::shared_ptr<Cifar10DataWrapper> &data_wrapper,
		int output_size,
		int batch_size,
		int max_epochs,
		int n_training_patterns,
		int n_test_patterns,
		int patience) :
		Classifier(descriptor,
				output_size,
				batch_size,
				max_epochs,
				n_training_patterns,
				n_test_patterns,
				patience),
		data_wrapper(data_wrapper) {

}


Cifar10Classifier::Cifar10Classifier() : Classifier() {

}


Cifar10Classifier::~Cifar10Classifier() {

}


void Cifar10Classifier::add_input_layer() {
	LayerParameter *input_parameters = parameters->add_layer();
    input_parameters->set_name("input");
    input_parameters->set_type("Input");
    input_parameters->add_top("input");
    input_parameters->add_top("label");

    InputParameter *input_param = new InputParameter();

    BlobShape *image_shape = input_param->add_shape();
    image_shape->add_dim(1);
    image_shape->add_dim(data_wrapper->get_channels());
    image_shape->add_dim(data_wrapper->get_height());
    image_shape->add_dim(data_wrapper->get_width());

    BlobShape *output_shape = input_param->add_shape();
    output_shape->add_dim(1);
    output_shape->add_dim(OUTPUT_SIZE);
    output_shape->add_dim(1);
    output_shape->add_dim(1);

	input_parameters->set_allocated_input_param(input_param);

}


void Cifar10Classifier::train() {

	char progress[] = {'|', '/', '-', '\\'};

	// Load training and test patterns
	pair<vector<Pattern>, vector<Pattern>> train_test_data = data_wrapper->load_train_test_data(N_TRAIN_PATTERNS, N_TEST_PATTERNS);

	vector<Pattern> train_vector = train_test_data.first;

	vector<Pattern> test_vector = train_test_data.second;

	if (train_vector.size() % BATCH_SIZE != 0) {
		cerr << "The number of training patterns must be a multiple of the batch size" << endl;
		exit(-1);
	}

	int batches_per_epoch = train_vector.size() / BATCH_SIZE;

	vector <float> v_test_acc;
	float best_test_acc = 0;
	float best_train_acc = 0;
	int epochs_without_improvement = 0;

	for (size_t i = 0; i < MAX_EPOCHS; ++i) {

		float avg_batch_processing_delay = -1.0;

		double elapsed;

		boost::posix_time::ptime start, stop;
		boost::posix_time::time_duration td;

		for (int j = 0; j < batches_per_epoch; ++j) {

			cout << boost::posix_time::to_simple_string(boost::posix_time::second_clock::local_time()) <<
					" Epoch: " << right << setw(4) << setfill('0') << i + 1 << " " << progress[j % 4] <<
					" Avg delay: " << avg_batch_processing_delay << flush;

			vector<Pattern>::const_iterator first = train_vector.begin() + j * BATCH_SIZE;
			vector<Pattern>::const_iterator last = train_vector.begin() + (j+1) * BATCH_SIZE;

			vector<Pattern> batch(first, last);

			start = boost::posix_time::microsec_clock::local_time();
			train_with_batch(batch);
			stop = boost::posix_time::microsec_clock::local_time();

			td = stop - start;

			elapsed = td.total_milliseconds();

			if (avg_batch_processing_delay < 0) {
				avg_batch_processing_delay = elapsed;
			} else {
				avg_batch_processing_delay = (avg_batch_processing_delay + elapsed) / 2;
			}

			cout << '\r';
		}

		float epoch_train_acc, epoch_test_acc;

		// Correct and wrong classifications in training and test data
		int nc_train = 0, nw_train = 0, nc_test = 0, nw_test = 0;

		// Training data
		for (size_t j = 0; j < train_vector.size(); ++j) {
			int pred;
			int truth = train_vector[j].label;

			classify(train_vector[j], pred);

			if (truth == pred) {
				nc_train++;
			} else {
				nw_train++;
			}
		}

		// Test data
		for (size_t j = 0; j < test_vector.size(); ++j) {

			int pred;
			int truth = test_vector[j].label;

			classify(test_vector[j], pred);

			if (truth == pred) {
				nc_test++;
			} else {
				nw_test++;
			}

		}

		epoch_train_acc = (float) nc_train / (nc_train + nw_train);
		epoch_test_acc  = (float) nc_test / (nc_test + nw_test);

		cout << boost::posix_time::to_simple_string(boost::posix_time::second_clock::local_time()) <<
				" Epoch: " << right << setw(4) << setfill('0') << i + 1 <<
				" | Train ACC: " << fixed << setprecision(4) << epoch_train_acc <<
				" | Test ACC: " << epoch_test_acc << endl;

		v_test_acc.push_back(epoch_test_acc);

		//cout << "[Epoch " << fixed << setw(4) << (i+1) << "] Train loss: " << setprecision(6) << epoch_training_loss <<
		//		" Test loss: " << epoch_test_loss << endl;

		if (epoch_test_acc > best_test_acc) {
			best_train_acc = epoch_train_acc;
			best_test_acc = epoch_test_acc;

			//cout << " * " << endl; // Mark improvement
			epochs_without_improvement = 0;

			snapshot();
		} else {
			epochs_without_improvement++;
			//cout << endl;

			if (epochs_without_improvement == PATIENCE) {
				// cout << "No improvement in the past " << PATIENCE << " epochs. Early stopping..." << endl;
				break;
			}
		}
	}

	cout << '\r' << flush;

	// Update regressor best train and test loss
	train_acc = best_train_acc;
	test_acc = best_test_acc;

}


float Cifar10Classifier::train_with_batch(const vector<Pattern>& batch) {

	assert(net->phase() == caffe::TRAIN);

	const size_t num_images = batch.size();

	// Set network inputs to the appropriate size and number
	// First image
	Blob<float>* input_image_layer = net->input_blobs()[0];
	input_image_layer->Reshape(num_images, data_wrapper->get_channels(), data_wrapper->get_height(), data_wrapper->get_width());

	// To backprop, we need to input the ground-truth bounding boxes
	// Reshape the bounding boxes
	Blob<float>* input_gt = net->input_blobs()[1];
	input_gt->Reshape(num_images, OUTPUT_SIZE, 1, 1);

	// Forward reshape
	net->Reshape();

	vector<vector<Mat>> image_channels;
	image_channels.resize(num_images);

	// Pointer to image data
	float* image_data = input_image_layer->mutable_cpu_data();

	// Pointer to label data
	float* input_gt_data = input_gt->mutable_cpu_data();

	for (size_t i = 0; i < num_images; ++i) {
		for (int ch = 0; ch < data_wrapper->get_channels(); ++ch) {
			Mat image_channel(data_wrapper->get_height(), data_wrapper->get_width(), CV_32FC1, image_data);
			image_channels[i].push_back(image_channel);
			image_data += data_wrapper->get_width() * data_wrapper->get_height();
		}

		split(batch[i].image, image_channels[i]);

		input_gt_data[i] = batch[i].label;
	}

	step();

	boost::shared_ptr<Blob<float>> accuracy_layer = net->blob_by_name("accuracy");

	return accuracy_layer->cpu_data()[0];

}


void Cifar10Classifier::classify(const Pattern& pattern, int& pred) {

	assert(test_net->phase() == caffe::TEST);

	Blob<float>* input_image_layer = test_net->input_blobs()[0];
	input_image_layer->Reshape(1, data_wrapper->get_channels(), data_wrapper->get_height(), data_wrapper->get_width());

	Blob<float>* input_labels_layer = test_net->input_blobs()[1];
	input_labels_layer->Reshape(1, OUTPUT_SIZE, 1, 1);

	test_net->Reshape();

	vector<Mat> image_channels;

	float* img_data = input_image_layer->mutable_cpu_data();

	for (int ch = 0; ch < data_wrapper->get_channels(); ++ch) {
		Mat image_channel(data_wrapper->get_height(), data_wrapper->get_width(), CV_32FC1, img_data);
		image_channels.push_back(image_channel);
		img_data += data_wrapper->get_width() * data_wrapper->get_height();
	}

	split(pattern.image, image_channels);

	test_net->Forward();

	boost::shared_ptr<Blob<float>> argmax_layer = test_net->blob_by_name("argmax");

	pred = (int) argmax_layer->cpu_data()[0];

}


void Cifar10Classifier::classify(Pattern& pattern) {

	assert(test_net->phase() == caffe::TEST);

	Blob<float>* input_image_layer = test_net->input_blobs()[0];
	input_image_layer->Reshape(1, data_wrapper->get_channels(), data_wrapper->get_height(), data_wrapper->get_width());

	test_net->Reshape();

	vector<Mat> image_channels;

	float* img_data = input_image_layer->mutable_cpu_data();

	for (int ch = 0; ch < data_wrapper->get_channels(); ++ch) {
		Mat image_channel(data_wrapper->get_height(), data_wrapper->get_width(), CV_32FC1, img_data);
		image_channels.push_back(image_channel);
		img_data += data_wrapper->get_width() * data_wrapper->get_height();
	}

	split(pattern.image, image_channels);

	test_net->Forward();

	boost::shared_ptr<Blob<float>> output_layer = test_net->blob_by_name("argmax");

	pattern.label = (int) output_layer->cpu_data()[0];

}
