#include "fitness-regressor.h"


FitnessRegressor::FitnessRegressor(const TopologyDescriptor &descriptor,
		int output_size,
		int batch_size,
		int max_epochs,
		int n_training_patterns,
		int n_test_patterns,
		int patience) :
		Regressor(descriptor, output_size, batch_size, max_epochs, n_training_patterns, n_test_patterns, patience),
		ready(false) {

	Regressor::init();

}

FitnessRegressor::FitnessRegressor(const TopologyDescriptor &descriptor,
		int output_size,
		int batch_size,
		int max_epochs,
		int patience) : Regressor(descriptor, output_size, batch_size, max_epochs, patience),
				ready(false) {

	Regressor::init();

}

FitnessRegressor::~FitnessRegressor() {

}


void FitnessRegressor::add_input_layer() {
	LayerParameter *input_parameters = parameters->add_layer();
	input_parameters->set_name("input");
	input_parameters->set_type("Input");
	input_parameters->add_top("input");
	input_parameters->add_top("target");

	InputParameter *input_param = new InputParameter();

	BlobShape *input_shape = input_param->add_shape();
	input_shape->add_dim(1);
	input_shape->add_dim(INPUT_VECTOR_LENGTH);

	BlobShape *target_shape = input_param->add_shape();
	target_shape->add_dim(1);
	target_shape->add_dim(OUTPUT_VECTOR_LENGTH);

	input_parameters->set_allocated_input_param(input_param);

}



void FitnessRegressor::train() {

	FitnessDataWrapper data_wrapper;

	if (data_wrapper.is_empty() || (data_wrapper.available_patterns() < 10)) {
		cout << "Fitness regressor not ready yet. Skipping training." << endl;
		ready = false;
		return;
	}

	// Load training and test patterns
	pair<vector<FitnessPattern>, vector<FitnessPattern>> train_test_data = data_wrapper.load_train_test_data();

	vector<FitnessPattern> train_vector = train_test_data.first;
	vector<FitnessPattern> test_vector = train_test_data.second;

	cout << "Loaded " << train_vector.size() << " training patterns and " << test_vector.size() << " test patterns." << endl;

	if (train_vector.size() % batch_size != 0) {
		cerr << "The number of training patterns must be a multiple of the batch size" << endl;
		exit(-1);
	}

	int batches_per_epoch = train_vector.size() / batch_size;

	cout << batches_per_epoch << " batches per epoch" << endl;

	vector <float> v_test_loss;
	float best_test_loss = 1e10, best_train_loss = 1e10;
	int epochs_without_improvement = 0;

	for (int i = 0; i < max_epochs; ++i) {

		float epoch_training_loss = 0;

		for (int j = 0; j < batches_per_epoch; ++j) {

			vector<FitnessPattern>::const_iterator first = train_vector.begin() + j * batch_size;
			vector<FitnessPattern>::const_iterator last = train_vector.begin() + (j+1) * batch_size;

			vector<FitnessPattern> batch(first, last);

			epoch_training_loss += train_with_batch(batch);
		}

		epoch_training_loss /= batches_per_epoch;

		float epoch_test_loss = 0;

		for (size_t j = 0; j < test_vector.size(); ++j) {
			vector<float> pred(output_size);

			float loss;

			regress(test_vector[j], pred, loss);

			epoch_test_loss += loss;

		}

		epoch_test_loss /= test_vector.size();

		v_test_loss.push_back(epoch_test_loss);

		if (epoch_training_loss < best_train_loss) {
			best_train_loss = epoch_training_loss;
		}

		if (epoch_test_loss < best_test_loss) {
			best_test_loss = epoch_test_loss;
			epochs_without_improvement = 0;

			snapshot();
		} else {
			epochs_without_improvement++;

			if (epochs_without_improvement == patience) {
				cout << "No improvement in the past " << patience << " epochs. Early stopping..." << endl;
				break;
			}
		}

	}

	// Update regressor best test loss
	test_loss = best_test_loss;

	ready = true;

	cout << "[Summary] Best train loss: " << best_train_loss << " | Best test loss: " << best_test_loss << endl;

	cout << "Finished fitness regressor training." << endl;

}


float FitnessRegressor::train_with_batch(const vector<FitnessPattern>& batch) {

	assert(net->phase() == caffe::TRAIN);

	const size_t num_patterns = batch.size();

	// Set network inputs to the appropriate size and number
	// First image
	Blob<float>* input_layer = net->input_blobs()[0];
	input_layer->Reshape(num_patterns, INPUT_VECTOR_LENGTH, 1, 1);

	// To backprop, we need to input the ground-truth bounding boxes
	// Reshape the bounding boxes
	Blob<float>* input_target = net->input_blobs()[1];
	input_target->Reshape(num_patterns, output_size, 1, 1);

	// Forward reshape
	net->Reshape();

	// Get a pointer to the bbox memory
	float* input_data = input_layer->mutable_cpu_data();
	float* input_gt_data = input_target->mutable_cpu_data();

	int input_data_counter = 0;
	int input_gt_data_counter = 0;

	for (size_t i = 0; i < batch.size(); ++i) {
		const FitnessPattern& patt = batch[i];

		for (size_t j = 0; j < patt.input.size(); ++j)
			input_data[input_data_counter++] = patt.input[j];

		input_gt_data[input_gt_data_counter++] = patt.test_acc;
	}

	step();

	boost::shared_ptr<Blob<float>> loss = net->blob_by_name("loss");

	return loss->cpu_data()[0];

	return 0;
}


void FitnessRegressor::regress(const FitnessPattern& pattern, vector<float> &pred, float &loss) {

	assert(test_net->phase() == caffe::TEST);

	Blob<float>* input_layer = test_net->input_blobs()[0];
	input_layer->Reshape(1, INPUT_VECTOR_LENGTH, 1, 1);

	Blob<float>* input_target = test_net->input_blobs()[1];
	input_target->Reshape(1, output_size, 1, 1);

	// Forward dimension change
	test_net->Reshape();

	// Get a pointer to the bbox memory
	float* input_data = input_layer->mutable_cpu_data();
	float* input_gt_data = input_target->mutable_cpu_data();

	int input_data_counter = 0;

	for (size_t j = 0; j < pattern.input.size(); ++j)
		input_data[input_data_counter++] = pattern.input[j];

	input_gt_data[0] = pattern.test_acc;

	test_net->Forward(&loss);

	boost::shared_ptr<Blob<float>> output_layer = test_net->blob_by_name("output");

	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_size;
	pred = vector<float>(begin, end);

}


void FitnessRegressor::regress(FitnessPattern& pattern) {

	assert(test_net->phase() == caffe::TEST);

	// Input layer dimensions
	// First image
	Blob<float>* input_layer = test_net->input_blobs()[0];
	input_layer->Reshape(1, INPUT_VECTOR_LENGTH, 1, 1);

	// Forward dimension change
	test_net->Reshape();

	float* input_data = input_layer->mutable_cpu_data();

	int input_data_counter = 0;

	for (size_t j = 0; j < pattern.input.size(); ++j) {
		input_data[input_data_counter++] = pattern.input[j];
	}

	test_net->Forward();

	boost::shared_ptr<Blob<float>> output_layer = test_net->blob_by_name("output");

	pattern.test_acc = output_layer->cpu_data()[0];

//	const float* begin = output_layer->cpu_data();
//	const float* end = begin + output_size;
//
//	pattern.fitness = vector<float>(begin, end)[0];

}


bool FitnessRegressor::is_ready() {
	return ready;
}

