#include "cifar10-data-wrapper.h"

#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>

const int IMG_WIDTH = 32;
const int IMG_HEIGHT = 32;
const int IMG_CHANNELS = 3;
const int IMG_TYPE = CV_32FC3;

Cifar10DataWrapper::Cifar10DataWrapper() : train_file("/home/fernando/data/cifar-10/training.bin"),
		test_file("/home/fernando/data/cifar-10/test.bin"),
		height(IMG_HEIGHT),
		width(IMG_WIDTH),
		channels(IMG_CHANNELS),
		type(IMG_TYPE) {

}


Cifar10DataWrapper::~Cifar10DataWrapper() {

}


vector<Pattern> Cifar10DataWrapper::load_patterns(const string& filename, const int n_patterns) const {

	vector<Pattern> patterns(n_patterns);

	ifstream input_file(filename, ios::binary);

	int added_patterns = 0;

	if (input_file.is_open()) {

		std::vector<char> buffer(
				(istreambuf_iterator<char>(input_file)),
				istreambuf_iterator<char>());
		size_t idx = 0;

		while (added_patterns < n_patterns) {
			patterns[added_patterns].label = (int) buffer[idx++];

			vector<Mat> bgr(3);
			bgr[0] = Mat::zeros(Size(height, width), CV_8UC1);
			bgr[1] = Mat::zeros(Size(height, width), CV_8UC1);
			bgr[2] = Mat::zeros(Size(height, width), CV_8UC1);

			// Cifar 10 is in RGB, while OpenCV expects BGR
			for (int channel = 2; channel >= 0; --channel) {
				for (int row = 0; row < height; ++row) {
					for (int column = 0; column < width; ++column) {
						bgr[channel].at<uchar>(row, column) = ((unsigned char) buffer[idx++]);
					}
				}
			}

			Mat image_uc3 = Mat::zeros(Size(height, width), CV_8UC3);
			merge(bgr, image_uc3);

			Mat image;

			image_uc3.convertTo(image, CV_32F, 1.0/255);

			patterns[added_patterns].image = image;

			added_patterns++;
		}

		input_file.close();

	} else {
		cerr << "Error opening file: " << filename << endl;
		exit(EXIT_FAILURE);
	}

	return patterns;

}


pair<vector<Pattern>, vector<Pattern>> Cifar10DataWrapper::load_train_test_data(unsigned int n_train_patterns, unsigned int n_test_patterns) {

	pair<vector<Pattern>, vector<Pattern>> train_test_patterns;

	vector<Pattern> train_patterns = load_patterns(train_file, n_train_patterns);
	vector<Pattern> test_patterns = load_patterns(test_file, n_test_patterns);

	// Check image shape
	if ((width != train_patterns[0].image.cols) || (height != train_patterns[0].image.rows) || (channels != train_patterns[0].image.channels())) {
		cerr << "Incorrect image dimensions. Found: " << train_patterns[0].image.cols << " " <<  train_patterns[0].image.rows << " " << train_patterns[0].image.channels() <<
				"while epecting " << width << " " << height << " " << channels << endl;
		exit(EXIT_FAILURE);
	}

	Mat mean_image = Mat::zeros(height, width, type);

	for (Pattern& p : train_patterns) {
		mean_image += p.image;
	}

	mean_image /= train_patterns.size();

	for (Pattern& p : train_patterns) {
		p.image -= mean_image;
	}

	for (Pattern& p : test_patterns) {
		p.image -= mean_image;
	}

	train_test_patterns.first = train_patterns;
	train_test_patterns.second = test_patterns;

	return train_test_patterns;

}


const int& Cifar10DataWrapper::get_height() const {
	return height;
}


const int& Cifar10DataWrapper::get_width() const {
	return width;
}


const int& Cifar10DataWrapper::get_channels() const {
	return channels;
}

const int& Cifar10DataWrapper::get_type() const {
	return type;
}


void Cifar10DataWrapper::create_train_test_files() const {

	int n_train_patterns = 2000;
	int n_test_patterns = 500;

	// Sample 2000 train patterns from the batch files. Force the creation of 200 samples for each class
	string batch_files[] = {
			"/home/fernando/data/cifar-10-binary/cifar-10-batches-bin/data_batch_1.bin",
			"/home/fernando/data/cifar-10-binary/cifar-10-batches-bin/data_batch_2.bin",
			"/home/fernando/data/cifar-10-binary/cifar-10-batches-bin/data_batch_3.bin",
			"/home/fernando/data/cifar-10-binary/cifar-10-batches-bin/data_batch_4.bin",
			"/home/fernando/data/cifar-10-binary/cifar-10-batches-bin/data_batch_5.bin"
	};

	vector<char> buffers[5];

	for (int i = 0; i < 5; ++i) {
		ifstream input_file(batch_files[i], ios::binary);
		if (input_file.is_open()) {
			buffers[i] = vector<char>(
					(istreambuf_iterator<char>(input_file)),
					istreambuf_iterator<char>());
		} else {
			cout << "Error opening input file " << batch_files[i] <<  endl;
			exit(EXIT_FAILURE);
		}
	}

	// Output file
	ofstream output_file("/home/fernando/data/cifar-10/training.bin", ios::binary);

	// Available images for selection
	vector<pair<int, int>> available_images;

	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < 10000; ++j) {
			available_images.push_back(pair<int, int>(i,j));
		}
	}


	// Track number of samples for each class
	map<int, int> images_per_class;

	int n_images = 0;

	while (n_images < n_train_patterns) {

		int chosen_sample = rand() % available_images.size();

		int buffer_idx = available_images[chosen_sample].first;
		int image_idx = available_images[chosen_sample].second;

		// Remove chosen sample so it does not get chosen again
		available_images.erase(available_images.begin() + chosen_sample);

		int pattern_length = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS + 1;

		int image_class = (int) buffers[buffer_idx][image_idx * pattern_length];

		auto found = images_per_class.find(image_class);

		if (found == images_per_class.end()) {
			output_file.write((char*) &buffers[buffer_idx][image_idx * pattern_length], pattern_length);
			images_per_class[image_class] = 1;
		} else {
			if (images_per_class[image_class] == (n_train_patterns / 10))
				continue;
			else {
				output_file.write((char*) &buffers[buffer_idx][image_idx * pattern_length], pattern_length);
				images_per_class[image_class]++;
			}
		}

		n_images++;

		cout << "MAP:" << endl;
		for (auto it = images_per_class.begin(); it != images_per_class.end(); ++it) {
			std::cout << it->first << ", " << it->second << endl;
		}
		cout << "###" << endl;

	}

	output_file.close();

	// Test patterns
	cout << "Creating test patterns..." << endl;

	ifstream input_file("/home/fernando/data/cifar-10-binary/cifar-10-batches-bin/test_batch.bin", ios::binary);

	vector<char> buffer;

	if (input_file.is_open()) {
		buffer = vector<char>((istreambuf_iterator<char>(input_file)), istreambuf_iterator<char>());
	} else {
		cout << "Error opening input file test_batch.bin" <<  endl;
		exit(EXIT_FAILURE);
	}

	// Output file
	output_file = ofstream("/home/fernando/data/cifar-10/test.bin", ios::binary);

	// Available images for selection
	vector<int> available_test_images(10000);

	for (int i = 0; i < 10000; ++i) {
		available_test_images[i] = i;
	}

	// Track number of samples for each class
	images_per_class = map<int, int>();

	n_images = 0;

	while (n_images < n_test_patterns) {

		int chosen_sample = rand() % available_test_images.size();

		int image_idx = available_test_images[chosen_sample];

		// Remove chosen sample so it does not get chosen again
		available_test_images.erase(available_test_images.begin() + chosen_sample);

		int pattern_length = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS + 1;

		int image_class = (int) buffer[image_idx * pattern_length];

		auto found = images_per_class.find(image_class);

		if (found == images_per_class.end()) {
			output_file.write((char*) &buffer[image_idx * pattern_length], pattern_length);
			images_per_class[image_class] = 1;
		} else {
			if (images_per_class[image_class] == (n_test_patterns / 10))
				continue;
			else {
				output_file.write((char*) &buffer[image_idx * pattern_length], pattern_length);
				images_per_class[image_class]++;
			}
		}

		n_images++;

		cout << "MAP:" << endl;
		for (auto it = images_per_class.begin(); it != images_per_class.end(); ++it) {
			std::cout << it->first << ", " << it->second << endl;
		}
		cout << "###" << endl;

	}

	output_file.close();

}


