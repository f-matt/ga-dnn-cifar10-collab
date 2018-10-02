#ifndef ANN_RSM_H_
#define ANN_RSM_H_

#define CPU_ONLY
#define USE_OPENCV

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;


const int OUTPUT_SIZE = 1;

const int N_CLASSES = 10;

const int BATCH_SIZE = 50;

const int MAX_EPOCHS = 100;

const int N_TRAIN_PATTERNS = 2000;

const int N_TEST_PATTERNS = 500;

const int PATIENCE = 10;


/*
 * OpenCV image + label
 */
struct Pattern {
	Mat image;
	int label;
};

/*
 * Fitness record, used for storage in db
 */
struct FitnessRecord {
	int id;
	string descriptor;
	string training_input;
	float train_acc;
	float test_acc;
	long n_weights;
};

/*
 * Fitness prediction pattern
 */
struct FitnessPattern {
	int id;
	string descriptor;
	vector<float> input;
	float train_acc;
	float test_acc;
	long n_weights;
};

#endif
