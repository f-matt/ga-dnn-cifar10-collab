#include "../ann-config.h"

#include <caffe/caffe.hpp>
#include <boost/test/included/unit_test.hpp>
#include <iostream>

using namespace std;
using namespace boost::unit_test;

// Descriptors
void import_descriptor();
void import_descriptor_random();
void output_vector();

// Data wrapper
void load_cifar10_train_test_data();

// Solution
void weight_count_test();
void weight_count_after_mutation();

// Classifier
void cifar10_classifier_creation_test();
void cifar10_classifier_n_weights_test();

// Population
void memory_consumption_test();

test_suite* init_unit_test_suite(int argc, char* argv[]) {

	srand(7);

	google::InitGoogleLogging("GA-DL");
	google::SetCommandLineOption("GLOG_minloglevel", "1");

	test_suite* descriptors_suite = BOOST_TEST_SUITE("descriptors");
	descriptors_suite->add(BOOST_TEST_CASE(&import_descriptor));
	descriptors_suite->add(BOOST_TEST_CASE(&import_descriptor_random));
	descriptors_suite->add(BOOST_TEST_CASE(&output_vector));

	test_suite* data_wrapper_suite = BOOST_TEST_SUITE("data_wrapper");
	data_wrapper_suite->add(BOOST_TEST_CASE(&load_cifar10_train_test_data));

	test_suite* solution_suite = BOOST_TEST_SUITE("solution");
	solution_suite->add(BOOST_TEST_CASE(&weight_count_test));
	solution_suite->add(BOOST_TEST_CASE(&weight_count_after_mutation));

	test_suite* classifier_suite = BOOST_TEST_SUITE("classifier");
	classifier_suite->add(BOOST_TEST_CASE(&cifar10_classifier_creation_test));
	classifier_suite->add(BOOST_TEST_CASE(&cifar10_classifier_n_weights_test));


	test_suite* population_suite = BOOST_TEST_SUITE("population");
	population_suite->add(BOOST_TEST_CASE(&memory_consumption_test));

	framework::master_test_suite().add(descriptors_suite);
	framework::master_test_suite().add(data_wrapper_suite);
	framework::master_test_suite().add(solution_suite);
	framework::master_test_suite().add(classifier_suite);
	framework::master_test_suite().add(population_suite);

	return 0;
}

