#include "../cifar10-data-wrapper.h"

#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace boost::unit_test;
using namespace std;


/*
 * Test the loading of train and test data
 */
void load_cifar10_train_test_data() {

	Cifar10DataWrapper data_wrapper;

	pair<vector<Pattern>, vector<Pattern>> train_test_data = data_wrapper.load_train_test_data(2000, 500);

	BOOST_CHECK_EQUAL(2000, train_test_data.first.size());
	BOOST_CHECK_EQUAL(500, train_test_data.second.size());

}
