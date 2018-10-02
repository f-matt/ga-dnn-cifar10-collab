#include "../solution.h"

#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace std;


/*
 * Test the weight count of a solution
 */
void weight_count_test() {

	boost::shared_ptr<Cifar10DataWrapper> data_wrapper(new Cifar10DataWrapper());

	Solution s(data_wrapper, "CS;2;9;3;2");

	BOOST_CHECK_EQUAL(1478, s.get_weights());

}


/*
 * Test the weight update after mutation
 */
void weight_count_after_mutation() {

	boost::shared_ptr<Cifar10DataWrapper> data_wrapper(new Cifar10DataWrapper());

	Solution s(data_wrapper, "CS;2;9;3;2");

	BOOST_CHECK_EQUAL(1478, s.get_weights());

	s.mutate_conv();

	BOOST_CHECK_EQUAL(818, s.get_weights());

}



