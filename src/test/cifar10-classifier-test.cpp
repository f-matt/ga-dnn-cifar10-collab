#include "../ann-config.h"
#include "../cifar10-classifier.h"

#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace std;


/*
 * Test the creation of a classifier
 */
void cifar10_classifier_creation_test() {

	string descriptor = "CR;32;11;4;2-FR;16;0.5";

	TopologyDescriptor topology_descriptor(descriptor);

	BOOST_CHECK_EQUAL( topology_descriptor.to_string(), descriptor );

	boost::shared_ptr<Cifar10DataWrapper> data_wrapper(new Cifar10DataWrapper);

	Cifar10Classifier classifier(topology_descriptor,
			data_wrapper,
			N_CLASSES,
			BATCH_SIZE,
			MAX_EPOCHS,
			N_TRAIN_PATTERNS,
			N_TEST_PATTERNS,
			PATIENCE);

	BOOST_CHECK_EQUAL(24634, classifier.get_n_weights());

}


/*
 * Test the number of weights of a given topology
 */
void cifar10_classifier_n_weights_test() {

	string descriptor = "CR;32;11;4;2-FR;16;0.5";

	TopologyDescriptor topology_descriptor(descriptor);

	boost::shared_ptr<Cifar10DataWrapper> data_wrapper(new Cifar10DataWrapper);

	Cifar10Classifier classifier(topology_descriptor,
			data_wrapper,
			N_CLASSES,
			BATCH_SIZE,
			MAX_EPOCHS,
			N_TRAIN_PATTERNS,
			N_TEST_PATTERNS,
			PATIENCE);

	BOOST_CHECK_EQUAL(24634, classifier.get_n_weights());


}


