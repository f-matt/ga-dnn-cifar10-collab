#include "../descriptors.h"

#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace boost::unit_test;
using namespace std;


/*
 * Test the creation of a topology descriptor from spec string
 */
void import_descriptor() {

	string descriptor = "CSN;32;11;4;2-FS;8;0.6-FL;16;0.2";

	TopologyDescriptor topology_descriptor(descriptor);

	BOOST_CHECK_EQUAL( topology_descriptor.to_string(), descriptor );

}


/*
 * Test the randomization of a topology descriptor built from a spec string
 */
void import_descriptor_random() {

	string descriptor = "CSN;32;11;4;2-FS;8;0.6-FL;16;0.2";

	string expected_descriptor = "CSN;32;11;4;2-FS;8;0-FL;16;0";

	TopologyDescriptor topology_descriptor(descriptor);

	topology_descriptor.randomize_norm();

	BOOST_CHECK_EQUAL( topology_descriptor.to_string(), expected_descriptor );

}


/*
 * Test the generation of the output vector for different topologies
 */
void output_vector() {

	TopologyDescriptor td01("CL;8;5;1;2");
	TopologyDescriptor td02("FR;8;0.2");
	TopologyDescriptor td03("CL;8;5;1;2-FR;8;0.2");

	BOOST_CHECK_EQUAL( OUTPUT_VECTOR_SIZE, td01.get_output_vector().size() );
	BOOST_CHECK_EQUAL( OUTPUT_VECTOR_SIZE, td02.get_output_vector().size() );
	BOOST_CHECK_EQUAL( OUTPUT_VECTOR_SIZE, td03.get_output_vector().size() );

}

