#ifndef REST_MANAGER_H_
#define REST_MANAGER_H_

#include "db-handler.h"

#include <boost/asio.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string.hpp>

#include <curl/curl.h>

#include <iostream>
#include <string>
#include <sstream>
#include "ann-config.h"

using boost::asio::ip::tcp;
using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;

using namespace std;

const string GET_URL = "http://fmatt.pythonanywhere.com/get-cifar10/";
const string POST_URL = "http://fmatt.pythonanywhere.com/post-cifar10";

class RestManager {
public:
	RestManager();

	virtual ~RestManager();

	FitnessRecord get(const string& descriptor) const;

	FitnessRecord get_by_descriptor(const string& descriptor) const;

	void insert(const FitnessRecord& fr) const;

};


struct json_data {
	const char* data;
	size_t sizeleft;
};

#endif
