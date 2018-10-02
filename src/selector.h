#ifndef SELECTOR_HPP_
#define SELECTOR_HPP_

#include <vector>
#include "solution.h"

class Selector {
public:
	Selector();

	virtual ~Selector();

	Solution roulette_select(vector<Solution>& population);

	Solution inverted_roulette_select(vector<Solution>& population);

};

#endif
