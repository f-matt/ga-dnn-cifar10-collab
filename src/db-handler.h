#ifndef DB_HANDLER_H_
#define DB_HANDLER_H_

#include "ann-config.h"

#include <sqlite3.h>
#include <string>
#include <iostream>

using namespace std;


class DbHandler {

public:
	DbHandler();

	virtual ~DbHandler();

	FitnessRecord get_by_descriptor(const string& descriptor) const;

	vector<FitnessRecord> get_all() const;

	vector<FitnessRecord> get_pareto_front() const;

	void insert(const FitnessRecord& fr) const;

	void clear() const;

	void clear_pareto_flag(int id);

	void set_pareto_flag(int id);

	void save_random_log(float train_acc, float test_acc, long weights, bool predicted);

	void save_grid_log(float train_acc, float test_acc, long weights, bool predicted);

	void save_ga_log(int generation, float train_acc, float test_acc, long weights, bool predicted);

};


#endif
