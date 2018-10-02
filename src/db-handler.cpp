#include "db-handler.h"

using namespace std;


DbHandler::DbHandler() {}


DbHandler::~DbHandler() {}


/*
 * General callback function for inserts, updates and deletes
 */
static int callback(void *NotUsed, int argc, char **argv, char **azColName) {

	int i;

	for(i = 0; i < argc; ++i) {
		cout << azColName[i] << (argv[i] ? argv[i] : "NULL") << endl;
	}

	cout << endl;

	return 0;
}


/*
 * Queries the database for previously evaluated model.
 * Returns the stored fitness and weights (if found) or -1 (if not found)
 */
static int cb_get_by_descriptor(void *data, int argc, char **argv, char **azColName) {

	FitnessRecord* fr = (FitnessRecord*) data;

	// Already initialized record, skip
	if (fr->n_weights > 0) {
		return 0;
	}

	for(int i = 0; i < argc; ++i) {

		if (string(azColName[i]) == "id")
			fr->id = atoi(argv[i]);
		else if (string(azColName[i]) == "descriptor")
			fr->descriptor = argv[i];
		else if (string(azColName[i]) == "training_input")
			fr->training_input = argv[i];
		else if (string(azColName[i]) == "train_acc")
			fr->train_acc = atof(argv[i]);
		else if (string(azColName[i]) == "test_acc")
			fr->test_acc = atof(argv[i]);
		else if (string(azColName[i]) == "n_weights")
			fr->n_weights = atol(argv[i]);
		else {
			cerr << "Unknown column: " << azColName[i] << endl;
			exit(EXIT_FAILURE);
		}

	}

	return 0;
}

FitnessRecord DbHandler::get_by_descriptor(const string& descriptor) const {

	sqlite3 *db;
	char *zErrMsg = 0;
	int rc;

	FitnessRecord fr;

	fr.train_acc = -1;
	fr.test_acc = -1;
	fr.n_weights = -1;

	rc = sqlite3_open("db/ga-dnn.db", &db);

	if (rc) {
		cerr << "Can't open database: " << sqlite3_errmsg(db) << endl;
		exit(-1);
	} else {
		stringstream ss;
		ss << "SELECT id, descriptor, training_input, train_acc, test_acc, n_weights "
		   << "FROM fitness_patterns_cifar10 "
		   << "WHERE descriptor = '" << descriptor << "';";

		rc = sqlite3_exec(db, ss.str().c_str(), cb_get_by_descriptor, (void*) &fr, &zErrMsg);

		if( rc != SQLITE_OK ){
			cerr << "SQL error: " << zErrMsg << endl;
			sqlite3_free(zErrMsg);
			exit(EXIT_FAILURE);
		}

		sqlite3_close(db);
	}

	return fr;

}


void DbHandler::insert(const FitnessRecord& fr) const {

	sqlite3 *db;
	char *zErrMsg = 0;
	int rc;

	rc = sqlite3_open("db/ga-dnn.db", &db);

	if (rc) {
		cerr << "Can't open database: " << sqlite3_errmsg(db) << endl;
		exit(EXIT_FAILURE);
	} else {
		stringstream ss;
		ss	<< "INSERT INTO fitness_patterns_cifar10 (descriptor, training_input, train_acc, test_acc, n_weights) "
			<< "VALUES ('" << fr.descriptor << "', '" << fr.training_input << "', " << fr.train_acc
			<< ", " << fr.test_acc << ", " << fr.n_weights << ");";

		rc = sqlite3_exec(db, ss.str().c_str(), callback, 0, &zErrMsg);

		if( rc != SQLITE_OK ){
			cerr << "SQL error: " << zErrMsg << endl;
			sqlite3_free(zErrMsg);
			exit(EXIT_FAILURE);
		}

		sqlite3_close(db);
	}

}


static int cb_get_all(void *data, int argc, char **argv, char **azColName) {

	vector<FitnessRecord>* rset = (vector<FitnessRecord>*) data;

	FitnessRecord fr;

	for(int i = 0; i < argc; ++i) {

		if (string(azColName[i]) == "id")
			fr.id = atoi(argv[i]);
		else if (string(azColName[i]) == "descriptor")
			fr.descriptor = argv[i];
		else if (string(azColName[i]) == "training_input")
			fr.training_input = argv[i];
		else if (string(azColName[i]) == "train_acc")
			fr.train_acc = atof(argv[i]);
		else if (string(azColName[i]) == "test_acc")
			fr.test_acc = atof(argv[i]);
		else if (string(azColName[i]) == "n_weights")
			fr.n_weights = atol(argv[i]);
		else {
			cerr << "Unknown column: " << azColName[i] << endl;
			exit(EXIT_FAILURE);
		}

	}

	rset->push_back(fr);

	return 0;
}


vector<FitnessRecord> DbHandler::get_all() const {

	sqlite3 *db;
	char *zErrMsg = 0;
	int rc;

	vector<FitnessRecord> rset;

	rc = sqlite3_open("db/ga-dnn.db", &db);

	if (rc) {
		cerr << "Can't open database: " << sqlite3_errmsg(db) << endl;
		exit(EXIT_FAILURE);
	} else {
		stringstream ss;
		ss << "SELECT id, descriptor, training_input, train_acc, test_acc, n_weights "
		   << "FROM fitness_patterns_cifar10;";

		rc = sqlite3_exec(db, ss.str().c_str(), cb_get_all, (void*) &rset, &zErrMsg);

		if( rc != SQLITE_OK ){
			cerr << "SQL error: " << zErrMsg << endl;
			sqlite3_free(zErrMsg);
			exit(EXIT_FAILURE);
		}

		sqlite3_close(db);
	}

	return rset;

}


static int cb_get_pareto_front(void *data, int argc, char **argv, char **azColName) {

	vector<FitnessRecord>* rset = (vector<FitnessRecord>*) data;

	FitnessRecord fr;

	for(int i = 0; i < argc; ++i) {
		if (string(azColName[i]) == "id")
			fr.id = atoi(argv[i]);
		else if (string(azColName[i]) == "descriptor")
			fr.descriptor = argv[i];
		else if (string(azColName[i]) == "training_input")
			fr.training_input = argv[i];
		else if (string(azColName[i]) == "train_acc")
			fr.train_acc = atof(argv[i]);
		else if (string(azColName[i]) == "test_acc")
			fr.test_acc = atof(argv[i]);
		else if (string(azColName[i]) == "n_weights")
			fr.n_weights = atol(argv[i]);
		else {
			cerr << "Unknown column: " << azColName[i] << endl;
			exit(EXIT_FAILURE);
		}

	}

	rset->push_back(fr);

	return 0;
}


vector<FitnessRecord> DbHandler::get_pareto_front() const {

	sqlite3 *db;
	char *zErrMsg = 0;
	int rc;

	vector<FitnessRecord> rset;

	rc = sqlite3_open("db/ga-dnn.db", &db);

	if (rc) {
		cerr << "Can't open database: " << sqlite3_errmsg(db) << endl;
		exit(EXIT_FAILURE);
	} else {
		stringstream ss;
		ss << "SELECT id, descriptor, training_input, train_acc, test_acc, n_weights "
		   << "FROM fitness_patterns_cifar10 "
		   << "WHERE pareto_front = 'TRUE';";

		rc = sqlite3_exec(db, ss.str().c_str(), cb_get_pareto_front, (void*) &rset, &zErrMsg);

		if( rc != SQLITE_OK ){
			cerr << "SQL error: " << zErrMsg << endl;
			sqlite3_free(zErrMsg);
			exit(EXIT_FAILURE);
		}

		sqlite3_close(db);
	}

	return rset;

}


/*
 * Clear fitness records from previous executions
 */
void DbHandler::clear() const {

	sqlite3 *db;
	char *zErrMsg = 0;
	int rc;

	rc = sqlite3_open("db/ga-dnn.db", &db);

	if (rc) {
		cerr << "Can't open database: " << sqlite3_errmsg(db) << endl;
		exit(EXIT_FAILURE);
	} else {
		rc = sqlite3_exec(db, "DELETE FROM fitness_patterns_cifar10", callback, 0, &zErrMsg);

		if( rc != SQLITE_OK ){
			cerr << "SQL error: " << zErrMsg << endl;
			sqlite3_free(zErrMsg);
			exit(EXIT_FAILURE);
		}

		sqlite3_close(db);
	}

}


void DbHandler::clear_pareto_flag(int id) {

	sqlite3 *db;
	char *zErrMsg = 0;
	int rc;

	rc = sqlite3_open("db/ga-dnn.db", &db);

	if (rc) {
		cerr << "Can't open database: " << sqlite3_errmsg(db) << endl;
		exit(EXIT_FAILURE);
	} else {
		stringstream ss;

		ss << "UPDATE fitness_patterns_cifar10 SET pareto_front = 'FALSE' WHERE id = " << id;

		rc = sqlite3_exec(db, ss.str().c_str(), callback, 0, &zErrMsg);

		if( rc != SQLITE_OK ){
			cerr << "SQL error: " << zErrMsg << endl;
			sqlite3_free(zErrMsg);
			exit(EXIT_FAILURE);
		}

		sqlite3_close(db);
	}

}


void DbHandler::set_pareto_flag(int id) {

	sqlite3 *db;
	char *zErrMsg = 0;
	int rc;

	rc = sqlite3_open("db/ga-dnn.db", &db);

	if (rc) {
		cerr << "Can't open database: " << sqlite3_errmsg(db) << endl;
		exit(EXIT_FAILURE);
	} else {
		stringstream ss;

		ss << "UPDATE fitness_patterns_cifar10 SET pareto_front = 'TRUE' WHERE id = " << id;

		rc = sqlite3_exec(db, ss.str().c_str(), callback, 0, &zErrMsg);

		if( rc != SQLITE_OK ){
			cerr << "SQL error: " << zErrMsg << endl;
			sqlite3_free(zErrMsg);
			exit(EXIT_FAILURE);
		}

		sqlite3_close(db);
	}

}


void DbHandler::save_random_log(float train_acc, float test_acc, long weights, bool predicted) {

	sqlite3 *db;
	char *zErrMsg = 0;
	int rc;

	rc = sqlite3_open("db/ga-dnn.db", &db);

	if (rc) {
		cerr << "Can't open database: " << sqlite3_errmsg(db) << endl;
	} else {
		stringstream ss;
		ss << "INSERT INTO random_log (train_acc, test_acc, weights, predicted) " <<
			  "VALUES (" << train_acc << ", " << test_acc << ", " <<
			  weights << ", " << predicted << "); ";

		rc = sqlite3_exec(db, ss.str().c_str(), callback, 0, &zErrMsg);

		if( rc != SQLITE_OK ){
			cerr << "SQL error: " << zErrMsg << endl;
			sqlite3_free(zErrMsg);
		}

		sqlite3_close(db);
	}

}


void DbHandler::save_grid_log(float train_acc, float test_acc, long weights, bool predicted) {

	sqlite3 *db;
	char *zErrMsg = 0;
	int rc;

	rc = sqlite3_open("db/ga-dnn.db", &db);

	if (rc) {
		cerr << "Can't open database: " << sqlite3_errmsg(db) << endl;
	} else {
		stringstream ss;
		ss << "INSERT INTO grid_log (train_acc, test_acc, weights, predicted) " <<
			  "VALUES (" << train_acc << ", " << test_acc << ", " <<
			  weights << ", " << predicted << "); ";

		rc = sqlite3_exec(db, ss.str().c_str(), callback, 0, &zErrMsg);

		if( rc != SQLITE_OK ){
			cerr << "SQL error: " << zErrMsg << endl;
			sqlite3_free(zErrMsg);
		}

		sqlite3_close(db);
	}

}


void DbHandler::save_ga_log(int generation, float train_acc, float test_acc, long weights, bool predicted) {

	sqlite3 *db;
	char *zErrMsg = 0;
	int rc;

	rc = sqlite3_open("db/ga-dnn.db", &db);

	if (rc) {
		cerr << "Can't open database: " << sqlite3_errmsg(db) << endl;
	} else {
		stringstream ss;
		ss << "INSERT INTO ga_log (generation, train_acc, test_acc, weights, predicted) " <<
			  "VALUES (" << generation << ", " << train_acc << ", " << test_acc << ", " <<
			  weights << ", " << predicted << "); ";

		rc = sqlite3_exec(db, ss.str().c_str(), callback, 0, &zErrMsg);

		if( rc != SQLITE_OK ){
			cerr << "SQL error: " << zErrMsg << endl;
			sqlite3_free(zErrMsg);
		}

		sqlite3_close(db);
	}

}

