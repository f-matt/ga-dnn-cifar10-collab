#ifndef FITNESS_REGRESSOR_H_
#define FITNESS_REGRESSOR_H_

#include "regressor.h"
#include "fitness-data-wrapper.h"

const int INPUT_VECTOR_LENGTH = 39;

const int OUTPUT_VECTOR_LENGTH = 1;


class FitnessRegressor : public Regressor<FitnessPattern> {
public:
	FitnessRegressor(const TopologyDescriptor &descriptor,
			int output_size,
			int batch_size,
			int max_epochs,
			int n_training_patterns,
			int n_test_patterns,
			int patience);

	FitnessRegressor(const TopologyDescriptor &descriptor,
			int output_size,
			int batch_size,
			int max_epochs,
			int patience);

	virtual ~FitnessRegressor();

	void train();

	float train_with_batch(const vector<FitnessPattern>& batch);

	void regress(const FitnessPattern& input, vector<float> &pred, float &loss);

	void regress(FitnessPattern& input);

	bool is_ready();

private:
	void add_input_layer();

	bool ready;

};

#endif /* FITNESS_REGRESSOR_H_ */
