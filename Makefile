CC = g++ -std=c++11 -Wall
LIBS = -pthread -lm -lboost_system -lboost_filesystem -lboost_date_time -lboost_system -lboost_thread \
	-lcaffe -lglog -lgflags -lprotobuf -lsqlite3 `pkg-config --libs opencv libcurl jsoncpp`
OBJS = build/descriptors.o build/population.o build/selector.o build/solution.o \
	build/fitness-data-wrapper.o build/fitness-regressor.o build/db-handler.o build/rest-manager.o \
	build/cifar10-data-wrapper.o build/cifar10-classifier.o build/utils.o
 
all: build/main

clean:
	rm -f build/*
	
build/main: build/main.o $(OBJS)
	$(CC) -o build/main build/main.o $(OBJS) $(LIBS)
	
build/main.o: src/main.cpp
	$(CC) -o build/main.o -c src/main.cpp
	
build/descriptors.o: src/descriptors.cpp src/descriptors.h
	$(CC) -o build/descriptors.o -c src/descriptors.cpp
	
build/population.o: src/population.cpp src/population.h
	$(CC) -o build/population.o -c src/population.cpp	

build/solution.o: src/solution.cpp src/solution.h
	$(CC) -o build/solution.o -c src/solution.cpp

build/selector.o: src/selector.cpp src/selector.h
	$(CC) -o build/selector.o -c src/selector.cpp	

build/fitness-data-wrapper.o: src/fitness-data-wrapper.cpp src/fitness-data-wrapper.h
	$(CC) -o build/fitness-data-wrapper.o -c src/fitness-data-wrapper.cpp	
	
build/fitness-regressor.o: src/fitness-regressor.cpp src/fitness-regressor.h src/regressor.h
	$(CC) -o build/fitness-regressor.o -c src/fitness-regressor.cpp	
	
build/db-handler.o: src/db-handler.cpp src/db-handler.h
	$(CC) -o build/db-handler.o -c src/db-handler.cpp
	
build/rest-manager.o: src/rest-manager.cpp src/rest-manager.h
	$(CC) -o build/rest-manager.o -c src/rest-manager.cpp		

build/cifar10-data-wrapper.o: src/cifar10-data-wrapper.cpp src/cifar10-data-wrapper.h
	$(CC) -o build/cifar10-data-wrapper.o -c src/cifar10-data-wrapper.cpp	
	
build/cifar10-classifier.o: src/cifar10-classifier.cpp src/cifar10-classifier.h src/classifier.h 
	$(CC) -o build/cifar10-classifier.o -c src/cifar10-classifier.cpp
	
build/utils.o: src/utils.cpp src/utils.h
	$(CC) -o build/utils.o -c src/utils.cpp
	
# Unit tests
TEST_OBJS = build/main-test.o build/descriptors-test.o build/cifar10-data-wrapper-test.o \
	build/solution-test.o build/cifar10-classifier-test.o	build/population-test.o

tests: $(OBJS) $(TEST_OBJS) 
	${CC} -o build/tests $(OBJS) $(TEST_OBJS) -lboost_unit_test_framework $(LIBS)	

build/main-test.o: src/test/main-test.cpp
	$(CC) -o build/main-test.o -c src/test/main-test.cpp
	
build/descriptors-test.o: src/test/descriptors-test.cpp
	$(CC) -o build/descriptors-test.o -c src/test/descriptors-test.cpp

build/cifar10-data-wrapper-test.o: src/test/cifar10-data-wrapper-test.cpp
	$(CC) -o build/cifar10-data-wrapper-test.o -c src/test/cifar10-data-wrapper-test.cpp
		
build/solution-test.o: src/test/solution-test.cpp
	$(CC) -o build/solution-test.o -c src/test/solution-test.cpp
	
build/cifar10-classifier-test.o: src/test/cifar10-classifier-test.cpp
	$(CC) -o build/cifar10-classifier-test.o -c src/test/cifar10-classifier-test.cpp	
	
build/population-test.o: src/test/population-test.cpp
	$(CC) -o build/population-test.o -c src/test/population-test.cpp
