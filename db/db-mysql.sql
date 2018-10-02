CREATE DATABASE ga_dnn;


--
-- REST
--
CREATE TABLE IF NOT EXISTS fitness_patterns_cifar10 (
    id SERIAL PRIMARY KEY,
    descriptor varchar(200) UNIQUE NOT NULL,
    training_input varchar(300),
    train_acc float NOT NULL,
    test_acc float NOT NULL,
    n_weights bigint NOT NULL,
    pareto_front boolean NOT NULL DEFAULT FALSE
);

