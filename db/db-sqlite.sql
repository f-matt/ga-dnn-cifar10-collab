CREATE TABLE IF NOT EXISTS fitness_patterns_cifar10 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    descriptor varchar(200) UNIQUE NOT NULL,
    training_input varchar(300),
    train_acc float NOT NULL,
    test_acc float NOT NULL,
    n_weights bigint NOT NULL,
    pareto_front boolean NOT NULL DEFAULT FALSE
);


CREATE TABLE IF NOT EXISTS random_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    train_acc float NOT NULL,
    test_acc float NOT NULL,
    weights bigint NOT NULL,
    predicted boolean NOT NULL DEFAULT FALSE
);


CREATE TABLE IF NOT EXISTS grid_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    train_acc float NOT NULL,
    test_acc float NOT NULL,
    weights bigint NOT NULL,
    predicted boolean NOT NULL DEFAULT FALSE
);


CREATE TABLE IF NOT EXISTS ga_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation integer NOT NULL,
    train_acc float NOT NULL,
    test_acc float NOT NULL,
    weights bigint NOT NULL,
    predicted boolean NOT NULL DEFAULT FALSE
);

