#!/usr/bin/python
# -*- coding:utf-8 -*-

import psycopg2

if __name__ == "__main__":

    conn = psycopg2.connect("host=localhost " 
                            "dbname=ga_dnn "
                            "user=postgres "
                            "password=123456")

    cur = conn.cursor()

    cur.execute("SELECT descriptor, training_input, fitness, n_weights FROM fitness_patterns_cifar10;")
    rset = cur.fetchall()

    cur.close()
    conn.close()

    lines = list()

    for r in rset:
        line = ("INSERT INTO fitness_patterns_cifar10 (descriptor, training_input, fitness, n_weights) "
                "VALUES ('{0}', '{1}', '{2}', '{3}');\n").format(r[0], r[1], r[2], r[3])

        lines.append(line)

    fid = open("inserts.sql", "w")
    fid.writelines(lines)
    fid.close()

    print('Done.')
