#!/usr/bin/python
# -*- coding:utf-8 -*-

import psycopg2

if __name__ == "__main__":

    conn = psycopg2.connect('''host=localhost 
                               dbname=ga_dnn_2 
                               user=postgres 
                               password=123456''')

    cur = conn.cursor()

    cur.execute("SELECT * FROM fitness_patterns_cifar10;")
    rset = cur.fetchall()

    cur.close()
    conn.close()

    conn_local = psycopg2.connect('''host=localhost 
                                     dbname=ga_dnn 
                                     user=postgres
                                     password=123456''')

    cur_local = conn_local.cursor()

    i = 1

    for r in rset:

        cur_local.execute("SELECT COUNT(*) "
                          "FROM fitness_patterns_cifar10 "
                          "WHERE descriptor = %s", (r[1],))

        count_result = cur_local.fetchone()

        if count_result[0] > 0:
            continue

        print("Inserting record " + str(i) + "...")
        cur_local.execute('''INSERT INTO fitness_patterns_cifar10 (descriptor, training_input, fitness, n_weights) 
                             VALUES (%s, %s, %s, %s)''',
                            (r[1], r[2], r[3], r[4]))
        conn_local.commit()
        i += 1

    cur_local.close()
    conn_local.close()

    print('Done.')