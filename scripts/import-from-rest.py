#!/usr/bin/python
# -*- coding:utf-8 -*-

import psycopg2

if __name__ == "__main__":

    conn = psycopg2.connect('''host=ec2-54-83-1-94.compute-1.amazonaws.com 
                               dbname=ddok7b265enr9k 
                               user=zukisrgjximwnk 
                               password=9f886a8443bb219fdd3b159de44555bba1f9ce92beb975c3fe6004ad02475f98''')

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
        print("Inserting record " + str(i) + "...")
        cur_local.execute('''INSERT INTO fitness_patterns_cifar10 (descriptor, training_input, fitness, n_weights) 
                             VALUES (%s, %s, %s, %s)''',
                            (r[1], r[2], r[3], r[4]))
        conn_local.commit()
        i += 1

    cur_local.close()
    conn_local.close()





    print('Done.')