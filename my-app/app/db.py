import mysql.connector


"""
    DB
"""
mydb = mysql.connector.connect(
    host="0.0.0.0",
    user="root",
    passwd="toor",
    database="db"
)


def select_all():

    cur = mydb.cursor()
    sql = '''SELECT * FROM data'''  # 조회 SQL

    cur.execute(sql)
    select_all_result = cur.fetchall()

    for x in select_all_result:
        print(x)
