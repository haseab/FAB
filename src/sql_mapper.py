import mysql.connector
import json
import asyncio
import asyncpg
import pandas as pd
import psycopg2
import time

class SqlMapper:
    def __init__(self):
        with open(r'C:\Users\haseab\Desktop\Python\PycharmProjects\FAB\local\fab_db_connect.txt', 'r') as file:
            self.kwargs = json.load(file)
        self.cursor = None
        self.conn = None
        self.psql = False

    def connect_psql(self):
        self.psql = True
        if not self.conn:
            self.conn = psycopg2.connect(**self.kwargs)
            print(f"Connected to Postgres Database: {self.kwargs['database']}")
        return self.conn

    def connect_mysql(self):
        self.psql = False
        self.conn = mysql.connector.connect(**self.kwargs)
        print(f"Connected to MySQL Database: {self.kwargs['database']}")
        return self.conn

    def CREATE_TABLE(self, name, *params, cursor=None):
        string = f"""CREATE TABLE {name} ("""
        for i in params:
            string += i + ","
        string = string[:-1] + ");"

        cursor.execute(string)
        self.conn.commit()
        return string

    def get_all_table_names(self, cursor=None):
        if self.psql:
            string = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' " \
                     "AND TABLE_SCHEMA = 'public';"
        else:
            string = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' " \
                     "AND TABLE_SCHEMA = 'fab';"
        cursor.execute(string)
        results = cursor.fetchall()
        results = [table_name[0] for table_name in results]
        return results

    def get_columns(self, table_name, cursor=None):
        string = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}' " \
                 f"ORDER BY ORDINAL_POSITION;"
        cursor.execute(string)
        myresult = cursor.fetchall()
        myresult = [column[0] for column in myresult]
        return myresult

    def DELETE_TABLE(self, table_name, cursor=None):
        string = f"DROP TABLE {table_name};"

        cursor.execute(string)
        self.conn.commit()
        return f"{table_name} Table Dropped"

    # Check this
    def clear_table(self, table_name, cursor=None):
        string = f"DELETE FROM {table_name}"

        cursor.execute(string)
        self.conn.commit()
        return f"{table_name} Table cleared."

    def count_rows(self, table_name, cursor=None):
        string = f"SELECT COUNT(*) FROM {table_name}"
        cursor.execute(string)
        count = cursor.fetchall()[0][0]
        return count

    def BULK_INSERT(self, table_name, columns, values, cursor = None, first_column=False):
        """Note: This function is NOT considering the ID when adding"""
        string_list = []
        string_list.append(f"INSERT INTO {table_name}(")
        if first_column:
            string_list.append(', '.join(columns) + ") VALUES ")
        else:
            string_list.append(', '.join(columns[1:]) + ") VALUES ")
        string_list.append(values)
        string = "".join(string_list) + ";"

        cursor.execute(string)
        self.conn.commit()
        return string

    def write_to_db(self, df, table_name, batch_size = 10000, cursor=None, first_column=False, columns=None):
        if not columns:
            columns = self.get_columns(table_name, cursor=cursor)
        list_values = df.values.tolist()

        for i in range(0, len(df) - 1, batch_size):
            df_list = list_values[i:i + batch_size]
            values = ", ".join([str(tuple(i)) for i in df_list])
            time.sleep(0.2)
            results = self.BULK_INSERT(table_name, columns, values, cursor, first_column)
        return "Write Done"


    def INSERT_INTO(self, table_name, columns, *values, cursor = None):
        """Note: This function is NOT considering the ID when adding"""
        string_list = []
        string_list.append(f"INSERT INTO {table_name}(")
        string_list.append(', '.join(columns[1:]) + ") VALUES(")
        string = ""

        for i in values[:-1]:
            string_list.append(f"'{i}'" + ",")

        string_list.append(f"'{values[-1]}'")
        string_list.append(");")
        string = "".join(string_list)


        cursor.execute(string)
        self.conn.commit()
        return string

    def sql_resp_into_df(self, columns, sql_list):
        return pd.DataFrame(sql_list, columns=columns)

    def get_active_connections(self, cursor):
        return self.SELECT("* FROM pg_stat_activity WHERE state = 'active' OR state = 'idle'", cursor)

    # def disconnect_all_connections(self, cursor):
    #     cursor.execute(f"""SELECT pg_terminate_backend(pg_stat_activity.pid)
    #                        FROM pg_stat_activity
    #                        WHERE datname = 'fab'
    #                        AND pid <> pg_backend_pid();""")
    #     return cursor.fetchall()

    def kill_all_connections(self, cursor):
        resp = []
        active_connections = self.get_active_connections(cursor)[1:]
        for pid in active_connections['pid']:
            cursor.execute(f"SELECT pg_terminate_backend({pid}) from pg_stat_activity")
            resp.append(cursor.fetchall())
        return resp

    def get_table_name_from_string(self, sql_statement, table_name = None):
        statement_list = sql_statement.split(" ")
        for i in range(len(statement_list)):
            if statement_list[i] in ["From", "FROM", "from"]:
                table_name = statement_list[i + 1]
        if not table_name:
            return "Table name is not after FROM statement"
        return table_name

    def SELECT(self, statement, cursor=None, psql = True):
        cursor.execute("SELECT " + statement)
        results = cursor.fetchall()
        if psql:
            columns = [cursor.description[i].name for i in range(len(cursor.description))]
        else:
            columns = [cursor.description[i][0] for i in range(len(cursor.description))]
        return self.sql_resp_into_df(columns, results)

    def UPDATE(self, table_name, indexing_column, indexing_row, column, new_value, cursor = None, bulk = False):
        string = f"UPDATE {table_name} set {column} = {new_value} where {indexing_column} = {indexing_row} "
        cursor.execute(string)
        if not bulk:
            self.conn.commit()
        return string
