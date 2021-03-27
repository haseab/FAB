import mysql.connector
import json
import asyncio
import asyncpg

class SqlMapper:
    def __init__(self):
        with open('fab_db_connect.txt', 'r') as file:
            self.kwargs = json.load(file)
        self.cursor = None
        self.conn = None
        self.psql = False

    def CONNECT(self, psql=False):s
        if psql:
            self.psql = True
            self.conn = await asyncpg.connect(**self.kwargs)
        else:
            self.mydb = mysql.connector.connect(**self.kwargs)
        return "Connected"

    def CREATE_TABLE(self, name, *params, cursor=None):
        string = f"""CREATE TABLE {name} ("""
        for i in params:
            string += i + ","
        string = string[:-1] + ");"
        cursor.execute(string)
        return string

    def get_columns(self, table_name, cursor=None):
        cursor.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}';")
        myresult = cursor.fetchall()
        myresult = [column[0] for column in myresult]
        return myresult

    def DELETE_TABLE(self, table_name, cursor=None):
        cursor.execute(f"DROP TABLE {table_name};")

    def clear_table(self, table_name, cursor=None):
        cursor.execute(f"DELETE FROM {table_name}")

    def len_table(self, table_name, cursor=None):
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchall()[0][0]
        return count

    def BULK_INSERT(self, table_name, columns, values, cursor = None):
        """Note: This function is NOT considering the ID when adding"""
        string_list = []
        string_list.append(f"INSERT INTO {table_name}(")
        string_list.append(', '.join(columns[1:]) + ") VALUES ")
        string_list.append(values)
        string = "".join(string_list) + ";"
        cursor.execute(string)
        self.mydb.commit()
        return string

    def write_to_db(self, df, table_name, batch_size = 10000):
        columns = self.get_columns(table_name)
        list_values = df.values.tolist()

        for i in range(0, len(df) - 1, batch_size):
            df_list = list_values[i:i + batch_size]
            values = ", ".join([str(tuple(i)) for i in df_list])
            results = self.BULK_INSERT(table_name, columns, values)
        return "Write Done"


    def INSERT_INTO(self, table_name, columns, *params, cursor = None):
        """Note: This function is NOT considering the ID when adding"""
        string_list = []
        string_list.append(f"INSERT INTO {table_name}(")
        string_list.append(', '.join(columns[1:]) + ") VALUES(")

        for i in params[:-1]:
            string_list.append(f"'{i}'" + ",")
        string_list.append(f'{i}')
        string_list.append(");")
        string = "".join(string_list)

        if self.psql:
            await self.conn.fetch(string)
            return string

        else:
            cursor.execute(string)
            self.mydb.commit()
            return string

    def SELECT(self, statement, cursor=None):
        if self.psql:
            return await self.conn.fetch(statement)
        else:
            cursor.execute("SELECT" + statement)
            return cursor.fetchall()

