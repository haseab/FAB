import mysql.connector
import json
import asyncio
import asyncpg
import pandas as pd
import psycopg2

class SqlMapper:
    def __init__(self):
        with open('fab_db_connect.txt', 'r') as file:
            self.kwargs = json.load(file)
        self.conn = None
        self.pool = None

    async def connect_pool(self):
        self.pool = await asyncpg.create_pool(**self.kwargs, command_timeout=60)
        return f"Connected to Postgres Database: {self.kwargs['database']} with Connection Pool"

    async def connect_psql(self, asynchronous=True):
        self.conn = await asyncpg.connect(**self.kwargs)
        return self.conn

    async def CREATE_TABLE(self, name, *params):
        string = f"""CREATE TABLE {name} ("""
        for i in params:
            string += i + ","
        string = string[:-1] + ");"
        await self.conn.fetch(string)

        return string

    async def get_all_table_names(self):
        string = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = " \
                 "'BASE TABLE' AND TABLE_SCHEMA = 'public';"
        results = await self.conn.fetch(string)
        return self.psql_to_list(results)

    async def get_columns(self, table_name):
        string = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}' " \
                 f"ORDER BY ORDINAL_POSITION;"

        myresult = await self.conn.fetch(string)
        myresult = self.psql_to_list(myresult)
        return myresult

    async def DELETE_TABLE(self, table_name):
        string = f"DROP TABLE {table_name};"
        await self.conn.fetch(string)
        return f"{table_name} Table Dropped"

    # Check this
    async def clear_table(self, table_name):
        string = f"DELETE FROM {table_name}"
        await self.conn.fetch(string)
        return f"{table_name} Table cleared."

    async def len_table(self, table_name):
        string = f"SELECT COUNT(*) FROM {table_name}"
        count = await self.conn.fetch(string)
        count = self.psql_to_list(count)[0]
        return count

    async def BULK_INSERT(self, table_name, columns, values):
        """Note: This function is NOT considering the ID when adding"""
        string_list = []
        string_list.append(f"INSERT INTO {table_name}(")
        string_list.append(', '.join(columns[1:]) + ") VALUES ")
        string_list.append(values)
        string = "".join(string_list) + ";"
        await self.conn.fetch(string)
        return string

    async def write_to_db(self, df, table_name, batch_size = 10000):
        columns = await self.get_columns(table_name)
        list_values = df.values.tolist()

        for i in range(0, len(df) - 1, batch_size):
            df_list = list_values[i:i + batch_size]
            values = ", ".join([str(tuple(i)) for i in df_list])
            results = await self.BULK_INSERT(table_name, columns, values)
        return "Write Done"

    async def INSERT_INTO(self, table_name, columns, *values):
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

        await self.conn.fetch(string)
        return string


    async def SELECT(self, statement):
        data = await self.conn.fetch("SELECT " + statement)
        return self.psql_to_df(data)

    @staticmethod
    def psql_to_list(psql_list):
        return list(map(lambda x: list(x.values())[0], psql_list))

    @staticmethod
    def psql_to_df(psql_list):
        try:
            columns = list(psql_list[0].keys())
        except Exception as e:
            return "Table is Empty"

        return pd.DataFrame(map(lambda x: list(x.values()), psql_list), columns=columns).set_index(columns[0])

    async def ASYNC_SELECT(self, statement, table_name, batch_size, count = None):
        statements = []

        if not count:
            count = await self.len_table(table_name)

        for offset in range(0, count-1, batch_size):
            new_statement = "".join([statement, f" LIMIT {batch_size}", f" OFFSET {offset}"])
            statements.append(new_statement)
        commands = [self.pool.fetch(statement) for statement in statements]

        results = await asyncio.gather(*commands)

        return results
