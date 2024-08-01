import sqlite3
import matplotlib.pyplot as plt
from tabulate import tabulate # for latex table generation

def db_setup(dbfile):
    ''' 
    set up a sqlite3 database for writing results.
    
    @param dbfile: path to database to connect to. Will be created if does not exist
    
    @return a pair of database connection and sqlite3 cursor object for 
            interacting with the database
    '''
    print(dbfile)
    connection = sqlite3.connect(dbfile);
    sqlcursor = connection.cursor()
    return connection, sqlcursor


def db_disconnect(connection, sqlcursor):
    '''
    close database connections
    
    @param connection: connection to the database to be closed
    @param sqlcursor: cursor to the database to be closed; must 
                      be from the same database as the connection param
    '''
    assert sqlcursor.connection == connection, "unable to close database correctly"
    
    sqlcursor.close()
    connection.close()


def db_create_table(sqlcursor, table_name, defs):
    '''
    Create a table using the given cursor
    
    @param sqlcursor: cursor to a sqlite3 database in which to create table
    @param table_name: name of table to create
    @param defs: col_name="DATA_TYPE" dict of all columns to include in this table
    '''
    # Check if table exists
    sqlcursor.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='%s' ''' % table_name)
    if sqlcursor.fetchone()[0]==1 :
        print("table %s already exists" % table_name)
        #exit(1)
        return
    
    cols = ', '.join((' '.join((n.lower(), t.upper())) for n,t in defs.items()))
    sqlcursor.execute("CREATE TABLE " + table_name + " (" + cols + ")")


def table_insert(sqlcursor, table_name, data):
    '''
    insert a record into the given table
    TODO: assertions to validate table cols against data
    
    @param sqlcursor:
    @param table_name:
    @param data: dict of data to be inserted for a single row
    '''
    cols, vals = zip(*data.items())
    sql_cmd = "INSERT INTO %s ('%s') VALUES (" % (table_name, "', '".join(cols))
    
    for v in vals:
        if isinstance(v,str):
            sql_cmd += "'" + v + "',"
        if isinstance(v,(int, float, complex)):
            sql_cmd += str(v) + ","
    
    # remove extra comma
    sql_cmd = sql_cmd[:-1] + ")"
    sqlcursor.execute(sql_cmd)


def table_select(sqlcursor, table_name):
    '''
    select all records in table
    
    @param sqlcursor:
    @param table_name:
    '''
    sql_cmd = "SELECT * FROM %s" % table_name
    
    sqlcursor.execute(sql_cmd)
    
    records = sqlcursor.fetchall()
    
    for r in records:
        print(r)


def check_completed_codes(sqlcursor):
    sql_cmd = "SELECT code FROM indigo"
    _, records = exec_query(sqlcursor, sql_cmd)
    
    return list(map(lambda x: x[0], records))


def exec_query(sqlcursor, query_string):
    '''
    get rows from user query
    
    @param sqlcursor:
    @param query_string: string containing query
    
    @return a list of tuples, one tuple per record.
    '''
    sqlcursor.execute(query_string)
    fields = list(map(lambda x: x[0], sqlcursor.description))
    
    records = sqlcursor.fetchall()
    
    return fields, records


def generate_tex_table(fields, records):
    '''
    generates a latex table from a list of records
    
    @records: a list of iterable records
    '''
    print('\n==============================================\n')
    print(tabulate(records, headers=fields, tablefmt='pretty'))
    print('\n==============================================\n')
    print(tabulate(records, headers=fields, tablefmt='latex'))
    print('\n==============================================\n')


def plot_lines(tool_results):
    '''
    plots line graph for each (tool, graph input) combo.
    
    @param tool_results: dict with key (tool, graph) and value list of tuple records
    '''
    for k, v in tool_results.items():
        data = list(zip(*v))
        plt.plot(data[0], data[1], label=k[0] + ' ' + k[1]) # currently data 0 is code, 1 is timing
    
    plt.xticks([])
    # plt.legend()
    plt.savefig('tool_results.png', bbox_inches='tight')
