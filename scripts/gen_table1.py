import os
import sys

import util

DATABASE = ''

def main():
    connection, sqlcursor = util.db_setup(DATABASE)
    try:
        # query for err nums
        query = '''
        SELECT
            i.graph as "Input Graph",
            COUNT(DISTINCT i.code) AS "Total Tests",
            SUM(CASE WHEN hirace.errors > 0 THEN 1 ELSE 0 END) AS "Hirace Races Found",
            SUM(CASE WHEN hirace.errors = 0 AND i.code LIKE "%Bug%" AND i.code NOT LIKE "%boundsBug%" THEN 1 ELSE 0 END) AS "Hirace Races Missed",
            SUM(CASE WHEN g.errors > 0 THEN 1 ELSE 0 END) AS "iGUARD Races Found",
            SUM(CASE WHEN g.errors = 0 AND i.code LIKE "%Bug%" AND i.code NOT LIKE "%boundsBug%" THEN 1 ELSE 0 END) AS "iGUARD Races Missed",
            SUM(CASE WHEN m.errors > 0 THEN 1 ELSE 0 END) AS "Compute Sanitizer Races Found",
            SUM(CASE WHEN m.errors = 0 AND i.code LIKE "%Bug%" AND i.code NOT LIKE "%boundsBug%" THEN 1 ELSE 0 END) AS "Compute Sanitizer Races Missed",
            SUM(CASE WHEN i.errors > 0 THEN 1 ELSE 0 END) AS "Sequential Comparison Races Found",
            SUM(CASE WHEN i.errors = 0 AND i.code LIKE "%Bug%" AND i.code NOT LIKE "%boundsBug%" THEN 1 ELSE 0 END) AS "Sequential Comparison Races Missed"
        FROM indigo i
            INNER JOIN iguard g
                ON g.code = i.code
                AND g.graph = i.graph
                AND g.threads_per_block = i.threads_per_block
                AND g.number_of_blocks = i.number_of_blocks
            INNER JOIN memcheck m
                ON m.code = i.code
                AND m.graph = i.graph
                AND m.threads_per_block = i.threads_per_block
                AND m.number_of_blocks = i.number_of_blocks
            INNER JOIN hirace
                ON i.code = REPLACE(hirace.code, "_hirace", "")
                AND i.graph = hirace.graph
                AND i.threads_per_block = hirace.threads_per_block
                AND i.number_of_blocks = hirace.number_of_blocks
        WHERE
            m.tool = 'racecheck'
        GROUP BY
            i.graph,
            i.threads_per_block,
            i.number_of_blocks;
        '''
        print(query)
        #query = '''
        #SELECT
        #    hirace.graph,
        #    hirace.threads_per_block,
        #    hirace.number_of_blocks,
        #    COUNT(DISTINCT hirace.code) AS num_tests,
        #    SUM(CASE WHEN hirace.errors > 0 THEN 1 ELSE 0 END) AS hirace_error_found,
        #    SUM(CASE WHEN g.errors > 0 THEN 1 ELSE 0 END) AS iguard_error_found,
        #    SUM(CASE WHEN m.errors > 0 THEN 1 ELSE 0 END) AS memcheck_error_found,
        #    SUM(CASE WHEN i.errors > 0 THEN 1 ELSE 0 END) AS indigo_error_found
        #FROM indigo i
        #    INNER JOIN iguard g
        #        ON g.code = i.code
        #        AND g.graph = i.graph
        #        AND g.threads_per_block = i.threads_per_block
        #        AND g.number_of_blocks = i.number_of_blocks
        #    INNER JOIN memcheck m
        #        ON m.code = i.code
        #        AND m.graph = i.graph
        #        AND m.threads_per_block = i.threads_per_block
        #        AND m.number_of_blocks = i.number_of_blocks
        #    INNER JOIN hirace
        #        ON i.code = REPLACE(hirace.code, "_hirace", "")
        #        AND i.graph = hirace.graph
        #        AND i.threads_per_block = hirace.threads_per_block
        #        AND i.number_of_blocks = hirace.number_of_blocks
        #WHERE
        #    m.tool = 'memcheck'
        #GROUP BY
        #    hirace.graph,
        #    hirace.threads_per_block,
        #    hirace.number_of_blocks;
        #'''
        fields, records = util.exec_query(sqlcursor, query)
        
        print("Compare to Table 1:\n")
        
        util.generate_tex_table(fields, records)
    except Exception as e:
        print(e)
    finally:
        # cleanup
        util.db_disconnect(connection, sqlcursor)


if __name__ == '__main__':
    args = sys.argv
    if (len(args) != 2):
        sys.exit('USAGE: provide path to database of results.\n')
    
    DATABASE = os.path.abspath(args[1])
    
    main()

