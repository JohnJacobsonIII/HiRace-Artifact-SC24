import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys

import util

DATABASE = ''

def main():
    connection, sqlcursor = util.db_setup(DATABASE)
    try:
        # generate box plot diagram
        iguard_speedup_query = '''
            SELECT
                hirace.code,
                hirace.graph,
                g.time_ns / CAST(hirace.time_ns AS REAL) AS speedup_iguard
            FROM
                hirace
                INNER JOIN iguard g
                    ON REPLACE(hirace.code, "_hirace", "") = g.code
                        AND hirace.graph = g.graph
                        AND hirace.threads_per_block = g.threads_per_block
                        AND hirace.number_of_blocks = g.number_of_blocks;
        '''
        
        print("running speedup query...")
        fields, records = util.exec_query(sqlcursor, iguard_speedup_query)
        df = pd.DataFrame(records, columns = fields)
        
        sns.set_theme(style="ticks")
        
        # Initialize the figure with a logarithmic x axis
        f, ax = plt.subplots(figsize=(7, 6))
        # ax.set_xscale("log")
        
        print(df)
        # Plot the orbital period with horizontal boxes
        sns.boxplot(x="speedup_iguard", y="graph", data=df,
                            whis=[0, 100], width=.6, palette="vlag")
        
        # Add in points to show each observation
        sns.stripplot(x="speedup_iguard", y="graph", data=df,
                              size=4, color=".3", linewidth=0)
        
        # Tweak the visual presentation
        ax.xaxis.grid(True)
        ax.set(xlabel="Speedup vs. iGuard")
        ax.set(ylabel="")
        sns.despine(trim=True, left=True)
        
        plt.savefig('compare_to_figure5.png', bbox_inches='tight')
    
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


