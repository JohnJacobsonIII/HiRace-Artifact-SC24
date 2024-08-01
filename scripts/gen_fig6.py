import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
                b.code as Benchmark,
                b.time_ns / CAST(h.time_ns AS REAL) AS Hirace_Speedup,
                g.time_ns / CAST(hirace.time_ns AS REAL) AS speedup_iguard
            FROM
                hirace
                INNER JOIN iguard g
                    ON REPLACE(hirace.code, "_hirace", "") = g.code
                        AND hirace.graph = g.graph
                        AND hirace.threads_per_block = g.threads_per_block
                        AND hirace.number_of_blocks = g.number_of_blocks
            WHERE
                hirace.time_ns > 0
                AND g.time_ns > 0;
        '''
        
        print("running speedup query...")
        fields, records = util.exec_query(sqlcursor, iguard_speedup_query)
        rodinia = pd.DataFrame(records, columns = fields)

    except Exception as e:
        print(e)
    finally:
        # cleanup
        util.db_disconnect(connection, sqlcursor)

    sns.set_style("ticks")
    
    #rodinia = pd.DataFrame(columns=["Benchmark", "Tool", "Slowdown"],
    #                      data=[["Backprop", "iGUARD", 313.319],["Backprop", "HiRace", 35.968],["Backprop", "Compute Sanitizer", 14.873],
    #                            ["Gaussian", "iGUARD", 146.286],["Gaussian", "HiRace", 5.584],["Gaussian", "Compute Sanitizer", 3.862],
    #                            ["SRAD", "iGUARD", 0],["SRAD", "HiRace", 13.965],["SRAD", "Compute Sanitizer", 76.696],
    #                            ["Black\nScholes", "iGUARD", 14.340],["Black\nScholes", "HiRace", 3.146],["Black\nScholes", "Compute Sanitizer", 2.118],
    #                            ["Convolution\nFFT2D", "iGUARD", 5.308],["Convolution\nFFT2D", "HiRace", 1.009],["Convolution\nFFT2D", "Compute Sanitizer", 4.730],
    #                            ["Fast\nWalsh\nTransform", "iGUARD", 2.239],["Fast\nWalsh\nTransform", "HiRace", 1.034],["Fast\nWalsh\nTransform", "Compute Sanitizer", 4.485]])
    
    # Draw a nested barplot by species and sex
    g = sns.catplot(
                data=rodinia, kind="bar",
                    x="Benchmark", y="Slowdown", hue="Tool",
                        #errorbar="sd", 
                        palette="bright", 
                        alpha=.6, height=6,
                        legend_out=False
                        )
    #g.despine(left=True)
    g.set_axis_labels("Benchmark", "Slowdown vs Base Program", weight='bold')
    #sns.move_legend(g, "upper right")
    #g.set_title("Rodinia Benchmarks")
    #g.set(title="Rodinia Benchmarks")
    #g.set_xlabels(g.get_xlabels(), fontdict={'weight': 'bold'})
    #g.set_ylabels(g.get_ylabels(), fontdict={'weight': 'bold'})
    
    #plt.xlabel('X-axis', weight='bold')
    #plt.ylabel('Y-axis', weight='bold')
    
    for tick in plt.gca().get_xticklabels():
        tick.set_fontweight('bold')
    for tick in plt.gca().get_yticklabels():
        tick.set_fontweight('bold')
    
    plt.yscale('log')
    
    plt.savefig('results/compare_to_figure6.png')#, bbox_inches='tight')


if __name__ == '__main__':
    args = sys.argv
    if (len(args) != 2):
        sys.exit('USAGE: provide path to database of results.\n')
    
    DATABASE = os.path.abspath(args[1])
    
    main()
