#!/usr/bin/python3 -u

import os
import sys
import glob
import configparser
import subprocess as subp

from time import perf_counter_ns

import util

DBNAME = 'empty_db_name'
BASELINE_TABLE = "baseline"
MEMCHECK_TABLE = "memcheck"
IGUARD_TABLE = "iguard"
HIRACE_TABLE = "hirace"

tables = [BASELINE_TABLE, MEMCHECK_TABLE, IGUARD_TABLE, HIRACE_TABLE]

VERSION        = ''
CXX            = VERSION + 'nvcc'
MEMCHECK       = VERSION + 'compute-sanitizer'

ROOT_DIR       = ''
IGUARD_PATH    = 'iGUARD-SOSP21/nvbit_release/tools/detector/detector.so'
HIRACE_SOURCE  = 'src/hirace/'
RESULTS_DIR    = 'results'
TEMP_DIR       = 'temp_files'

TEST_FILE = ''


def parse_log_baseline(fname):
    return {'errors': 0}


def parse_log_memcheck_racecheck(fname):
    '''
    parse logfile for error counts
    TODO: scrape more error info?
    
    @param fname: path to log file
    
    @return TODO 
    '''
    memcheck_key = 'Race reported'
    num_keys_found = 0
    
    with open(fname) as logfile:
        for line in logfile:
            if memcheck_key in line:
                num_keys_found += 1
    
    return {'errors': num_keys_found}


def parse_log_iguard(fname):
    '''
    parse logfile for error counts
    TODO: scrape more error info?
    
    @param fname: path to log file
    
    @return TODO 
    '''
    iguard_key = 'Race:'
    num_keys_found = 0
    errs = 0
    
    with open(fname) as logfile:
        for line in logfile:
            if iguard_key in line:
                num_keys_found += 1
    
    errs = num_keys_found
    
    return {'errors': errs}


def parse_log_hirace(fname):
    '''
    parse logfile for error counts
    TODO: scrape more error info?
    
    @param fname: path to log file
    
    @return TODO 
    '''
    race_key = 'Race'
    num_keys_found = 0
    errs = 0
    
    with open(fname) as logfile:
        for line in logfile:
            if race_key in line:
                num_keys_found += 1
    
    errs = num_keys_found
    
    return {'errors': errs}


def extract_data(parser, logfile, sqlcursor, table_name, **kwargs):
    '''
    read logs and write relevant data to sql database
    
    @param parser: function which takes a filepath to a log file and returns
                   a dictionary of values to be inserted in a sqlite table
    @param logfile: logfile from kernel execution
    @param sql_cursor: cursor for executing sql statements,
                       to insert data into a table
    @param table_name: sql table to insert this files data
    
    '''
    vals = parser(logfile)
    record = {**kwargs, **vals}
    
    util.table_insert(sqlcursor, table_name, record)


def run_baseline(code_file, graph_file,
                 threads_per_block, number_of_blocks,
                 sqlcursor, table_name):
    logfile = os.path.abspath(os.path.join(TEMP_DIR, "base.log"))
    
    cmd = '%s > %s' % (code_file, logfile)
    runtimes = []
    owd = os.getcwd()
    try:
        os.chdir(os.path.dirname(code_file))
        for _ in range(10):
            try:
                start = perf_counter_ns()
                subp.run(cmd, shell=True, timeout=300)
                end = perf_counter_ns()
                time = end - start
                runtimes.append(time)
            except subp.TimeoutExpired as err:
                time = -1
            print("Time: {} seconds\n".format(time/1e9))
    finally:
        os.chdir(owd)
    
    if not runtimes:
        runtime = 0
    else:
        runtime = sum(runtimes)/len(runtimes)
    # extract from cuda's memcheck logs
    extract_data(parse_log_baseline, logfile, sqlcursor, table_name,
                 code=os.path.basename(code_file), graph='',
                 tool='', threads_per_block=threads_per_block,
                 number_of_blocks=number_of_blocks, time_ns=runtime)
    
    if os.path.isfile(logfile):
        subp.run('rm ' + logfile, shell=True)


def run_memcheck(code_file, graph_file,
                 threads_per_block, number_of_blocks,
                 sqlcursor, table_name):
    '''
    execute cuda-memcheck on a given input (specific kernel, input graph, and grid dimensions)
    generates a log file, then parses into a sql table for bulk analysis.
    
    @param code_file: filepath for code to execute
    @param graph_file: filepath for graph input
    @param threads_per_block: cuda dim parameter
    @param number_of_blocks: cuda dim parameter
    @param sql_cursor: cursor for executing sql statements, 
                       to insert data into a table
    @param table_name: sql table to insert this files data
    '''
    logfile = os.path.abspath(os.path.join(TEMP_DIR, "mb.log"))
    
    # change from cuda-memcheck to compute-sanitizer - getting errors though
    cmd = '%s --log-file %s --tool %s /bin/bash %s > /dev/null' % (MEMCHECK, logfile, 'racecheck', code_file)
    runtimes = []
    owd = os.getcwd()
    try:
        os.chdir(os.path.dirname(code_file))
        for _ in range(10):
            try:
                start = perf_counter_ns()
                subp.run(cmd, shell=True, timeout=300)
                end = perf_counter_ns()
                time = end - start
                runtimes.append(time)
            except subp.TimeoutExpired as err:
                time = -1
            print("Time: {} seconds\n".format(time/1e9))
    finally:
        os.chdir(owd)
    
    if not runtimes:
        runtime = 0
    else:
        runtime = sum(runtimes)/len(runtimes)
    # extract from cuda's memcheck logs
    parser = parse_log_memcheck_racecheck
    extract_data(parser, logfile, sqlcursor, table_name,
                 code=os.path.basename(code_file), graph='',
                 tool='racecheck', threads_per_block=threads_per_block,
                 number_of_blocks=number_of_blocks, time_ns=runtime)
    
    if os.path.isfile(logfile):
        subp.run('rm ' + logfile, shell=True)


def run_iguard(code_file, graph_file,
                 threads_per_block, number_of_blocks,
                 sqlcursor, table_name):
    '''
    execute iGUARD on a given input (specific kernel, input graph, and grid dimensions)
    generates a log file, then parses into a sql table for bulk analysis.
    
    @param code_file: filepath for code to execute
    @param graph_file: filepath for graph input
    @param threads_per_block: cuda dim parameter
    @param number_of_blocks: cuda dim parameter
    @param sql_cursor: cursor for executing sql statements, 
                       to insert data into a table
    @param table_name: sql table to insert this files data
    '''
    logfile = os.path.abspath(os.path.join(TEMP_DIR, "iguard.log"))
    
    iguard_args = 'LD_PRELOAD=%s' % IGUARD_PATH
    cmd = '%s /bin/bash %s' % (iguard_args, code_file)
    cmd += ' > %s' % logfile
    runtimes = []
    owd = os.getcwd()
    try:
        os.chdir(os.path.dirname(code_file))
        for _ in range(10):
            try:
                start = perf_counter_ns()
                subp.run(cmd, shell=True, timeout=300)
                end = perf_counter_ns()
                time = end - start
                runtimes.append(time)
            except subp.TimeoutExpired as err:
                time = -1
            print("Time: {} seconds\n".format(time/1e9))
    finally:
        os.chdir(owd)
    
    if not runtimes:
        runtime = 0
    else:
        runtime = sum(runtimes)/len(runtimes)
    # extract from cuda's memcheck logs
    extract_data(parse_log_iguard, logfile, sqlcursor, table_name,
                 code=os.path.basename(code_file), graph='',
                 tool='', threads_per_block=threads_per_block,
                 number_of_blocks=number_of_blocks, time_ns=runtime)
    
    if os.path.isfile(logfile):
        subp.run('rm ' + logfile, shell=True)
    

def run_hirace(code_file, graph_file,
                 threads_per_block, number_of_blocks,
                 sqlcursor, table_name):
    '''
    execute our hirace race tool on a given input (specific kernel, input graph, and grid dimensions)
    generates a log file, then parses into a sql table for bulk analysis.
    
    @param code_file: filepath for code to execute
    @param graph_file: filepath for graph input
    @param threads_per_block: cuda dim parameter
    @param number_of_blocks: cuda dim parameter
    @param sql_cursor: cursor for executing sql statements, 
                       to insert data into a table
    @param table_name: sql table to insert this files data
    '''
    hirace_test = code_file + "_hirace"
    logfile = os.path.abspath(os.path.join(TEMP_DIR, 'hirace.log'))
    cmd = '%s > %s' % (hirace_test, logfile)
    runtimes = []
    owd = os.getcwd()
    try:
        os.chdir(os.path.dirname(code_file))
        for _ in range(10):
            try:
                start = perf_counter_ns()
                subp.run(cmd, shell=True, timeout=300)
                end = perf_counter_ns()
                time = end - start
                runtimes.append(time)
            except subp.TimeoutExpired as err:
                time = -1
            print("Time: {} seconds\n".format(time/1e9))
    finally:
        os.chdir(owd)
    
    if not runtimes:
        runtime = 0
    else:
        runtime = sum(runtimes)/len(runtimes)
    # parse log
    if os.path.isfile(logfile):
        extract_data(parse_log_hirace, logfile, sqlcursor, table_name,
                 code=os.path.basename(code_file), graph='',
                 tool='', threads_per_block=threads_per_block,
                 number_of_blocks=number_of_blocks, time_ns=runtime)
    
    # cleanup intermediate files
    if os.path.isfile(logfile):
        subp.run('rm ' + logfile, shell=True)


def main(*args, **kwargs):
    '''
    Utility script for aggregating data from multiple CUDA Memcheck analyses
    TODO: 
        extract build to separate function... maybe make build/run more generic.
        add argparse, allow recursive flag for dir.
    
    @param code_path: path to source files. This directory will be recursively expanded.
    @param input_path: path to input files. This directory will be recursively expanded.
    @param threads_per_block: integer cuda param
    @param number_of_blocks: integer cuda param
    '''
        
    if not os.path.exists(TEMP_DIR):
        subp.run('mkdir -p ' + TEMP_DIR, shell=True)
    if not os.path.exists(RESULTS_DIR):
        subp.run('mkdir -p ' + RESULTS_DIR, shell=True)
            
    
    # prep sql dbase
    table_fields = {
            'code': 'TEXT',
            'graph': 'TEXT',
            'tool': 'TEXT',
            'threads_per_block': 'INTEGER',
            'number_of_blocks': 'INTEGER',
            'errors': 'INTEGER',
            'time_ns': 'INTEGER'
            }
    connection, sqlcursor = util.db_setup(os.path.join(RESULTS_DIR,DBNAME))
    
    for tbl in tables:
        util.db_create_table(sqlcursor, tbl, table_fields)
    
    try:
        print('=============================================\n')
        print('Running Baseline...\n')
        run_memcheck(TEST_FILE, '', 0, 0, sqlcursor, BASELINE_TABLE)
        
        print('Running Compute Sanitizer...\n')
        run_memcheck(TEST_FILE, '', 0, 0, sqlcursor, MEMCHECK_TABLE)
        
        print('Running IGUARD...\n')
        run_iguard(TEST_FILE, '', 0, 0, sqlcursor, IGUARD_TABLE)
        
        print('Running HiRace...\n')
        run_hirace(TEST_FILE, '', 0, 0, sqlcursor, HIRACE_TABLE)
        print('-----------------\n')
        
        print('=============================================\n')
        connection.commit()
    except Exception as e:
        print(type(e), e)
    finally:        
        util.db_disconnect(connection, sqlcursor)
        if os.path.exists(TEMP_DIR):
            subp.run('rm -rf ' + TEMP_DIR, shell=True)


if __name__ == "__main__":
    args = sys.argv
    if (len(args) != 3):
        sys.exit('USAGE: <path to test executable> <result database name>.\n')
    
    TEST_FILE = os.path.abspath(args[1])
    print(TEST_FILE)
    DBNAME = args[2] 
    
    IGUARD_PATH = os.path.abspath(IGUARD_PATH)
    
    main()
