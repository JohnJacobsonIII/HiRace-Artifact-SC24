#!/usr/bin/python3 -u

import os
import sys
import glob
import configparser
import subprocess as subp

from time import perf_counter_ns

import util


# TODO: remove when indigo parsing generalized
DBNAME = 'hirace_correctness_results.sqlite3'
INDIGO_TABLE = "indigo"
MEMCHECK_TABLE = "memcheck"
IGUARD_TABLE = "iguard"
HIRACE_TABLE = "hirace"

tables = [INDIGO_TABLE, MEMCHECK_TABLE, IGUARD_TABLE, HIRACE_TABLE]

VERSION        = ''
CXX            = VERSION + 'nvcc'
MEMCHECK       = VERSION + 'compute-sanitizer'

ROOT_DIR       = ''
INPUT_PATH     = 'indigo/input'
IGUARD_PATH    = 'iGUARD-SOSP21/nvbit_release/tools/detector/detector.so'
HIRACE_SOURCE  = 'src/hirace/'
INDIGO_INCLUDE = 'indigo/indigo_include'
INDIGO_TESTS   = 'indigo/indigo_sources/'
HIRACE_TESTS   = 'indigo/indigo_hirace_sources/'
RESULTS_DIR    = 'results'
TEMP_DIR       = 'temp_files'


def parse_log_indigo(fname):
    '''
    parse logfile for error counts
    
    @param fname: path to log file
    
    @return TODO 
    '''
    return {'errors': 0}


def parse_log_memcheck(fname):
    '''
    parse logfile for error counts
    TODO: scrape more error info?
    
    @param fname: path to log file
    
    @return TODO 
    '''
    memcheck_key = 'ERROR SUMMARY:'
    num_keys_found = 0
    errs = 0
    
    with open(fname) as logfile:
        for line in logfile:
            if memcheck_key in line:
                num_keys_found += 1
                nlist = [int(n) for n in line.split() if n.isdigit()]
                assert len(nlist) == 1, "multiple error values reported"
                errs = nlist[0]
    
    assert num_keys_found < 2, "problem with error reports in logfile"
    
    return {'errors': errs}


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


def run_indigo(code_file, graph_file,
                 threads_per_block, number_of_blocks,
                 sqlcursor, table_name):
    '''
    execute indigo benchmark on a given input (specific kernel, input graph, and grid dimensions)
    generates a log file, then parses into a sql table for bulk analysis.
    
    @param code_file: filepath for code to execute
    @param graph_file: filepath for graph input
    @param threads_per_block: cuda dim parameter
    @param number_of_blocks: cuda dim parameter
    @param sql_cursor: cursor for executing sql statements, 
                       to insert data into a table
    @param table_name: sql table to insert this files data
    '''
    logfile = os.path.join(TEMP_DIR, "indigo.log")
    cmd = './%s %s %s %s' % (os.path.join(TEMP_DIR,'microbenchmark'), graph_file, threads_per_block, number_of_blocks)
    cmd += ' > %s' % logfile
    start = perf_counter_ns()
    subp.run(cmd, shell=True)
    end = perf_counter_ns()
    time = end - start
    
    # extract from indigo stdout
    extract_data(parse_log_indigo, logfile, sqlcursor, INDIGO_TABLE,
                 code=os.path.basename(code_file), graph=os.path.basename(graph_file),
                 tool='', threads_per_block=threads_per_block,
                 number_of_blocks=number_of_blocks, time_ns=time)
        
    if os.path.isfile(logfile):
        subp.run('rm ' + logfile, shell=True)
    # TODO: validate all tests ran.


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
    logfile = os.path.join(TEMP_DIR, "mb.log")
    tools = {'memcheck','initcheck','synccheck', 'racecheck'} # TODO: add racecheck
    
    for tool in tools:
        # change from cuda-memcheck to compute-sanitizer - getting errors though
        cmd = '%s --log-file %s --tool %s ./%s' % (MEMCHECK, logfile, tool, os.path.join(TEMP_DIR,'microbenchmark'))
        cmd += ' %s %s %s > /dev/null' % (graph_file, threads_per_block, number_of_blocks)
        start = perf_counter_ns()
        subp.run(cmd, shell=True)
        end = perf_counter_ns()
        time = end - start
        
        # extract from cuda's memcheck logs
        parser = parse_log_memcheck_racecheck if tool == 'racecheck' else parse_log_memcheck
        extract_data(parser, logfile, sqlcursor, table_name,
                     code=os.path.basename(code_file), graph=os.path.basename(graph_file),
                     tool=tool, threads_per_block=threads_per_block,
                     number_of_blocks=number_of_blocks, time_ns=time)
        
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
    logfile = os.path.join(TEMP_DIR, "iguard.log")
    
    iguard_args = 'LD_PRELOAD=%s' % IGUARD_PATH
    cmd = '%s ./%s' % (iguard_args, os.path.join(TEMP_DIR,'microbenchmark'))
    # !!NOTE!! iguard fails catastrophically if 2^22 blocks are requested
    cmd += ' %s %s %s' % (graph_file, threads_per_block, number_of_blocks)
    # pipe to file for log parsing.
    # TODO: extract this to generic process
    # cmd += ' | tee %s' % indigo_logfile
    cmd += ' > %s' % logfile
    start = perf_counter_ns()
    subp.run(cmd, shell=True)
    end = perf_counter_ns()
    time = end - start
    
    # extract from cuda's memcheck logs
    extract_data(parse_log_iguard, logfile, sqlcursor, table_name,
                 code=os.path.basename(code_file), graph=os.path.basename(graph_file),
                 tool='', threads_per_block=threads_per_block,
                 number_of_blocks=number_of_blocks, time_ns=time)
    
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
    binfile = os.path.join(TEMP_DIR, 'hirace_microbenchmark')
    logfile = os.path.join(TEMP_DIR, 'hirace.log')
    run_args = ' %s %s %s > ' % (graph_file, threads_per_block, number_of_blocks)
    cmd = './%s' % binfile
    cmd += run_args + logfile
    if os.path.isfile(binfile):
        start = perf_counter_ns()
        subp.run(cmd, shell=True)
        end = perf_counter_ns()
        time = end - start
    # parse log
    if os.path.isfile(logfile):
        extract_data(parse_log_hirace, logfile, sqlcursor, table_name,
                 code=os.path.basename(code_file), graph=os.path.basename(graph_file),
                 tool='', threads_per_block=threads_per_block,
                 number_of_blocks=number_of_blocks, time_ns=time)
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
    code_files = map(os.path.abspath,glob.glob(INDIGO_TESTS + '/**/*.cu', recursive=True))
    hirace_code_files = map(os.path.abspath,glob.glob(HIRACE_TESTS + '/**/*.cu', recursive=True))
    graph_files = map(os.path.abspath,glob.glob(INPUT_PATH + '/**/*.egr', recursive=True))
    
    code_files = list(code_files)
    hirace_code_files = list(hirace_code_files)
    graph_files = list(graph_files)
    
    # manually created hirace files, pairing them up until automatic instrumentation is done
    code_files = [(a,b) for a in code_files for b in hirace_code_files if a == (b.replace("_hirace",""))]
    threads_per_block = 256
    number_of_blocks = 1024
    
    prod = len(code_files) * len(graph_files)
    complete = 0
        
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
    
    completed_codes = util.check_completed_codes(sqlcursor)
    
    try:
        # walk files and build + execute test kernels on all inputs
        for (ifile, hfile) in code_files:
            print('=============================================\n')
            if os.path.basename(ifile) not in completed_codes:
                print('\ncompile : %s\n' % os.path.basename(ifile))
                subp.run("%s -I%s %s -o %s" % (CXX, INDIGO_INCLUDE, ifile, os.path.join(TEMP_DIR,'microbenchmark')), shell=True)
                subp.run("nvcc %s -DRACECHECK -I%s -o %s" % (hfile, HIRACE_SOURCE, os.path.join(TEMP_DIR,'hirace_microbenchmark')), shell=True)
                print(os.path.exists(os.path.join(TEMP_DIR,'microbenchmark')))
                for graph_file in graph_files:
                    print('input: %s\nmicrobenchmark: %s\n' % (os.path.basename(graph_file),
                                                                 os.path.basename(ifile)))
                    print('threads_per_block: %s\nnumber_of_blocks: %s\n' % (threads_per_block, number_of_blocks))
        
                    print('Running Indigo base microbenchmark...\n')
                    run_indigo(ifile, graph_file, threads_per_block,
                                 number_of_blocks, sqlcursor, INDIGO_TABLE)
                    
                    print('Running Compute Sanitizer...\n')
                    run_memcheck(ifile, graph_file, threads_per_block,
                                 number_of_blocks, sqlcursor, MEMCHECK_TABLE)
                    
                    print('Running IGUARD...\n')
                    run_iguard(ifile, graph_file, threads_per_block,
                               number_of_blocks, sqlcursor, IGUARD_TABLE)
                    
                    print('Running HiRace...\n')
                    run_hirace(hfile, graph_file, threads_per_block,
                            number_of_blocks, sqlcursor, HIRACE_TABLE)
                    print('-----------------\n')
            else:
                print('Skipping %s, entry found in %s.\n' % (os.path.basename(ifile), DBNAME))
            
            complete += 1
            print('%d of %d complete ({:.2f}%%)\n'.format(complete/prod * 100) % (complete, prod))
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
    if (len(args) != 5):
        sys.exit('USAGE: <path to project root> <path to test code directory> <path to test input directory> <result database name>.\n')
    
    os.chdir(os.path.abspath(args[1]))
    
    #ROOT_DIR = os.path.abspath(args[1])
    INDIGO_TESTS = os.path.abspath(args[2])
    HIRACE_TESTS = INDIGO_TESTS.replace('indigo_sources', 'indigo_hirace_sources')
    INPUT_PATH = os.path.abspath(args[3])
    
    DBNAME = args[4] 

    #INPUT_PATH     = os.path.join(ROOT_DIR, INPUT_PATH)
    IGUARD_PATH    = os.path.join(ROOT_DIR, IGUARD_PATH)
    HIRACE_SOURCE  = os.path.join(ROOT_DIR, HIRACE_SOURCE)
    INDIGO_INCLUDE = os.path.join(ROOT_DIR, INDIGO_INCLUDE)
    #INDIGO_TESTS   = os.path.join(ROOT_DIR, INDIGO_TESTS)
    #HIRACE_TESTS   = os.path.join(ROOT_DIR, HIRACE_TESTS)
    RESULTS_DIR    = os.path.join(ROOT_DIR, RESULTS_DIR)
    TEMP_DIR       = os.path.join(ROOT_DIR, TEMP_DIR)
    
    print(HIRACE_TESTS)
    print(RESULTS_DIR)
    
    main()
