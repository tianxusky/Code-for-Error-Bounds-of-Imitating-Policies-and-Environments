# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from termcolor import colored
import datetime
import sys
import os
from collections import Counter, defaultdict
import json_tricks


def a():
    pass


_srcfile = os.path.normcase(a.__code__.co_filename)


class BaseSink(object):
    @staticmethod
    def _time():
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')

    def info(self, fmt, *args, **kwargs):
        raise NotImplementedError

    def warning(self, fmt, *args, **kwargs):
        self.info(fmt, *args, **kwargs)

    def verbose(self, fmt, *args, **kwargs):
        pass


class StdoutSink(BaseSink):
    def __init__(self):
        self.freq_count = Counter()

    def info(self, fmt, *args, freq=1, caller=None):
        if args:
            fmt = fmt % args
        self.freq_count[caller] += 1
        if self.freq_count[caller] % freq == 0:
            print("%s - %s - %s" % (colored(self._time(), 'green'),
                                    colored(caller, 'cyan'), fmt), flush=True)

    def warning(self, fmt, *args, **kwargs):
        if args:
            fmt = fmt % args
        self.info(colored(fmt, 'yellow'), **kwargs)


class FileSink(BaseSink):
    def __init__(self, fn):
        self._fn = fn
        self.log_file = open(fn, 'w')
        self.callers = {}

    def info(self, fmt, *args, **kwargs):
        self._kv(level='info', fmt=fmt, args=args, **kwargs)

    def warning(self, fmt, *args, **kwargs):
        self._kv(level='warning', fmt=fmt, args=args, **kwargs)

    def _kv(self, **kwargs):
        kwargs.update(time=datetime.datetime.now())
        if self._fn.endswith('.json'):
            self.log_file.write(json_tricks.dumps(kwargs, primitives=True) + '\n')
        elif self._fn.endswith('.txt'):
            content = ''
            if kwargs.get('time'): content += '{} - '.format(kwargs.pop('time'))
            if kwargs.get('caller'): content += '{} - '.format(kwargs.pop('caller'))
            content += kwargs['fmt']
            if len(kwargs['args']) > 0: content = content % kwargs['args']
            for key, val in kwargs.items():
                if key in {'level', 'fmt', 'args'}: continue
                content += '{}: {} '.format(key, val)
            self.log_file.write(content+'\n')
        self.log_file.flush()

    def verbose(self, fmt, *args, **kwargs):
        self._kv(level='verbose', fmt=fmt, args=args, **kwargs)


class LibLogger(object):
    logfile = ""

    def __init__(self, name='logger', is_root=True):
        self.name = name
        self.is_root = is_root
        self.tab_keys = None
        self.sinks = []
        self.key_prior = defaultdict(np.random.randn)
        self.csv_writer = None

    def add_sink(self, sink):
        self.sinks.append(sink)

    def add_csvwriter(self, writer):
        self.csv_writer = writer

    def write_kvs(self, kvs):
        self.csv_writer.writekvs(kvs)

    def info(self, fmt, *args, **kwargs):
        caller = self.find_caller()
        for sink in self.sinks:
            sink.info(fmt, *args, caller=caller, **kwargs)

    def warning(self, fmt, *args, **kwargs):
        caller = self.find_caller()
        for sink in self.sinks:
            sink.warning(fmt, *args, caller=caller, **kwargs)

    def verbose(self, fmt, *args, **kwargs):
        caller = self.find_caller()
        for sink in self.sinks:
            sink.verbose(fmt, *args, caller=caller, **kwargs)

    def find_caller(self):
        """
        Copy from `python.logging` module

        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.
        """
        f = sys._getframe(1)
        if f is not None:
            f = f.f_back
        caller = ''
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename == _srcfile:
                f = f.f_back
                continue
            # if stack_info:
            #     sio = io.StringIO()
            #     sio.write('Stack (most recent call last):\n')
            #     traceback.print_stack(f, file=sio)
            #     sio.close()
            # rv = (co.co_filename, f.f_lineno, co.co_name, sinfo)
            rel_path = os.path.relpath(co.co_filename, '')
            caller = f'{rel_path}:{f.f_lineno}'
            break
        return caller


class CSVWriter(object):
    def __init__(self, filename):
        self.file = open(filename, 'w+t')
        self.keys = []
        self.sep = ','

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(',')
                self.file.write(k)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write('\n')
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write('\n')
        self.file.flush()

    def close(self):
        self.file.close()


def get_logger(name):
    return LibLogger(name)


def _log_numerical(number):
    if isinstance(number, (int, np.int32, np.int64)):
        return '%s = %d '
    elif isinstance(number, (float, np.float32, np.float64)):
        return '%s = %.4f '
    else:
        raise TypeError('{} = {} is not recognized'.format(number, type(number)))


def log_kvs(kvs: dict, prefix: str):
    kvs_ = dict()
    format_ = '[%s] ' % prefix
    args_ = []
    for key, val in kvs.items():
        assert isinstance(key, str)
        if isinstance(val, dict):
            if len(val) <= 3:
                format_ += '%s={ '
                args_.append(key)
            for sub_key, sub_val in val.items():
                if len(val) <= 2:
                    format_ += _log_numerical(sub_val)
                    args_.extend([sub_key, sub_val])
                kvs_[prefix+'/'+key+'_'+str(sub_key).lower()] = sub_val
            if len(val) <= 2:
                format_ += '} '
        else:
            format_ += _log_numerical(val)
            args_.extend([key, val])
            kvs_[prefix+'/'+key] = val
    logger.write_kvs(kvs_)
    logger.info(format_, *args_)


logger = get_logger('Logger')
logger.add_sink(StdoutSink())
