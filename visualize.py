import matplotlib
params = {
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'lines.linewidth': 2.5,
}
matplotlib.rcParams.update(params)
import matplotlib.pyplot as plt
import os
import collections
import numpy as np
import yaml
from utils import plot_utils as pu


Result = collections.namedtuple('Result', 'progress dir root_dir')

COLORS = ['red', 'darkgreen', 'black', 'orange', 'blue', 'crimson', 'magenta', 'yellow', 'cyan','purple', 'pink',
        'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender', 'turquoise',
        'green', 'tan', 'salmon', 'gold',  'darkred', 'darkblue']

MARKERS = ['*', 'o', 'x', 'D', 'p', 's']


def read_evaluate_yaml_file(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError('%s not found' % filename)
    kvs = yaml.load(open(filename, 'r'))
    print('load %s successful' % filename)
    return kvs


def read_progress(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError('%s not found' % filename)
    kvs = dict()
    keys = []
    file = open(filename, 'r')
    for line in file.readlines():
        line = line[:-1]
        if len(kvs) == 0:
            for key in line.split(','):
                keys.append(key)
            for key in keys:
                kvs[key] = []
        else:
            for i, val in enumerate(line.split(',')):
                if len(val) > 0:
                    kvs[keys[i]].append(float(val))
    for key, val in kvs.items():
        kvs[key] = np.array(val)
    print('load %s successful' % filename)
    return kvs


def xy_fn(r):
    progress = r.progress
    x = [0.9, 0.99, 0.999]
    y = [progress[k][0] for k in x]
    return x, y


def split_fn(r):
    splits = r.dir.split('-')
    env_name = splits[1] + '-' + splits[2]
    return env_name


def group_fn(r):
    splits = r.root_dir.split('/')
    alg_name = splits[-1]
    if alg_name == 'bc':
        return 'BC'
    elif alg_name == 'dagger':
        return 'DAgger'
    elif alg_name == 'gail_nn':
        return 'GAIL'
    elif alg_name == 'gail_l2':
        return 'FEM'
    elif alg_name == 'gail_simplex':
        return 'GTAL'
    elif alg_name == 'expert':
        return 'Expert'
    elif alg_name == 'gail_w':
        return 'WGAIL'
    else:
        raise ValueError('%s is not supported.' % alg_name)


def gail_regular_l2_xy_fn(r):
    progress = r.progress
    x_name, y_name = 'Evaluate/iter', 'Evaluate/discounted_episode_returns'
    assert x_name in progress.keys(), '{} not in {}'.format(x_name, list(progress.keys()))
    assert y_name in progress.keys(), '{} not in {}'.format(y_name, list(progress.keys()))
    x = progress[x_name]
    y = progress[y_name]
    index = 0
    while y[index] == 0.:
        index += 1
    y[:index] = y[index]
    return x, y


def mbrl_xy_stochastic_fn(r):
    progress = r.progress
    y_name = 'Evaluate/evaluation_error_stochastic_abs'
    y = progress[y_name]
    subsampling_rate = len(y) // 150
    indices = np.arange(0, len(y), len(y)/150).astype(np.int32)
    y = y[indices]
    x = np.arange(len(y))
    return x, y


def mbrl_xy_deterministic_fn(r):
    progress = r.progress
    y_name = 'Evaluate/evaluation_error_deterministic_abs'
    y = progress[y_name]
    subsampling_rate = len(y) // 150
    indices = np.arange(0, len(y), len(y)/150).astype(np.int32)
    y = y[indices]
    x = np.arange(len(y))
    return x, y


def mbrl_group_fn(r):
    splits = r.dir.split('-')
    alg_name = splits[0]
    if alg_name == 'mbrl_bc' or alg_name == 'mbrl2_bc':
        return 'MB-BC'
    elif alg_name == 'mbrl_gail' or alg_name == 'mbrl2_gail':
        return 'MB-GAIL'
    else:
        raise ValueError('%s is not supported' % alg_name)


def gail_regular_l2_group_fn(r):
    splits = r.dir.split('-')[0]
    if splits == 'gail':
        return r'GAIL($\lambda=0.0$)'
    splits = splits.split('_')
    alg_name = splits[0]
    assert alg_name == 'gail', 'splits = %s' % splits
    coef = splits[2]
    return r'GAIL($\lambda={}$)'.format(coef)


def plot_results(
        allresults, *,
        xy_fn=pu.default_xy_fn,
        split_fn=pu.default_split_fn,
        group_fn=pu.default_group_fn,
        average_group=False,
        shaded_std=True,
        shaded_err=True,
        figsize=None,
        legend_outside=False,
        resample=0,
        smooth_step=1.0,
        xlabel=None,
        ylabel=None,
):
    '''
    Plot multiple Results objects

    xy_fn: function Result -> x,y           - function that converts results objects into tuple of x and y values.
                                              By default, x is cumsum of episode lengths, and y is episode rewards

    split_fn: function Result -> hashable   - function that converts results objects into keys to split curves into sub-panels by.
                                              That is, the results r for which split_fn(r) is different will be put on different sub-panels.
                                              By default, the portion of r.dirname between last / and -<digits> is returned. The sub-panels are
                                              stacked vertically in the figure.

    group_fn: function Result -> hashable   - function that converts results objects into keys to group curves by.
                                              That is, the results r for which group_fn(r) is the same will be put into the same group.
                                              Curves in the same group have the same color (if average_group is False), or averaged over
                                              (if average_group is True). The default value is the same as default value for split_fn

    average_group: bool                     - if True, will average the curves in the same group and plot the mean. Enables resampling
                                              (if resample = 0, will use 512 steps)

    shaded_std: bool                        - if True (default), the shaded region corresponding to standard deviation of the group of curves will be
                                              shown (only applicable if average_group = True)

    shaded_err: bool                        - if True (default), the shaded region corresponding to error in mean estimate of the group of curves
                                              (that is, standard deviation divided by square root of number of curves) will be
                                              shown (only applicable if average_group = True)

    figsize: tuple or None                  - size of the resulting figure (including sub-panels). By default, width is 6 and height is 6 times number of
                                              sub-panels.


    legend_outside: bool                    - if True, will place the legend outside of the sub-panels.

    resample: int                           - if not zero, size of the uniform grid in x direction to resample onto. Resampling is performed via symmetric
                                              EMA smoothing (see the docstring for symmetric_ema).
                                              Default is zero (no resampling). Note that if average_group is True, resampling is necessary; in that case, default
                                              value is 512.

    smooth_step: float                      - when resampling (i.e. when resample > 0 or average_group is True), use this EMA decay parameter (in units of the new grid step).
                                              See docstrings for decay_steps in symmetric_ema or one_sided_ema functions.

    '''

    if split_fn is None: split_fn = lambda _: ''
    if group_fn is None: group_fn = lambda _: ''
    sk2r = collections.defaultdict(list)  # splitkey2results
    for result in allresults:
        splitkey = split_fn(result)
        sk2r[splitkey].append(result)
    assert len(sk2r) > 0
    assert isinstance(resample, int), "0: don't resample. <integer>: that many samples"
    nrows = 1
    ncols = len(sk2r)
    figsize = figsize or (6.5 * ncols, 4 * nrows)
    f, axarr = plt.subplots(nrows, ncols, sharex=False, squeeze=False, figsize=figsize, dpi=300)
    plt.subplots_adjust(wspace=0.2, hspace=0.35)

    groups = sorted(list(set(group_fn(result) for result in allresults)))

    default_samples = 512
    # if average_group:
    #     resample = resample or default_samples

    for (isplit, sk) in enumerate(sorted(sk2r.keys())):
        g2l = {}
        g2c = collections.defaultdict(int)
        sresults = sk2r[sk]
        gresults = collections.defaultdict(list)
        ax = axarr[0][isplit]
        for result in sresults:
            group = group_fn(result)
            g2c[group] += 1
            x, y = xy_fn(result)
            if x is None: x = np.arange(len(y))
            x, y = map(np.asarray, (x, y))
            if average_group:
                gresults[group].append((x, y))
            else:
                if resample:
                    x, y, counts = pu.symmetric_ema(x, y, x[0], x[-1], resample, decay_steps=smooth_step)
                l, = ax.plot(x, y, color=COLORS[groups.index(group) % len(COLORS)])
                g2l[group] = l
        if average_group:
            for group in sorted(groups):
                xys = gresults[group]
                if not any(xys):
                    continue
                color = COLORS[groups.index(group) % len(COLORS)]
                marker = MARKERS[groups.index(group) % len(MARKERS)]
                origxs = [xy[0] for xy in xys]
                minxlen = min(map(len, origxs))

                def allequal(qs):
                    return all((q == qs[0]).all() for q in qs[1:])

                if resample:
                    low = max(x[0] for x in origxs)
                    high = min(x[-1] for x in origxs)
                    usex = np.linspace(low, high, resample)
                    ys = []
                    for (x, y) in xys:
                        ys.append(pu.symmetric_ema(x, y, low, high, resample, decay_steps=smooth_step)[1])
                else:
                    assert allequal([x[:minxlen] for x in origxs]), \
                        'If you want to average unevenly sampled data, set resample=<number of samples you want>'
                    usex = origxs[0]
                    ys = [xy[1][:minxlen] for xy in xys]
                ymean = np.mean(ys, axis=0)
                ystd = np.std(ys, axis=0)
                ystderr = ystd / np.sqrt(len(ys))
                print(sk, group, ymean, ystd)

                xticks = list(range(1, len(usex) + 1))
                xticklables = [str(x_) for x_ in usex]
                l, = axarr[0][isplit].plot(xticks, ymean, color=color, marker=marker, markersize=10)
                g2l[group] = l
                if shaded_err:
                    ax.fill_between(xticks, ymean - ystderr, ymean + ystderr, color=color, alpha=.4)
                if shaded_std:
                    ax.fill_between(xticks, ymean - ystd, ymean + ystd, color=color, alpha=.2)

                axarr[0][isplit].set_xticks(xticks)
                axarr[0][isplit].grid(True)
                axarr[0][isplit].set_xticklabels(xticklables)

        # https://matplotlib.org/users/legend_guide.html
        plt.tight_layout()
        ax.set_title(sk)

    # g2l_sorted = collections.OrderedDict()
    # for key in ['Expert', 'GAIL', 'BC', 'FEM', 'DAgger', 'GTAL']:
    #     g2l_sorted[key] = g2l[key]
    # g2l = g2l_sorted
    if any(g2l.keys()):
        f.legend(
            g2l.values(),
            ['%s' % g for g in g2l] if average_group else g2l.keys(),
            loc='upper center' if legend_outside else None,
            fancybox=True,
            ncol=3,
            bbox_to_anchor=(0.5, 1.35) if legend_outside else None)

    # add xlabels, but only to the bottom row
    if xlabel is not None:
        for ax in axarr[-1]:
            plt.sca(ax)
            plt.xlabel(xlabel)
    # add ylabels, but only to left column
    if ylabel is not None:
        for ax in axarr[:, 0]:
            plt.sca(ax)
            plt.ylabel(ylabel)

    return f, axarr


def plot_gail_regular_l2_results(
        allresults, *,
        xy_fn=pu.default_xy_fn,
        split_fn=pu.default_split_fn,
        group_fn=pu.default_group_fn,
        average_group=False,
        shaded_std=True,
        shaded_err=True,
        figsize=None,
        legend_outside=False,
        resample=0,
        smooth_step=1.0,
        xlabel=None,
        ylabel=None,
):
    '''
    Plot multiple Results objects

    xy_fn: function Result -> x,y           - function that converts results objects into tuple of x and y values.
                                              By default, x is cumsum of episode lengths, and y is episode rewards

    split_fn: function Result -> hashable   - function that converts results objects into keys to split curves into sub-panels by.
                                              That is, the results r for which split_fn(r) is different will be put on different sub-panels.
                                              By default, the portion of r.dirname between last / and -<digits> is returned. The sub-panels are
                                              stacked vertically in the figure.

    group_fn: function Result -> hashable   - function that converts results objects into keys to group curves by.
                                              That is, the results r for which group_fn(r) is the same will be put into the same group.
                                              Curves in the same group have the same color (if average_group is False), or averaged over
                                              (if average_group is True). The default value is the same as default value for split_fn

    average_group: bool                     - if True, will average the curves in the same group and plot the mean. Enables resampling
                                              (if resample = 0, will use 512 steps)

    shaded_std: bool                        - if True (default), the shaded region corresponding to standard deviation of the group of curves will be
                                              shown (only applicable if average_group = True)

    shaded_err: bool                        - if True (default), the shaded region corresponding to error in mean estimate of the group of curves
                                              (that is, standard deviation divided by square root of number of curves) will be
                                              shown (only applicable if average_group = True)

    figsize: tuple or None                  - size of the resulting figure (including sub-panels). By default, width is 6 and height is 6 times number of
                                              sub-panels.


    legend_outside: bool                    - if True, will place the legend outside of the sub-panels.

    resample: int                           - if not zero, size of the uniform grid in x direction to resample onto. Resampling is performed via symmetric
                                              EMA smoothing (see the docstring for symmetric_ema).
                                              Default is zero (no resampling). Note that if average_group is True, resampling is necessary; in that case, default
                                              value is 512.

    smooth_step: float                      - when resampling (i.e. when resample > 0 or average_group is True), use this EMA decay parameter (in units of the new grid step).
                                              See docstrings for decay_steps in symmetric_ema or one_sided_ema functions.

    '''

    if split_fn is None: split_fn = lambda _: ''
    if group_fn is None: group_fn = lambda _: ''
    sk2r = collections.defaultdict(list)  # splitkey2results
    for result in allresults:
        splitkey = split_fn(result)
        sk2r[splitkey].append(result)
    assert len(sk2r) > 0
    assert isinstance(resample, int), "0: don't resample. <integer>: that many samples"
    nrows = 1
    ncols = len(sk2r)
    figsize = figsize or (6.5 * ncols, 4 * nrows)
    f, axarr = plt.subplots(nrows, ncols, sharex=False, squeeze=False, figsize=figsize, dpi=300)

    groups = list(set(group_fn(result) for result in allresults))
    groups.sort()

    default_samples = 512
    # if average_group:
    #     resample = resample or default_samples

    for (isplit, sk) in enumerate(sorted(sk2r.keys())):
        g2l = {}
        g2c = collections.defaultdict(int)
        sresults = sk2r[sk]
        gresults = collections.defaultdict(list)
        ax = axarr[0][isplit]
        for result in sresults:
            group = group_fn(result)
            g2c[group] += 1
            x, y = xy_fn(result)
            if x is None: x = np.arange(len(y))
            x, y = map(np.asarray, (x, y))
            if average_group:
                gresults[group].append((x, y))
            else:
                if resample:
                    x, y, counts = symmetric_ema(x, y, x[0], x[-1], resample, decay_steps=smooth_step)
                l, = ax.plot(x, y, color=COLORS[groups.index(group) % len(COLORS)])
                g2l[group] = l
        if average_group:
            for group in sorted(groups):
                xys = gresults[group]
                if not any(xys):
                    continue
                # color = COLORS[groups.index(group) % len(COLORS)]
                color = {
                    r'GAIL($\lambda=0.0$)': 'darkgreen',
                    r'GAIL($\lambda=0.1$)': 'red',
                    r'GAIL($\lambda=1.0$)': 'blue',
                    r'GAIL($\lambda=10.0$)': 'orange',
                }[group]
                origxs = [xy[0] for xy in xys]
                minxlen = min(map(len, origxs))

                def allequal(qs):
                    return all((q == qs[0]).all() for q in qs[1:])

                if resample:
                    low = max(x[0] for x in origxs)
                    high = min(x[-1] for x in origxs)
                    usex = np.linspace(low, high, resample)
                    ys = []
                    for (x, y) in xys:
                        ys.append(pu.symmetric_ema(x, y, low, high, resample, decay_steps=smooth_step)[1])
                else:
                    assert allequal([x[:minxlen] for x in origxs]), \
                        'If you want to average unevenly sampled data, set resample=<number of samples you want>'
                    usex = origxs[0]
                    ys = [xy[1][:minxlen] for xy in xys]
                ymean = np.mean(ys, axis=0)
                ystd = np.std(ys, axis=0)
                ystderr = ystd / np.sqrt(len(ys))
                l, = axarr[0][isplit].plot(usex, ymean, color=color)

                xticks = range(0, int(40e5), int(10e5))
                xticklables = ['0', '1M',  '2M', '3M']
                print(sk, group, ymean[-1], ystd[-1])
                g2l[group] = l
                if shaded_err:
                    ax.fill_between(usex, ymean - ystderr, ymean + ystderr, color=color, alpha=.4)
                if shaded_std:
                    ax.fill_between(usex, ymean - ystd, ymean + ystd, color=color, alpha=.2)

                axarr[0][isplit].set_xticks(xticks)
                axarr[0][isplit].grid(True)
                axarr[0][isplit].set_xticklabels(xticklables)

        if sk == 'Hopper-v2':
            ax.set_title('Hopper-v2')
            g2l['Expert'] = ax.axhline(2223.49, usex[0], usex[-1], ls='--')
        if sk == 'HalfCheetah-v2':
            ax.set_title('HalfCheetah-v2')
            g2l['Expert'] = ax.axhline(4097.30, usex[0], usex[-1], ls='--')
        if sk == 'Walker2d-v2':
            ax.set_title('Walker2d-v2')
            g2l['Expert'] = ax.axhline(3151.77, usex[0], usex[-1], ls='--')
        # https://matplotlib.org/users/legend_guide.html
        plt.tight_layout()
    g2l_sorted = collections.OrderedDict()
    for key in ['Expert'] + list(g2l.keys())[:-1]:
        g2l_sorted[key] = g2l[key]
    g2l = g2l_sorted
    if any(g2l.keys()):
        f.legend(
            g2l.values(),
            ['%s' % g for g in g2l] if average_group else g2l.keys(),
            loc='upper center' if legend_outside else None,
            fancybox=True,
            ncol=5,
            bbox_to_anchor=(0.5, 1.2) if legend_outside else None)

    # add xlabels, but only to the bottom row
    if xlabel is not None:
        for ax in axarr[-1]:
            plt.sca(ax)
            plt.xlabel(xlabel)
    # add ylabels, but only to left column
    if ylabel is not None:
        for ax in axarr[:, 0]:
            plt.sca(ax)
            plt.ylabel(ylabel)
    return f, axarr


def plot_mbrl_results(
        allresults, *,
        xy_fn=pu.default_xy_fn,
        split_fn=pu.default_split_fn,
        group_fn=pu.default_group_fn,
        average_group=False,
        shaded_std=True,
        shaded_err=True,
        figsize=None,
        legend_outside=False,
        resample=0,
        smooth_step=1.0,
        xlabel=None,
        ylabel=None,
):
    '''
    Plot multiple Results objects

    xy_fn: function Result -> x,y           - function that converts results objects into tuple of x and y values.
                                              By default, x is cumsum of episode lengths, and y is episode rewards

    split_fn: function Result -> hashable   - function that converts results objects into keys to split curves into sub-panels by.
                                              That is, the results r for which split_fn(r) is different will be put on different sub-panels.
                                              By default, the portion of r.dirname between last / and -<digits> is returned. The sub-panels are
                                              stacked vertically in the figure.

    group_fn: function Result -> hashable   - function that converts results objects into keys to group curves by.
                                              That is, the results r for which group_fn(r) is the same will be put into the same group.
                                              Curves in the same group have the same color (if average_group is False), or averaged over
                                              (if average_group is True). The default value is the same as default value for split_fn

    average_group: bool                     - if True, will average the curves in the same group and plot the mean. Enables resampling
                                              (if resample = 0, will use 512 steps)

    shaded_std: bool                        - if True (default), the shaded region corresponding to standard deviation of the group of curves will be
                                              shown (only applicable if average_group = True)

    shaded_err: bool                        - if True (default), the shaded region corresponding to error in mean estimate of the group of curves
                                              (that is, standard deviation divided by square root of number of curves) will be
                                              shown (only applicable if average_group = True)

    figsize: tuple or None                  - size of the resulting figure (including sub-panels). By default, width is 6 and height is 6 times number of
                                              sub-panels.


    legend_outside: bool                    - if True, will place the legend outside of the sub-panels.

    resample: int                           - if not zero, size of the uniform grid in x direction to resample onto. Resampling is performed via symmetric
                                              EMA smoothing (see the docstring for symmetric_ema).
                                              Default is zero (no resampling). Note that if average_group is True, resampling is necessary; in that case, default
                                              value is 512.

    smooth_step: float                      - when resampling (i.e. when resample > 0 or average_group is True), use this EMA decay parameter (in units of the new grid step).
                                              See docstrings for decay_steps in symmetric_ema or one_sided_ema functions.

    '''

    if split_fn is None: split_fn = lambda _: ''
    if group_fn is None: group_fn = lambda _: ''
    sk2r = collections.defaultdict(list)  # splitkey2results
    for result in allresults:
        splitkey = split_fn(result)
        sk2r[splitkey].append(result)
    assert len(sk2r) > 0
    assert isinstance(resample, int), "0: don't resample. <integer>: that many samples"
    nrows = 1
    ncols = len(sk2r)
    figsize = figsize or (6.5 * ncols, 4 * nrows)
    f, axarr = plt.subplots(nrows, ncols, sharex=False, squeeze=False, figsize=figsize, dpi=300)

    groups = list(set(group_fn(result) for result in allresults))
    groups.sort()

    default_samples = 512
    # if average_group:
    #     resample = resample or default_samples

    for (isplit, sk) in enumerate(sorted(sk2r.keys())):
        g2l = {}
        g2c = collections.defaultdict(int)
        sresults = sk2r[sk]
        gresults = collections.defaultdict(list)
        ax = axarr[0][isplit]
        for result in sresults:
            group = group_fn(result)
            g2c[group] += 1
            x, y = xy_fn(result)
            if x is None: x = np.arange(len(y))
            x, y = map(np.asarray, (x, y))
            if average_group:
                gresults[group].append((x, y))
            else:
                if resample:
                    x, y, counts = symmetric_ema(x, y, x[0], x[-1], resample, decay_steps=smooth_step)
                l, = ax.plot(x, y, color=COLORS[groups.index(group) % len(COLORS)])
                g2l[group] = l
        if average_group:
            for group in sorted(groups):
                xys = gresults[group]
                if not any(xys):
                    continue
                # color = COLORS[groups.index(group) % len(COLORS)]
                color = {
                    'MB-BC': 'darkgreen',
                    'MB-GAIL': 'red',
                }[group]
                origxs = [xy[0] for xy in xys]
                minxlen = min(map(len, origxs))

                def allequal(qs):
                    return all((q == qs[0]).all() for q in qs[1:])

                if resample:
                    low = max(x[0] for x in origxs)
                    high = min(x[-1] for x in origxs)
                    usex = np.linspace(low, high, resample)
                    ys = []
                    for (x, y) in xys:
                        ys.append(pu.symmetric_ema(x, y, low, high, resample, decay_steps=smooth_step)[1])
                else:
                    assert allequal([x[:minxlen] for x in origxs]), \
                        'If you want to average unevenly sampled data, set resample=<number of samples you want>'
                    usex = origxs[0]
                    ys = [xy[1][:minxlen] for xy in xys]
                ymean = np.mean(ys, axis=0)
                ystd = np.std(ys, axis=0)
                ystderr = ystd / np.sqrt(len(ys))
                usex = np.arange(150)
                l, = axarr[0][isplit].plot(usex, ymean, color=color)

                xticks = range(0, 200, 50)
                xticklables = ['0', '100', '200', '300']
                print(sk, group, ymean[-1], ystd[-1])
                g2l[group] = l
                if shaded_err:
                    ax.fill_between(usex, ymean - ystderr, ymean + ystderr, color=color, alpha=.4)
                if shaded_std:
                    ax.fill_between(usex, ymean - ystd, ymean + ystd, color=color, alpha=.2)

                axarr[0][isplit].set_xticks(xticks)
                axarr[0][isplit].grid(True)
                axarr[0][isplit].set_xticklabels(xticklables)

        ax.set_title(sk)
        # https://matplotlib.org/users/legend_guide.html
        plt.tight_layout()
    if any(g2l.keys()):
        f.legend(
            g2l.values(),
            ['%s' % g for g in g2l] if average_group else g2l.keys(),
            loc='upper center' if legend_outside else None,
            fancybox=True,
            ncol=2,
            bbox_to_anchor=(0.5, 1.2) if legend_outside else None)

    # add xlabels, but only to the bottom row
    if xlabel is not None:
        for ax in axarr[-1]:
            plt.sca(ax)
            plt.xlabel(xlabel)
    # add ylabels, but only to left column
    if ylabel is not None:
        for ax in axarr[:, 0]:
            plt.sca(ax)
            plt.ylabel(ylabel)
    return f, axarr


def plot_dist(allresults):
    env_list = ['HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
    method_list = ['Expert', 'DAgger', 'GAIL', 'WGAIL', 'FEM', 'GTAL',  'BC']
    dict_result = {
        0.9:    {method: {env: [] for env in env_list} for method in method_list},
        0.99:   {method: {env: [] for env in env_list} for method in method_list},
        0.999:  {method: {env: [] for env in env_list} for method in method_list},
    }
    for result in allresults:
        xs, ys = xy_fn(result)
        method = group_fn(result)
        env = split_fn(result)
        for x, y in zip(xs, ys):
            dict_result[x][method][env].append(y)
    for key, val in dict_result.items():
        for key_, val_ in val.items():
            for key__, val__ in val_.items():
                if key_ == 'Expert':
                    assert len(val__) == 1
                    val_[key__] = [np.mean(val__), 0.]
                else:
                    assert len(val__) == 3, 'gamma={}, method={}, env={}, val={}'.format(key, key_, key__, val__)
                    val_[key__] = [np.mean(val__), np.std(val__)]

    ncols= len(env_list)
    nrows = 1
    figsize = (6 * ncols, 4 * nrows)
    f, axarr = plt.subplots(nrows, ncols, sharex=False, squeeze=False, figsize=figsize, dpi=300)
    plt.subplots_adjust(wspace=0.3, hspace=0.35)

    for i in range(len(dict_result)):
        ax = axarr[0][i]
        gamma = list(dict_result.keys())[i]
        data = dict_result[gamma]
        x = np.arange(0, 2*len(env_list), 2).astype(np.float32) - 0.5
        for method in data.keys():
            performance = [meanstd[0] for meanstd in data[method].values()]
            err = [meanstd[1] for meanstd in data[method].values()]
            color = COLORS[method_list.index(method)]
            ax.bar(x=x, height=performance,
                   width=0.15, ecolor='black', alpha=0.8, capsize=5,
                   color=color, label=method)
            x += 0.15
        x = np.arange(0, 2*len(env_list), 2).astype(np.float32)
        ax.set_xticks(x)
        ax.set_xticklabels(env_list, rotation=0, fontsize=13)
        ax.yaxis.grid(True)
        plt.tight_layout()
        ax.set_title(r'Discount factor $\gamma={}$'.format(gamma))

    axarr[0][1].legend(
        loc='upper center',
        fancybox=False,
        ncol=len(method_list),
        bbox_to_anchor=(0.5, 1.35),
        edgecolor='black'
    )

    ylabel = 'Return'
    for ax in axarr[:, 0]:
        plt.sca(ax)
        plt.ylabel(ylabel)
    return f, axarr


def main(root_dir):
    all_results = []
    for root, dirs, files in os.walk(root_dir):
        for dir_ in sorted(dirs):
            if 'Hopper-v2' in dir_ or 'Walker2d-v2' in dir_ or 'HalfCheetah-v2' in dir_:
                progress_path = os.path.join(root, dir_, 'evaluate.yml')
                progress = read_evaluate_yaml_file(progress_path)
                all_results.append(Result(progress=progress, dir=dir_, root_dir=root))
    print('load %s results' % len(all_results))

    fig, axes = plot_results(all_results, xy_fn=xy_fn, split_fn=split_fn, group_fn=group_fn,
                             resample=0, average_group=True, shaded_std=True, shaded_err=False,
                             legend_outside=True, xlabel=r'Discount Factor $\gamma$', ylabel='Discounted Return')
    save_path = os.path.join(root_dir, 'result.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    save_path = os.path.join(root_dir, 'result.png')
    plt.savefig(save_path, bbox_inches='tight')
    print('save result fig into %s' % save_path)

    fig, axes = plot_dist(all_results)
    save_path = os.path.join(root_dir, 'gamma_return_dist.png')
    plt.savefig(save_path, bbox_inches='tight')
    save_path = os.path.join(root_dir, 'gamma_return_dist.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    print('save result fig into %s' % save_path)


def plot_gail_regular_l2(root_dir):
    all_results = []
    for root, dirs, files in os.walk(root_dir):
        for dir_ in sorted(dirs):
            if 'Hopper-v2' in dir_ or 'Walker2d-v2' in dir_ or 'HalfCheetah-v2' in dir_:
                progress_path = os.path.join(root, dir_, 'progress.csv')
                progress = read_progress(progress_path)
                all_results.append(Result(progress=progress, dir=dir_, root_dir=root))
    print('load %s results' % len(all_results))

    fig, axes = plot_gail_regular_l2_results(
        all_results, xy_fn=gail_regular_l2_xy_fn, split_fn=split_fn, group_fn=gail_regular_l2_group_fn,
        resample=0, average_group=True, shaded_std=True, shaded_err=False,
        legend_outside=True, xlabel='Number of Samples', ylabel='Return')
    save_path = os.path.join(root_dir, 'gail_regular_l2.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    save_path = os.path.join(root_dir, 'gail_regular_l2.png')
    plt.savefig(save_path, bbox_inches='tight')
    print('save result fig into %s' % save_path)


def plot_mbrl_figure(root_dir):
    all_results = []
    for root, dirs, files in os.walk(root_dir):
        for dir_ in sorted(dirs):
            if 'Hopper-v2' in dir_ or 'Walker2d-v2' in dir_ or 'HalfCheetah-v2' in dir_:
                progress_path = os.path.join(root, dir_, 'progress.csv')
                progress = read_progress(progress_path)
                all_results.append(Result(progress=progress, dir=dir_, root_dir=root))
    print('load %s results' % len(all_results))

    fig, axes = plot_mbrl_results(
        all_results, xy_fn=mbrl_xy_stochastic_fn, split_fn=split_fn, group_fn=mbrl_group_fn,
        resample=0, average_group=True, shaded_std=True, shaded_err=False,
        legend_outside=True, xlabel='Number of Training Epochs', ylabel='Policy Evaluation Error')

    save_path = os.path.join(root_dir, 'mbrl_stochastic.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    save_path = os.path.join(root_dir, 'mbrl_stochastic.png')
    plt.savefig(save_path, bbox_inches='tight')
    print('save result fig into %s' % save_path)

    fig, axes = plot_mbrl_results(
        all_results, xy_fn=mbrl_xy_deterministic_fn, split_fn=split_fn, group_fn=mbrl_group_fn,
        resample=0, average_group=True, shaded_std=True, shaded_err=False,
        legend_outside=True, xlabel='Number of Training Epochs', ylabel='Policy Evaluation Error')

    save_path = os.path.join(root_dir, 'mbrl_deterministic.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    save_path = os.path.join(root_dir, 'mbrl_deterministic.png')
    plt.savefig(save_path, bbox_inches='tight')
    print('save result fig into %s' % save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='benchmarks')

    args = parser.parse_args()

    # main(args.root_dir)
    # plot_gail_regular_l2(root_dir=args.root_dir)
    plot_mbrl_figure(root_dir=args.root_dir)
