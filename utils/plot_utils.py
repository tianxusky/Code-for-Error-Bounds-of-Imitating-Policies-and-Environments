import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import numpy as np


COLORS = ['darkgreen', 'red', 'blue', 'orange', 'black', 'yellow', 'magenta', 'cyan', 'purple', 'pink',
          'brown', 'orange', 'teal', 'lightblue', 'lime', 'lavender', 'turquoise',
          'green', 'tan', 'salmon', 'gold', 'darkred', 'darkblue']

def smooth(y, radius, mode='two_sided', valid_only=False):
    '''
    Smooth signal y, where radius is determines the size of the window

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]

    valid_only: put nan in entries where the full-sized window is not available

    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out

def one_sided_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    '''
    perform one-sided (causal) EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]

    Arguments:

    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds

    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]

    n: int                - number of points in new x grid

    decay_steps: float    - EMA decay factor, expressed in new x grid steps.

    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN

    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid

    '''

    low = xolds[0] if low is None else low
    high = xolds[-1] if high is None else high

    assert xolds[0] <= low, 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
    assert xolds[-1] >= high, 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(high, xolds[-1])
    assert len(xolds) == len(yolds), 'length of xolds ({}) and yolds ({}) do not match!'.format(len(xolds), len(yolds))

    xolds = xolds.astype('float64')
    yolds = yolds.astype('float64')

    luoi = 0  # last unused old index
    sum_y = 0.
    count_y = 0.
    xnews = np.linspace(low, high, n)
    decay_period = (high - low) / (n - 1) * decay_steps
    interstep_decay = np.exp(- 1. / decay_steps)
    sum_ys = np.zeros_like(xnews)
    count_ys = np.zeros_like(xnews)
    for i in range(n):
        xnew = xnews[i]
        sum_y *= interstep_decay
        count_y *= interstep_decay
        while True:
            xold = xolds[luoi]
            if xold <= xnew:
                decay = np.exp(- (xnew - xold) / decay_period)
                sum_y += decay * yolds[luoi]
                count_y += decay
                luoi += 1
            else:
                break
            if luoi >= len(xolds):
                break
        sum_ys[i] = sum_y
        count_ys[i] = count_y

    ys = sum_ys / count_ys
    ys[count_ys < low_counts_threshold] = np.nan

    return xnews, ys, count_ys


def symmetric_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    '''
    perform symmetric EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]

    Arguments:

    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds

    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]

    n: int                - number of points in new x grid

    decay_steps: float    - EMA decay factor, expressed in new x grid steps.

    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN

    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid

    '''
    xs, ys1, count_ys1 = one_sided_ema(xolds, yolds, low, high, n, decay_steps, low_counts_threshold=0)
    _, ys2, count_ys2 = one_sided_ema(-xolds[::-1], yolds[::-1], -high, -low, n, decay_steps, low_counts_threshold=0)
    ys2 = ys2[::-1]
    count_ys2 = count_ys2[::-1]
    count_ys = count_ys1 + count_ys2
    ys = (ys1 * count_ys1 + ys2 * count_ys2) / count_ys
    ys[count_ys < low_counts_threshold] = np.nan
    return xs, ys, count_ys


def default_xy_fn(r):
    progress = r.progress
    x_name, y_name = 'iter', 'episode_return'
    for key in progress.keys():
        if y_name in key:
            y_name = key
        if x_name in key:
            x_name = key
    x = progress[x_name]
    y = progress[y_name]
    return x, y


def default_split_fn(r):
    splits = r.dir.split('-')
    env_name = splits[1] + '-' + splits[2]
    return env_name


def default_group_fn(r):
    splits = r.dir.split('-')
    alg_name = splits[0]
    return alg_name


def plot_results(
        allresults, *,
        xy_fn=default_xy_fn,
        split_fn=default_split_fn,
        group_fn=default_group_fn,
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
    figsize = figsize or (6 * ncols, 4 * nrows)
    f, axarr = plt.subplots(nrows, ncols, sharex=False, squeeze=False, figsize=figsize, dpi=300)

    groups = list(set(group_fn(result) for result in allresults))
    groups.sort()

    default_samples = 512
    if average_group:
        resample = resample or default_samples

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
                color = COLORS[groups.index(group) % len(COLORS)]
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
                        ys.append(symmetric_ema(x, y, low, high, resample, decay_steps=smooth_step)[1])
                else:
                    assert allequal([x[:minxlen] for x in origxs]), \
                        'If you want to average unevenly sampled data, set resample=<number of samples you want>'
                    usex = origxs[0]
                    ys = [xy[1][:minxlen] for xy in xys]
                ymean = np.mean(ys, axis=0)
                ystd = np.std(ys, axis=0)
                ystderr = ystd / np.sqrt(len(ys))
                l, = axarr[0][isplit].plot(usex, ymean, color=color)
                print(sk, group, ymean[-1], ystd[-1])
                g2l[group] = l
                if shaded_err:
                    ax.fill_between(usex, ymean - ystderr, ymean + ystderr, color=color, alpha=.4)
                if shaded_std:
                    ax.fill_between(usex, ymean - ystd, ymean + ystd, color=color, alpha=.2)

        if sk == 'Hopper-v2':
            if 'mbrl' in group:
                pass
                # ax.axhline(3557.428054, usex[0], usex[-1], ls='--')
            else:
                ax.axhline(2223.49, usex[0], usex[-1], ls='--')
        if sk == 'HalfCheetah-v2':
            if 'mbrl' in group:
                pass
                # ax.axhline(6618.1416, usex[0], usex[-1], ls='--')
            else:
                ax.axhline(4097.30, usex[0], usex[-1], ls='--')
        if sk == 'Walker2d-v2':
            if 'mbrl' in group:
                pass
                # ax.axhline(2791.4262, usex[0], usex[-1], ls='--')
            else:
                ax.axhline(3151.77, usex[0], usex[-1], ls='--')
        # https://matplotlib.org/users/legend_guide.html
        plt.tight_layout()
        if any(g2l.keys()):
            ax.legend(
                g2l.values(),
                ['%s' % g for g in g2l] if average_group else g2l.keys(),
                loc=2 if legend_outside else None,
                bbox_to_anchor=(1, 1) if legend_outside else None)
        ax.set_title(sk)

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

