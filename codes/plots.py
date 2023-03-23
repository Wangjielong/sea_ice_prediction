import numpy as np
from utility import non_land_mask
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data_preprocessing import read_sic

plt.rc('font', family='Times New Roman')

epsilon = 0.00001


def corr(x, y):
    """
    calculate the correlation coefficient
    """
    dim_x = len(x.shape)
    dim_y = len(y.shape)
    assert dim_x == dim_y
    if dim_x <= 2:
        x_mean, y_mean = np.mean(x), np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sqrt(np.sum((x - x_mean) ** 2)) * np.sqrt(np.sum((y - y_mean) ** 2))
        return numerator / (denominator + epsilon)
    elif dim_x == 3:  # calculate the spatial corr
        x_mean, y_mean = np.mean(x, axis=0), np.mean(y, axis=0)
        numerator = np.sum((x - x_mean) * (y - y_mean), axis=0)
        denominator = np.sqrt(np.sum((x - x_mean) ** 2, axis=0)) * np.sqrt(np.sum((y - y_mean) ** 2, axis=0))
        return numerator / (denominator + epsilon)


def mae(result='result_melting/none.npz'):
    """
    mean absolute error (MAE)
    np.savez('result/none.npz', target=target_test, model=model_test, time=testY_T)
    :param result:
    :return:
    """
    res = np.load(result, allow_pickle=True)

    target = res['target']
    prediction = res['model']
    time = res['time']

    diff = np.abs(target - prediction)
    diff_sum = np.sum(diff, axis=(1, 2))

    non_land = non_land_mask()
    total = np.sum(non_land)

    MAE = diff_sum / total

    return MAE, time


def nse(result='result_melting/none.npz'):
    """
    Nash–Sutcliffe efficiency (NSE)
    :param result:
    :return:
    """
    res = np.load(result, allow_pickle=True)

    target = res['target']
    prediction = res['model']
    time = res['time']

    NSE = np.zeros((len(target), 1))
    numerator = np.sum((target - prediction) ** 2, axis=(1, 2))

    non_land = non_land_mask()
    total = np.sum(non_land)

    for ii in range(len(target)):
        mean_target = np.sum(target[ii]) / total
        denominator = np.sum((target[ii] - mean_target) ** 2)
        NSE[ii] = 1 - numerator[ii] / denominator

    return NSE, time


# Figure 2, Figure 3
def plot_spatial(result='result_melting/none.npz', period='melting'):
    res = np.load(result, allow_pickle=True)
    target = res['target']
    prediction = res['model']
    time = res['time']

    non_land = non_land_mask()
    land_mask = ~(np.array(non_land, dtype=bool))

    nrows = 3
    ncols = 6
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(28, 24))

    getattr(mpl.cm, 'Blues_r').set_bad(color='green')
    cmap = getattr(mpl.cm, 'Blues_r')

    cmap_name = 'my_list'
    colors = [(8 / 255.0, 48 / 255.0, 107 / 255.0), (0, 0, 1), (1, 1.0, 0), (0, 1, 1),
              (1, 0, 0)]
    cmap_mae = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=5)
    cmap_mae.set_bad(color='green')

    clim = (0, 1)

    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0.1, hspace=0.2)  # set the spacing between axes.
    axes = []

    title_list = [chr(i) for i in range(97, 115)]
    date_list = []
    if period == 'melting':
        date_list = ['Apr 2021', 'May 2021', 'Jun 2021', 'Jul 2021', 'Aug 2021', 'Sept 2021']
    elif period == 'icing':
        date_list = ['Jan 2021', 'Feb 2021', 'Mar 2021', 'Oct 2021', 'Nov 2021', 'Dec 2021']
    elif period == 'whole_melting':
        date_list = ['Apr 2021', 'May 2021', 'Jun 2021', 'Jul 2021', 'Aug 2021', 'Sept 2021']
        # select the Apr-Sept 2021
        time = time[-9:-3, ...]
        target = target[-9:-3, ...]
        prediction = prediction[-9:-3, ...]
    elif period == 'whole_icing':
        date_list = ['Jan 2021', 'Feb 2021', 'Mar 2021', 'Oct 2021', 'Nov 2021', 'Dec 2021']
        time = np.concatenate((time[-12:-9, ...], time[-3:, ...]), axis=0)
        target = np.concatenate((target[-12:-9, ...], target[-3:, ...]), axis=0)
        prediction = np.concatenate((prediction[-12:-9, ...], prediction[-3:, ...]), axis=0)

    img_id = 0

    for row in range(nrows):
        for col in range(ncols):
            ax = plt.subplot(gs.new_subplotspec((row, col)))
            ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

            title = f'({title_list[img_id]}) ' + date_list[col]
            t = plt.text(8, 20, title, fontweight='bold', fontsize=20)
            t.set_bbox(dict(facecolor='lightgray', alpha=0.8))

            if row == 0:
                # target
                target_img = target[-6 + col]
                target_img[land_mask] = np.nan
                im_target0 = ax.imshow(target_img, cmap=cmap, clim=clim)

                if col == 5:
                    position = fig.add_axes([0.60, 0.64, 0.3, 0.01])  # left bottom width height
                    cbar = fig.colorbar(im_target0, cax=position, orientation='horizontal')
                    cbar.set_label('Observed SIC', fontsize=25, labelpad=10, fontweight='bold')
                    cbar.ax.tick_params(labelsize=24)

            if row == 1:
                # prediction
                prediction_img = prediction[-6 + col]
                prediction_img[land_mask] = np.nan
                im_prediction = ax.imshow(prediction_img, cmap=cmap, clim=clim)

                if col == 5:
                    position = fig.add_axes([0.60, 0.37, 0.3, 0.01])  # left bottom width height
                    cbar = fig.colorbar(im_prediction, cax=position, orientation='horizontal')
                    cbar.set_label('Predicted SIC', fontsize=24, labelpad=10, fontweight='bold')
                    cbar.ax.tick_params(labelsize=24)

            if row == 2:
                # mae
                mae_img = np.abs(prediction[-6 + col] - target[-6 + col])
                mae_img[land_mask] = np.nan
                #  special processing
                mae_img[mae_img > 0.75] = 0
                im_mae = ax.imshow(mae_img, cmap=cmap_mae, clim=clim)

                if col == 5:
                    position = fig.add_axes([0.60, 0.1, 0.3, 0.01])  # left bottom width height
                    cbar = fig.colorbar(im_mae, cax=position, orientation='horizontal')
                    cbar.set_label('MAE', fontsize=24, labelpad=10, fontweight='bold')
                    cbar.ax.tick_params(labelsize=24)

            axes.append(ax)
            img_id += 1

    # fig.savefig('test.png', format='png', bbox_inches='tight', dpi=1200)

    return time


def plot_diff():
    non_land = non_land_mask()
    land_mask = ~(np.array(non_land, dtype=bool))

    res_melting = np.load('result_melting/none.npz', allow_pickle=True)
    # target_melting = res_melting['target']
    prediction_melting = res_melting['model']
    # time_melting = res_melting['time']

    res_icing = np.load('result_icing/none.npz', allow_pickle=True)
    # target_icing = res_icing['target']
    prediction_icing = res_icing['model']
    # time_icing = res_icing['time']

    res_whole = np.load('result_whole/none.npz', allow_pickle=True)
    # target_whole = res_whole['target']
    prediction_whole = res_whole['model']
    # time_whole = res_whole['time']

    # icing_melting_time = np.concatenate(
    #     (time_icing[-6:-3, ...], time_melting[-6:, ...], time_icing[-3:, ...]))

    icing_melting = np.concatenate(
        (prediction_icing[-6:-3, ...], prediction_melting[-6:, ...], prediction_icing[-3:, ...]))

    whole = prediction_whole[-12:, ...]

    title_list = [chr(i) for i in range(97, 109)]
    date_list = ['Jan 2021', 'Feb 2021', 'Mar 2021', 'Apr 2021', 'May 2021', 'Jun 2021', 'Jul 2021', 'Aug 2021',
                 'Sept 2021', 'Oct 2021', 'Nov 2021', 'Dec 2021']

    cmap_name = 'my_list'
    colors = [(3 / 255.0, 168 / 255.0, 158 / 255.0), (0, 0, 1),
              (245 / 255.0, 245 / 255.0, 245 / 255.0), (1, 127 / 255.0, 80 / 255.0),
              (156 / 255.0, 102 / 255.0, 31 / 255.0),
              (1, 0, 0)]
    cmap_mae = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=6)
    cmap_mae.set_bad(color=(211 / 255.0, 211 / 255.0, 211 / 255.0))
    clim = (0, 0.6)

    nrows = 3
    ncols = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(28, 24))
    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0.05, hspace=0.05)  # set the spacing between axes.
    axes = []

    img_id = 0
    for row in range(nrows):
        for col in range(ncols):
            ax = plt.subplot(gs.new_subplotspec((row, col)))
            ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

            title = f'({title_list[img_id]}) ' + date_list[img_id]
            t = plt.text(8, 20, title, fontweight='bold', fontsize=20)
            t.set_bbox(dict(facecolor='lightgray', alpha=0.8))

            diff_img = np.abs(icing_melting[img_id] - whole[img_id])
            diff_img[land_mask] = np.nan

            im_diff = ax.imshow(diff_img, cmap=cmap_mae, clim=clim)

            axes.append(ax)
            img_id += 1

            if row == 2 and col == 3:
                position = fig.add_axes([0.60, 0.08, 0.3, 0.01])  # left bottom width height
                cbar = fig.colorbar(im_diff, cax=position, orientation='horizontal', extend='max')
                cbar.set_label('MAE', fontsize=24, labelpad=10, fontweight='bold')
                cbar.ax.tick_params(labelsize=24)


def plot_sept():
    non_land = non_land_mask()
    land_mask = ~(np.array(non_land, dtype=bool))

    res_melting = np.load('result_melting/none.npz', allow_pickle=True)
    target = res_melting['target']
    prediction = res_melting['model']

    title_list = [chr(i) for i in range(97, 112)]
    date_list = ['Sept 2017', 'Sept 2018', 'Sept 2019', 'Sept 2020', 'Sept 2021']

    getattr(mpl.cm, 'Blues_r').set_bad(color='green')
    cmap = getattr(mpl.cm, 'Blues_r')

    cmap_name = 'my_list'
    colors = [(8 / 255.0, 48 / 255.0, 107 / 255.0), (0, 0, 1), (1, 1.0, 0), (0, 1, 1),
              (1, 0, 0)]
    cmap_mae = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=5)
    cmap_mae.set_bad(color='green')
    clim = (0, 1)

    nrows = 3
    ncols = 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(28, 24))
    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0.1, hspace=0.3)  # set the spacing between axes.
    axes = []

    img_id = 0
    for row in range(nrows):
        for col in range(ncols):
            ax = plt.subplot(gs.new_subplotspec((row, col)))
            ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

            title = f'({title_list[img_id]}) ' + date_list[col]
            t = plt.text(8, 20, title, fontweight='bold', fontsize=20)
            t.set_bbox(dict(facecolor='lightgray', alpha=0.8))

            if row == 0:
                # target
                target_img = target[col * 6 + 1]
                target_img[land_mask] = np.nan
                im_target = ax.imshow(target_img, cmap=cmap, clim=clim)

                if col == ncols - 1:
                    position = fig.add_axes([0.60, 0.65, 0.3, 0.01])  # left bottom width height
                    cbar = fig.colorbar(im_target, cax=position, orientation='horizontal')
                    cbar.set_label('Observed SIC', fontsize=25, labelpad=10, fontweight='bold')
                    cbar.ax.tick_params(labelsize=24)

            if row == 1:
                # prediction
                prediction_img = prediction[col * 6 + 1]
                prediction_img[land_mask] = np.nan
                im_prediction = ax.imshow(prediction_img, cmap=cmap, clim=clim)

                if col == ncols - 1:
                    position = fig.add_axes([0.60, 0.37, 0.3, 0.01])  # left bottom width height
                    cbar = fig.colorbar(im_prediction, cax=position, orientation='horizontal')
                    cbar.set_label('Predicted SIC', fontsize=24, labelpad=10, fontweight='bold')
                    cbar.ax.tick_params(labelsize=24)

            if row == 2:
                # mae
                mae_img = np.abs(prediction[col * 6 + 1] - target[col * 6 + 1])
                mae_img[land_mask] = np.nan
                #  special processing
                mae_img[mae_img > 0.75] = 0
                im_mae = ax.imshow(mae_img, cmap=cmap_mae, clim=clim)

                if col == ncols - 1:
                    position = fig.add_axes([0.60, 0.09, 0.3, 0.01])  # left bottom width height
                    cbar = fig.colorbar(im_mae, cax=position, orientation='horizontal')
                    cbar.set_label('MAE', fontsize=24, labelpad=10, fontweight='bold')
                    cbar.ax.tick_params(labelsize=24)

            axes.append(ax)
            img_id += 1


def plot_mae():
    non_land = non_land_mask()

    res_melting = np.load('result_melting/none.npz', allow_pickle=True)
    prediction_melting = res_melting['model']

    res_icing = np.load('result_icing/none.npz', allow_pickle=True)

    prediction_icing = res_icing['model']

    res_whole = np.load('result_whole/none.npz', allow_pickle=True)
    target_whole = res_whole['target']
    prediction_whole = res_whole['model']

    icing_melting_avg = np.zeros((len(target_whole)))
    whole_avg = np.sum(prediction_whole, axis=(1, 2)) / np.sum(non_land)
    target_avg = np.sum(target_whole, axis=(1, 2)) / np.sum(non_land)

    fontsize = 18
    legend_fontsize = 15
    params = {
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'legend.fontsize': legend_fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
    }
    mpl.rcParams.update(params)

    melting_mae, melting_t = mae(result='result_melting/none.npz')
    icing_mae, icing_t = mae(result='result_icing/none.npz')
    whole_mae, whole_t = mae(result='result_whole/none.npz')

    dates = np.arange(np.datetime64('2017-11'), np.datetime64('2022-01'),
                      np.timedelta64(1, 'M'))

    icing_melting_mae = np.zeros(len(dates))

    for ii in range(len(melting_t)):
        year = melting_t[ii][0].year
        month = melting_t[ii][0].month
        month_start = '%02d' % month
        month_end = '%02d' % (month + 1)

        date_start = str(year) + '-' + month_start
        date_end = str(year) + '-' + month_end
        date = np.arange(np.datetime64(date_start), np.datetime64(date_end),
                         np.timedelta64(1, 'M'))

        year_icing = icing_t[ii][0].year
        month_icing = icing_t[ii][0].month

        icing_start = '%02d' % month_icing
        icing_end = '%02d' % (month_icing + 1)

        icing_start = str(year_icing) + '-' + icing_start

        if month_icing + 1 > 12:
            icing_end = str(year_icing + 1) + '-' + '01'
        else:
            icing_end = str(year_icing) + '-' + icing_end
        date_icing = np.arange(np.datetime64(icing_start), np.datetime64(icing_end),
                               np.timedelta64(1, 'M'))

        if date in dates:
            icing_melting_mae[np.where(dates == date)] = melting_mae[ii]

            melting = prediction_melting[ii]
            icing_melting_avg[np.where(dates == date)] = np.sum(melting) / np.sum(non_land)

        if date_icing in dates:
            icing_melting_mae[np.where(dates == date_icing)] = icing_mae[ii]

            icing = prediction_icing[ii]
            icing_melting_avg[np.where(dates == date_icing)] = np.sum(icing) / np.sum(non_land)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))

    ax[1].plot(dates, whole_mae[2:, ...] * 100, color='orange', linewidth=3,
               label=f'WHOLE ({np.mean(whole_mae) * 100:.2f}%)')

    for ii in range(len(dates)):
        if int(str(dates[ii])[5:]) in range(4, 10):
            line_melting, = ax[1].plot(dates[ii], icing_melting_mae[ii] * 100, 'o', color='b', markersize=12,
                                       alpha=1, markerfacecolor=np.array([1, 1, 1]))

        else:
            line_icing, = ax[1].plot(dates[ii], icing_melting_mae[ii] * 100, 's', color='r', markersize=12,
                                     alpha=1, markerfacecolor=np.array([1, 1, 1]))

    ax[1].set_ylabel('MAE [%]')
    ax[1].axhline(y=2.67, color='r', linestyle='--', linewidth=2, label='BM=2.67%')
    line_melting.set_label(f'melting ({np.mean(melting_mae) * 100:.2f}%)')
    line_icing.set_label(f'icing ({np.mean(icing_mae) * 100:.2f}%)')
    ax[1].legend(framealpha=0.9)
    ax[1].set_facecolor(color=(0.3, 0.3, 0.3))
    ax[1].set_title('(b)')

    ax[0].plot(dates, whole_avg[2:, ...] * 100, color='orange', linewidth=3,
               label=f'WHOLE')

    ax[0].plot(dates, target_avg[2:, ...] * 100, color='white', linewidth=3, label=f'Observed SIC')

    for ii in range(len(dates)):
        if int(str(dates[ii])[5:]) in range(4, 10):
            lineMelting, = ax[0].plot(dates[ii], icing_melting_avg[ii] * 100, 'o', color='b', markersize=12,
                                      alpha=1, markerfacecolor=np.array([1, 1, 1]))

        else:
            lineIcing, = ax[0].plot(dates[ii], icing_melting_avg[ii] * 100, 's', color='r', markersize=12,
                                    alpha=1, markerfacecolor=np.array([1, 1, 1]))

    ax[0].set_ylabel('SIC [%]')
    lineMelting.set_label(f'melting')
    lineIcing.set_label(f'icing')
    ax[0].legend(framealpha=0.9)
    ax[0].set_facecolor(color=(0.3, 0.3, 0.3))
    ax[0].set_title('(a)')

    return None


def plot_nse():
    non_land = non_land_mask()

    res_melting = np.load('result_melting/none.npz', allow_pickle=True)
    prediction_melting = res_melting['model']

    res_icing = np.load('result_icing/none.npz', allow_pickle=True)

    prediction_icing = res_icing['model']

    res_whole = np.load('result_whole/none.npz', allow_pickle=True)
    target_whole = res_whole['target']
    prediction_whole = res_whole['model']

    icing_melting_avg = np.zeros((len(target_whole)))
    whole_avg = np.sum(prediction_whole, axis=(1, 2)) / np.sum(non_land)
    target_avg = np.sum(target_whole, axis=(1, 2)) / np.sum(non_land)

    fontsize = 18
    legend_fontsize = 15
    params = {
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'legend.fontsize': legend_fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
    }
    mpl.rcParams.update(params)

    melting_nse, melting_t = nse(result='result_melting/none.npz')
    icing_nse, icing_t = nse(result='result_icing/none.npz')
    whole_nse, whole_t = nse(result='result_whole/none.npz')

    dates = np.arange(np.datetime64('2017-11'), np.datetime64('2022-01'),
                      np.timedelta64(1, 'M'))

    icing_melting_nse = np.zeros(len(dates))

    for ii in range(len(melting_t)):
        year = melting_t[ii][0].year
        month = melting_t[ii][0].month
        month_start = '%02d' % month
        month_end = '%02d' % (month + 1)

        date_start = str(year) + '-' + month_start
        date_end = str(year) + '-' + month_end
        date = np.arange(np.datetime64(date_start), np.datetime64(date_end),
                         np.timedelta64(1, 'M'))

        year_icing = icing_t[ii][0].year
        month_icing = icing_t[ii][0].month

        icing_start = '%02d' % month_icing
        icing_end = '%02d' % (month_icing + 1)

        icing_start = str(year_icing) + '-' + icing_start

        if month_icing + 1 > 12:
            icing_end = str(year_icing + 1) + '-' + '01'
        else:
            icing_end = str(year_icing) + '-' + icing_end
        date_icing = np.arange(np.datetime64(icing_start), np.datetime64(icing_end),
                               np.timedelta64(1, 'M'))

        if date in dates:
            icing_melting_nse[np.where(dates == date)] = melting_nse[ii]

            melting = prediction_melting[ii]
            icing_melting_avg[np.where(dates == date)] = np.sum(melting) / np.sum(non_land)

        if date_icing in dates:
            icing_melting_nse[np.where(dates == date_icing)] = icing_nse[ii]

            icing = prediction_icing[ii]
            icing_melting_avg[np.where(dates == date_icing)] = np.sum(icing) / np.sum(non_land)

    return None


def iiee(result='result_melting/none.npz'):
    """
    integrated ice edge error
    :return:
    """

    res = np.load(result, allow_pickle=True)

    target = res['target']
    prediction = res['model']
    time = res['time']

    prediction[prediction > 0.15] = 1
    prediction[prediction <= 0.15] = 0

    target[target > 0.15] = 1
    target[target <= 0.15] = 0

    diff = np.abs(prediction - target)

    IIEE = np.sum(diff, axis=(1, 2))

    return IIEE, time


def ba(result='result_melting/none.npz'):
    """
    Binary Accuracy
    :param result:
    :return:
    """

    IIEE, time = iiee(result)
    _, sic = read_sic()
    sic[sic > 0.15] = 1
    max_sic_extent = np.max(np.sum(sic, axis=(1, 2)))
    area = max_sic_extent
    BA = 1 - IIEE / area
    return BA, time


def plot_ba():
    melting_ba, melting_t = ba(result='result_melting/none.npz')
    icing_ba, icing_t = ba(result='result_icing/none.npz')
    whole_ba, _ = ba(result='result_whole/none.npz')

    dates = np.arange(np.datetime64('2017-11'), np.datetime64('2022-01'),
                      np.timedelta64(1, 'M'))

    icing_melting_ba = np.zeros(len(dates))

    for ii in range(len(melting_t)):
        year = melting_t[ii][0].year
        month = melting_t[ii][0].month
        month_start = '%02d' % month
        month_end = '%02d' % (month + 1)

        date_start = str(year) + '-' + month_start
        date_end = str(year) + '-' + month_end
        date = np.arange(np.datetime64(date_start), np.datetime64(date_end),
                         np.timedelta64(1, 'M'))

        year_icing = icing_t[ii][0].year
        month_icing = icing_t[ii][0].month

        icing_start = '%02d' % month_icing
        icing_end = '%02d' % (month_icing + 1)

        icing_start = str(year_icing) + '-' + icing_start

        if month_icing + 1 > 12:
            icing_end = str(year_icing + 1) + '-' + '01'
        else:
            icing_end = str(year_icing) + '-' + icing_end
        date_icing = np.arange(np.datetime64(icing_start), np.datetime64(icing_end),
                               np.timedelta64(1, 'M'))

        if date in dates:
            icing_melting_ba[np.where(dates == date)] = melting_ba[ii]

        if date_icing in dates:
            icing_melting_ba[np.where(dates == date_icing)] = icing_ba[ii]

    return whole_ba, icing_melting_ba


def plot_boundary(result='result_melting/none.npz'):
    BA, _ = ba(result)

    res = np.load(result, allow_pickle=True)
    target = res['target']
    prediction = res['model']
    time = res['time']

    non_land = non_land_mask()
    land_mask = ~(np.array(non_land, dtype=bool))

    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 24))

    getattr(mpl.cm, 'Blues_r').set_bad(color='grey')
    cmap = getattr(mpl.cm, 'Blues_r')

    cmap_name = 'my_list'
    colors = [(8 / 255.0, 48 / 255.0, 107 / 255.0), (0, 0, 1), (1, 1.0, 0), (0, 1, 1),
              (1, 0, 0)]
    cmap_mae = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=5)
    cmap_mae.set_bad(color='green')

    clim = (0, 1)

    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0.05, hspace=0.05)  # set the spacing between axes.
    axes = []

    title_list = [chr(i) for i in range(97, 115)]
    date_list = []
    date_list = ['Sept 2018', 'Sept 2019', 'Sept 2020', 'Sept 2021']

    img_id = 0

    for row in range(nrows):
        for col in range(ncols):
            ax = plt.subplot(gs.new_subplotspec((row, col)))
            ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

            title = f'({title_list[img_id]}) ' + date_list[img_id]
            t = plt.text(8, 20, title, fontweight='bold', fontsize=36)
            t.set_bbox(dict(facecolor='lightgray', alpha=0.8))

            if row == 0:
                # target
                target_img = target[7 + 6 * col]
                target_img[land_mask] = np.nan

                bacc = BA[7 + 6 * col]
                t = plt.text(8, 50, f'BA: {bacc * 100:.3f}%', fontweight='bold', fontsize=28)
                t.set_bbox(dict(facecolor='white'))

                prediction_img = prediction[7 + 6 * col]
                prediction_img[land_mask] = np.nan

                im_target0 = ax.imshow(target_img, cmap=cmap, clim=clim)

                ax.contour(target_img, levels=[0.15], colors='orange', linewidths=4.5)
                ax.contour(prediction_img, levels=[0.15], colors='red', linewidths=4.5)

            if row == 1:
                bacc = BA[19 + 6 * col]
                t = plt.text(8, 50, f'BA: {bacc * 100:.3f}%', fontweight='bold', fontsize=28)
                t.set_bbox(dict(facecolor='white'))

                # target
                target_img = target[19 + 6 * col]
                target_img[land_mask] = np.nan

                prediction_img = prediction[19 + 6 * col]
                prediction_img[land_mask] = np.nan

                im_target0 = ax.imshow(target_img, cmap=cmap, clim=clim)

                ax.contour(target_img, levels=[0.15], colors='orange', linewidths=4.5)
                ax.contour(prediction_img, levels=[0.15], colors='red', linewidths=4.5)

                if col == 1:
                    proxy = [plt.Rectangle((0, 0), 1, 1, fc='orange'),
                             plt.Rectangle((0, 0), 1, 1, fc='red')]
                    legend1 = ax.legend(proxy, ['Observed ice edge',
                                                'Predicted ice edge'],
                                        loc='lower right', fontsize=28)

                    # Add the legend manually to the current Axes so that another legend can be added
                    ax.add_artist(legend1)

            axes.append(ax)
            img_id += 1

    return None


# Figure 2, Figure 3
def plot_spatial_modified(result='result_icing/none.npz', period='icing'):
    res = np.load(result, allow_pickle=True)
    target = res['target']
    prediction = res['model']
    time = res['time']

    non_land = non_land_mask()
    land_mask = ~(np.array(non_land, dtype=bool))

    nrows = 3
    ncols = 6
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(28, 22), layout='constrained')

    getattr(mpl.cm, 'Blues_r').set_bad(color='grey')
    cmap = getattr(mpl.cm, 'Blues_r')

    cmap_name = 'my_list'
    colors = ['#082f69', '#084990', '#1663aa', '#2d7dbb', '#4a98c9', '#69add5']
    cmap_mae = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    cmap_mae.set_bad(color='grey')

    clim = (0, 1)

    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0.1, hspace=0.15)  # set the spacing between axes.
    axes = []

    title_list = ['a', 'b', 'c']
    text_list = ['Observed SIC', 'Predicted SIC', 'MAE']

    date_list = []
    if period == 'melting':
        date_list = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept']
    elif period == 'icing':
        date_list = ['Jan', 'Feb', 'Mar', 'Oct', 'Nov', 'Dec']
    elif period == 'whole_melting':
        date_list = ['Apr 2021', 'May 2021', 'Jun 2021', 'Jul 2021', 'Aug 2021', 'Sept 2021']
        # select the Apr-Sept 2021
        time = time[-9:-3, ...]
        target = target[-9:-3, ...]
        prediction = prediction[-9:-3, ...]
    elif period == 'whole_icing':
        date_list = ['Jan 2021', 'Feb 2021', 'Mar 2021', 'Oct 2021', 'Nov 2021', 'Dec 2021']
        time = np.concatenate((time[-12:-9, ...], time[-3:, ...]), axis=0)
        target = np.concatenate((target[-12:-9, ...], target[-3:, ...]), axis=0)
        prediction = np.concatenate((prediction[-12:-9, ...], prediction[-3:, ...]), axis=0)

    img_id = 0

    for row in range(nrows):
        for col in range(ncols):
            ax = plt.subplot(gs.new_subplotspec((row, col)))
            ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

            if row == 2:
                title = date_list[col]
                plt.text(100, 500, title, fontweight='bold', fontsize=32)

            if col == 0:
                plt.text(0, -20, f'({title_list[row]}) {text_list[row]}', fontweight='bold', fontsize=32)

            if row == 0:
                # target
                target_img = target[-6 + col]
                target_img[land_mask] = np.nan
                im_target0 = ax.imshow(target_img, cmap=cmap, clim=clim)

                if col == 5:
                    position = fig.add_axes([0.91, 0.65, 0.012, 0.225])  # left bottom width height
                    cbar = fig.colorbar(im_target0, cax=position, orientation='vertical')
                    # cbar.set_label('Observed SIC', fontsize=25, labelpad=10, fontweight='bold')
                    cbar.ax.tick_params(labelsize=32)
                    # for yl in cbar.ax.yaxis.get_ticklabels():
                    #     yl.set_family('黑体')

            if row == 1:
                # prediction
                prediction_img = prediction[-6 + col]
                prediction_img[land_mask] = np.nan
                im_prediction = ax.imshow(prediction_img, cmap=cmap, clim=clim)

                if col == 5:
                    position = fig.add_axes([0.91, 0.381, 0.012, 0.225])  # left bottom width height
                    cbar = fig.colorbar(im_prediction, cax=position, orientation='vertical')
                    # cbar.set_label('Predicted SIC', fontsize=24, labelpad=10, fontweight='bold')
                    cbar.ax.tick_params(labelsize=32)

            if row == 2:
                # mae
                mae_img = np.abs(prediction[-6 + col] - target[-6 + col])
                mae_img[land_mask] = np.nan
                im_mae = ax.imshow(mae_img, cmap=cmap, clim=clim)

                if col == 5:
                    position = fig.add_axes([0.91, 0.113, 0.012, 0.225])  # left bottom width height
                    cbar = fig.colorbar(im_mae, cax=position, orientation='vertical')
                    # cbar.set_label('MAE', fontsize=24, labelpad=10, fontweight='bold')
                    cbar.ax.tick_params(labelsize=32)

            axes.append(ax)
            img_id += 1

    fig.savefig('graph/Fig2.png', format='png', bbox_inches='tight', dpi=800)

    return time


# Figure 4
def plot_boundary_modified(result='result_melting/none.npz'):
    BA, _ = ba(result)

    res = np.load(result, allow_pickle=True)
    target = res['target']
    prediction = res['model']
    time = res['time']

    non_land = non_land_mask()
    land_mask = ~(np.array(non_land, dtype=bool))

    nrows = 1
    ncols = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6))

    getattr(mpl.cm, 'Blues_r').set_bad(color='grey')
    cmap = getattr(mpl.cm, 'Blues_r')

    cmap_name = 'my_list'
    colors = [(8 / 255.0, 48 / 255.0, 107 / 255.0), (0, 0, 1), (1, 1.0, 0), (0, 1, 1),
              (1, 0, 0)]
    cmap_mae = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=5)
    cmap_mae.set_bad(color='green')

    clim = (0, 1)

    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0.15, hspace=0.05)  # set the spacing between axes.
    axes = []

    title_list = [chr(i) for i in range(97, 115)]
    date_list = []
    date_list = ['Sept 2018', 'Sept 2019', 'Sept 2020', 'Sept 2021']

    img_id = 0
    ice_edge_rgba = mpl.cm.binary(255)

    for row in range(nrows):
        for col in range(ncols):
            ax = plt.subplot(gs.new_subplotspec((row, col)))
            ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

            title = f'({title_list[img_id]}) ' + date_list[img_id]
            # t = plt.text(8, 20, title, fontweight='bold', fontsize=36)
            # t.set_bbox(dict(facecolor='lightgray', alpha=0.8))
            ax.set_title(title, fontsize=10)

            if row == 0:
                # target
                target_img = target[7 + 6 * col]
                target_img[land_mask] = np.nan

                bacc = BA[7 + 6 * col]
                t = plt.text(175, 430, f'BA: {bacc * 100:.3f}%', fontsize=8)
                t.set_bbox(dict(facecolor='white', alpha=0.9))

                prediction_img = prediction[7 + 6 * col]
                prediction_img[land_mask] = np.nan

                im_target0 = ax.imshow(target_img, cmap=cmap, clim=clim)

                ax.contour(target_img, levels=[0.15], colors='orange', linewidths=1.5)
                ax.contour(prediction_img, levels=[0.15], colors='red', linewidths=1.5)

                if col == 0:
                    proxy = [plt.Rectangle((0, 0), 1, 1, fc='orange'),
                             plt.Rectangle((0, 0), 1, 1, fc='red')]
                    legend1 = ax.legend(proxy, ['Observed ice edge',
                                                'Predicted ice edge'],
                                        loc='upper left', fontsize=7)

                    # Add the legend manually to the current Axes so that another legend can be added
                    ax.add_artist(legend1)

                if col == 3:
                    position = fig.add_axes([0.91, 0.28, 0.012, 0.43])  # left bottom width height
                    cbar = fig.colorbar(im_target0, cax=position, orientation='vertical')
                    cbar.set_label('Observed SIC', fontsize=10, labelpad=10)
                    cbar.ax.tick_params(labelsize=10)

            axes.append(ax)
            img_id += 1

    fig.savefig('graph/Fig4.png', format='png', bbox_inches='tight', dpi=1000)
    return None


# Figure 5
def plot_diff_modified():
    non_land = non_land_mask()
    land_mask = ~(np.array(non_land, dtype=bool))

    getattr(mpl.cm, 'Blues_r').set_bad(color='grey')
    cmap = getattr(mpl.cm, 'Blues_r')

    res_melting = np.load('result_melting/none.npz', allow_pickle=True)
    # target_melting = res_melting['target']
    prediction_melting = res_melting['model']
    # time_melting = res_melting['time']

    res_icing = np.load('result_icing/none.npz', allow_pickle=True)
    # target_icing = res_icing['target']
    prediction_icing = res_icing['model']
    # time_icing = res_icing['time']

    res_whole = np.load('result_whole/none.npz', allow_pickle=True)
    # target_whole = res_whole['target']
    prediction_whole = res_whole['model']
    # time_whole = res_whole['time']

    # icing_melting_time = np.concatenate(
    #     (time_icing[-6:-3, ...], time_melting[-6:, ...], time_icing[-3:, ...]))

    icing_melting = np.concatenate(
        (prediction_icing[-6:-3, ...], prediction_melting[-6:, ...], prediction_icing[-3:, ...]))

    whole = prediction_whole[-12:, ...]

    # title_list = [chr(i) for i in range(97, 109)]
    date_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                 'Sept', 'Oct', 'Nov', 'Dec']

    cmap_name = 'my_list'
    colors = [(8 / 255.0, 48 / 255.0, 107 / 255.0), (0, 0, 1), (1, 1.0, 0), (0, 1, 1),
              (1, 0, 0)]
    cmap_mae = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=5)
    cmap_mae.set_bad(color='grey')
    clim = (0, 0.5)

    nrows = 3
    ncols = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 24))
    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0.05, hspace=0.05)  # set the spacing between axes.
    axes = []

    img_id = 0
    for row in range(nrows):
        for col in range(ncols):
            ax = plt.subplot(gs.new_subplotspec((row, col)))
            ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

            title = date_list[img_id]
            t = plt.text(8, 30, title, fontweight='bold', fontsize=36)
            t.set_bbox(dict(facecolor='white', alpha=0.6))

            diff_img = np.abs(icing_melting[img_id] - whole[img_id])
            diff_img[land_mask] = np.nan

            im_diff = ax.imshow(diff_img, cmap=cmap_mae, clim=clim)

            axes.append(ax)
            img_id += 1

            if row == 2 and col == 3:
                position = fig.add_axes([0.90, 0.35, 0.015, 0.3])  # left bottom width height
                cbar = fig.colorbar(im_diff, cax=position, orientation='vertical', extend='max')
                cbar.set_label('MAE', fontsize=36, labelpad=10, fontweight='bold')
                cbar.ax.tick_params(labelsize=36)

    fig.savefig('graph/Fig5.png', format='png', bbox_inches='tight', dpi=800)
    return fig


# scatter plot
def plot_correlation():
    data_icing = np.load('result_icing/none.npz', allow_pickle=True)
    data_melting = np.load('result_melting/none.npz', allow_pickle=True)

    icing_target, melting_target = data_icing['target'], data_melting['target']
    icing_pre, melting_pre = data_icing['model'], data_melting['model']
    icing_t, melting_t = data_icing['time'], data_melting['time']
    # only for 2021
    t = np.concatenate((icing_t[-6:-3], melting_t[-6:], icing_t[-3:]), axis=0)
    obs = np.concatenate((icing_target[-6:-3], melting_target[-6:], icing_target[-3:]), axis=0)
    pre = np.concatenate((icing_pre[-6:-3], melting_pre[-6:], icing_pre[-3:]), axis=0)

    non_land = non_land_mask()
    land_mask = ~(np.array(non_land, dtype=bool))

    fig, ax = plt.subplots(nrows=3, ncols=4, layout='constrained')
    img_idx = 0
    for row in range(3):
        for col in range(4):
            temp_obs, temp_pre = obs[img_idx], pre[img_idx]
            temp_obs[land_mask], temp_pre[land_mask] = np.nan, np.nan
            temp_pre[temp_pre >= 1] = 1
            temp_obs, temp_pre = temp_obs.flatten(), temp_pre.flatten()
            temp_obs = temp_obs[~np.isnan(temp_obs)]
            temp_pre = temp_pre[~np.isnan(temp_pre)]
            ax[row, col].scatter(temp_obs, temp_pre, s=0.5, c='r')
            ax[row, col].plot([0, 1], [0, 1], lw=1, ls='--', c='b')
            print(corr(temp_obs, temp_pre))

            img_idx += 1

    plt.show()


if __name__ == '__main__':
    print('starting...')
    # plot_spatial_modified()
    # plot_boundary_modified()
    plot_spatial_modified()
    plt.show()
