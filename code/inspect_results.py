import time
import os as os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import differential_evolution, minimize
from scipy.optimize import LinearConstraint
import multiprocessing as mp

from matplotlib import rc
import matplotlib as mpl

mpl.use('pgf')
plt.rcParams.update({
    "pgf.texsystem":
    # "xelatex",
    "lualatex",
    "font.family":
    "serif",
    "text.usetex":
    False,
    "pgf.rcfonts":
    False,
    "pgf.preamble": [
        "\\usepackage{fontspec}", "\\usepackage{amsmath,amsfonts,amssymb}",
        "\\usepackage{gensymb}", r"\setmainfont{Arial}"
    ]
})


def bootstrap_ci(x, n, alpha):
    x_boot = np.zeros(n)
    for i in range(n):
        x_boot[i] = np.random.choice(x, x.shape, replace=True).mean()
        ci = np.percentile(x_boot, [alpha / 2, 1.0 - alpha / 2])
    return (ci)


def bootstrap_t(x_obs, y_obs, x_samp_dist, y_samp_dist, n):
    d_obs = x_obs - y_obs

    d_boot = np.zeros(n)
    xs = np.random.choice(x_samp_dist, n, replace=True)
    ys = np.random.choice(y_samp_dist, n, replace=True)
    d_boot = xs - ys
    d_boot = d_boot - d_boot.mean()

    p_null = (1 + np.sum(np.abs(d_boot) > np.abs(d_obs))) / (n + 1)
    return (p_null)


def g_func_cos(theta, theta_mu, k):
    return np.cos(k * np.deg2rad(theta) - theta_mu) + 1


def g_func_gauss(theta, theta_mu, sigma):
    if sigma != 0:
        G = np.exp(-(theta - theta_mu)**2 / (2 * sigma**2))
    else:
        G = np.zeros(11)
    return G


def g_func_flat(amp):
    G = amp * np.ones(11)
    return G


def simulate_state_space_with_g_func_2_state(p, rot):
    alpha_s = p[0]
    beta_s = p[1]
    g_sigma_s = p[2]
    alpha_f = p[3]
    beta_f = p[4]
    g_sigma_f = p[5]
    beta_s_2 = p[6]
    beta_f_2 = p[7]

    num_trials = rot.shape[0]

    delta = np.zeros(num_trials)

    theta_values = np.array(
        [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    theta_train_ind = np.where(theta_values == 0)[0][0]
    theta_ind = theta_train_ind * np.ones(num_trials, dtype=np.int8)

    x = np.zeros((11, num_trials))
    xs = np.zeros((11, num_trials))
    xf = np.zeros((11, num_trials))
    for i in range(0, num_trials - 1):
        if np.isnan(rot[i]):
            delta[i] = 0.0
        else:
            delta[i] = x[theta_ind[i], i] - rot[i]

        # Gs = g_func_cos(theta_values, theta_values[theta_ind[i]], g_sigma_s)
        Gs = g_func_gauss(theta_values, theta_values[theta_ind[i]], g_sigma_s)
        Gf = g_func_flat(g_sigma_f)

        if i < 306:
            if np.isnan(rot[i]):
                xs[:, i + 1] = (1 - beta_s) * xs[:, i]
                xf[:, i + 1] = (1 - beta_f) * xf[:, i]
            else:
                xs[:, i +
                   1] = (1 - beta_s) * xs[:, i] - alpha_s * delta[i] * Gs
                xf[:, i +
                   1] = (1 - beta_f) * xf[:, i] - alpha_f * delta[i] * Gf

        elif i > 306:
            # xs[:, i + 1] = (1 - beta_s_2) * xs[:, i]
            # xf[:, i + 1] = (1 - beta_f_2) * xf[:, i]
            xs[:, i + 1] = xs[:, i]
            xf[:, i + 1] = xf[:, i]

        elif i == 306:
            # xs[:, i + 1] = (1 - beta_s_2) * xs[:, i]
            # xf[:, i + 1] = (1 - beta_f_2) * xf[:, i]
            # xs[:, i + 1] = xs[:, i]
            # xf[:, i + 1] = xf[:, i]
            xs[:, i + 1] = xs[:, i] * beta_s_2
            xf[:, i + 1] = xf[:, i] * beta_f_2

        x[:, i + 1] = xs[:, i + 1] + xf[:, i + 1]

    return (x.T, xs.T, xf.T)


def fit_obj_func_sse(params, *args):
    x_obs = args[0]
    rot = args[1]
    x_pred = simulate_state_space_with_g_func_2_state(params, rot)[0]

    sse_rec = np.zeros(11)
    for i in range(11):
        sse_rec[i] = np.nansum(
            (x_obs[200:306, i] - x_pred[200:306, i])**2) + 2 * np.nansum(
                (x_obs[306:, i] - x_pred[306:, i])**2)
        sse = np.nansum(sse_rec)
    return sse


def prep_for_fits():
    d = pd.read_csv('../data/MIS_DATA_LONG_21122019.csv')
    # d = pd.read_csv('../data/MIS_DATA_LONG_NBC.csv')

    d.rename(columns={'Subject': 'subject'}, inplace=True)
    d.rename(columns={'cursor_error': 'error'}, inplace=True)
    d.sort_values(by=['group', 'rot_direction', 'subject', 'trial'],
                  inplace=True)

    d.loc[d.phase == 'Adaptation', 'trial'] += 198
    d.loc[d.phase == 'Generalisation', 'trial'] += 198 + 110

    dd = d.groupby(['group', 'subject', 'phase', 'trial',
                    'target'])['hand_angle'].mean().reset_index()

    dd = d.pivot_table(index=['group', 'subject', 'phase', 'trial'],
                       columns='target',
                       values='hand_angle').reset_index()

    # fig, ax = plt.subplots(nrows=2, ncols=10)
    # for j, g in enumerate(d.group.unique()):
    #     for k, s in enumerate(d.subject.unique()):
    #         for l in range(11):
    #             ddd = dd[(dd['group'] == g) & (dd['subject'] == s)]
    #             x = ddd['trial'].values
    #             y = ddd[ddd.columns[l + 4]].values
    #             ax[j, k].plot(x, y, '.')
    #             ax[j, k].set_xlim(306)
    # plt.show()

    dd = d.groupby(['rot_direction', 'group', 'phase', 'trial',
                    'target'])['hand_angle'].mean().reset_index()

    dd = d.pivot_table(index=['rot_direction', 'group', 'phase', 'trial'],
                       columns='target',
                       values='hand_angle').reset_index()

    # fig, ax = plt.subplots(nrows=2, ncols=2)
    # for i, r in enumerate(dd.rot_direction.unique()):
    #     for j, g in enumerate(dd.group.unique()):
    #         for k in range(11):
    #             x = dd[(dd['rot_direction'] == r)
    #                    & (dd['group'] == g)]['trial'].values
    #             y = dd[(dd['rot_direction'] == r)
    #                    & (dd['group'] == g)][dd.columns[k + 4]].values
    #             ax[i, j].plot(x, y)
    # plt.show()

    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # for i, r in enumerate(dd.rot_direction.unique()):
    #     for j, g in enumerate(dd.group.unique()):
    #         x = np.arange(0, 11, 1)
    #         y = np.nanmean(
    #             dd[(dd['phase'] == 'Generalisation')
    #                & (dd['rot_direction'] == r) &
    #                (dd['group'] == g)][dd.columns[4:]].values, 0)
    #         ax[i].plot(x, y)
    # plt.show()

    # NOTE: rotation column is still messed up, so just make some new ones here
    rot = np.concatenate(
        (np.zeros(198), 30 * np.ones(110), np.nan * np.ones(66)))

    d.loc[(d['phase'] == 'Adaptation') &
          (d['rot_direction'] == 'CW'), 'hand_angle'] = d.loc[
              (d['phase'] == 'Adaptation') &
              (d['rot_direction'] == 'CW'), 'hand_angle'] * -1

    return (d, rot)


def fit_state_space_with_g_func_2_state():

    d, rot = prep_for_fits()

    for i in d.group.unique():
        p_rec = np.empty((0, 8))

        x_obs = d.groupby(['group', 'target', 'trial']).mean()
        x_obs.reset_index(inplace=True)
        x_obs = x_obs[x_obs['group'] == i]
        x_obs = x_obs[['hand_angle', 'target', 'trial']]
        x_obs = x_obs.pivot(index='trial',
                            columns='target',
                            values='hand_angle')
        x_obs = x_obs.values

        # for k in range(11):
        #     plt.plot(x_obs[:, k])
        #     plt.plot(rot)
        # plt.show()

        results = fit(x_obs, rot)
        p = results["x"]
        print(p)

        # x_pred = simulate_state_space_with_g_func_2_state(p, rot)[0]
        # fig, ax = plt.subplots(nrows=1, ncols=2)
        # c = cm.rainbow(np.linspace(0, 1, 11))
        # for k in range(11):
        #     ax[0].plot(x_obs[:, k], '.', color=c[k])
        #     ax[0].plot(x_pred[:, k], '-', color=c[k])
        #     ax[0].plot(rot, 'k')

        # x = np.arange(0, 11, 1)
        # y_obs = np.nanmean(x_obs[-65:-1, :], 0)
        # y_pred = np.nanmean(x_pred[-65:-1, :], 0)
        # ax[1].plot(x, y_obs)
        # ax[1].plot(x, y_pred)

        # plt.show()

        p_rec = np.append(p_rec, [p], axis=0)

        f_name_p = '../fits/fit_group_' + str(i) + '.txt'
        with open(f_name_p, 'w') as f:
            np.savetxt(f, p_rec, '%0.4f', '\n')


def fit_state_space_with_g_func_2_state_boot():

    n_boot_samp = 4000

    d, rot = prep_for_fits()

    for i in d.group.unique():

        p_rec = -1 * np.ones((1, 8))
        for b in range(n_boot_samp):
            print(i, b)

            subs = d['subject'].unique()
            boot_subs = np.random.choice(subs,
                                         size=subs.shape[0],
                                         replace=True)

            x_boot_rec = []
            for k in boot_subs:
                x_boot_rec.append(d[d['subject'] == k])
                x_boot = pd.concat(x_boot_rec)

            x_obs = x_boot.groupby(['group', 'target', 'trial']).mean()
            x_obs.reset_index(inplace=True)
            x_obs = x_obs[x_obs['group'] == i]
            x_obs = x_obs[['hand_angle', 'target', 'trial']]
            x_obs = x_obs.pivot(index='trial',
                                columns='target',
                                values='hand_angle')
            x_obs = x_obs.values
            results = fit(x_obs, rot)
            p_rec[0, :] = results["x"]

            f_name_p = '../fits/fit_group_' + str(i) + '_boot.txt'
            with open(f_name_p, 'a') as f:
                np.savetxt(f, p_rec, '%0.4f', ',')


def fit(x_obs, rot):
    constraints = LinearConstraint(A=[[1, 0, 0, -1, 0, 0, 0, 0],
                                      [0, 1, 0, 0, -1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]],
                                   lb=[-1, -1, 0, 0, 0, 0, 0, 0],
                                   ub=[0, 0, 0, 0, 0, 0, 0, 0])

    args = (x_obs, rot)
    bounds = ((0, 1), (0, 1), (0, 150), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1))

    results = differential_evolution(
        func=fit_obj_func_sse,
        bounds=bounds,
        constraints=constraints,
        args=args,
        maxiter=300,
        # tol=1e-13,
        disp=False,
        polish=False,
        updating='deferred',
        workers=-1)
    return results


def inspect_results_boot():
    d, rot = prep_for_fits()

    x_obs = d.groupby(['group', 'target', 'trial']).mean()
    x_obs.reset_index(inplace=True)
    x_obs = x_obs[x_obs['group'] == 0]
    x_obs = x_obs[['hand_angle', 'target', 'trial']]
    x_obs = x_obs.pivot(index='trial', columns='target', values='hand_angle')
    x_obs_0 = x_obs.values

    x_obs = d.groupby(['group', 'target', 'trial']).mean()
    x_obs.reset_index(inplace=True)
    x_obs = x_obs[x_obs['group'] == 1]
    x_obs = x_obs[['hand_angle', 'target', 'trial']]
    x_obs = x_obs.pivot(index='trial', columns='target', values='hand_angle')
    x_obs_1 = x_obs.values

    p0 = np.loadtxt('../fits/fit_group_0.txt')
    p1 = np.loadtxt('../fits/fit_group_1.txt')

    x_pred_0 = simulate_state_space_with_g_func_2_state(p0, rot)[0]
    x_pred_1 = simulate_state_space_with_g_func_2_state(p1, rot)[0]

    x_pred_0_s = simulate_state_space_with_g_func_2_state(p0, rot)[1]
    x_pred_0_f = simulate_state_space_with_g_func_2_state(p0, rot)[2]

    x_pred_1_s = simulate_state_space_with_g_func_2_state(p1, rot)[1]
    x_pred_1_f = simulate_state_space_with_g_func_2_state(p1, rot)[2]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    c = cm.Set1(np.linspace(0, 1, 11))
    for k in range(11):
        pred_c = 'k'
        ax[0, 0].plot(x_obs_0[:, k], '.', color=c[k], alpha=0.1)
        ax[0, 0].plot(x_pred_0_s[:, 5], ':', color=c[5])
        ax[0, 0].plot(x_pred_0_f[:, 5], '--', color=c[5])
        ax[0, 0].plot(x_pred_0[:, 5], '-', color=c[5])
        ax[0, 0].plot(rot, 'k')
        ax[0, 0].set_title('Controls', size=16)
        ax[0, 0].set_ylabel('Hand Angle', size=16)
        ax[0, 0].set_xlabel('Trial', size=16)
        ax[0, 0].set_ylim(-10, 35)
        ax[0, 0].set_xlim(0, 400)
        ax[0, 0].set_xticks(np.arange(0, 450, 50))

        ax[1, 0].plot(x_obs_1[:, k], '.', color=c[k], alpha=0.1)
        ax[1, 0].plot(x_pred_1_s[:, 5], ':', color=c[5])
        ax[1, 0].plot(x_pred_1_f[:, 5], '--', color=c[5])
        ax[1, 0].plot(x_pred_1[:, 5], '-', color=c[5])
        ax[1, 0].plot(rot, 'k')
        ax[1, 0].set_title('Surgeons', size=16)
        ax[1, 0].set_ylabel('Hand Angle', size=16)
        ax[1, 0].set_xlabel('Trial', size=16)
        ax[1, 0].set_ylim(-10, 35)
        ax[1, 0].set_xlim(0, 400)
        ax[1, 0].set_xticks(np.arange(0, 450, 50))

    dd = d[['group', 'subject', 'target', 'trial',
            'hand_angle']].groupby(['group', 'subject', 'target',
                                    'trial']).mean()
    dd.reset_index(inplace=True)
    dd = dd[dd['trial'] > 374 - 68]

    ddd = dd.groupby(['group', 'target']).mean()
    ddd.reset_index(inplace=True)
    ddd_0 = ddd[ddd['group'] == 0][['target', 'hand_angle']]
    ddd_1 = ddd[ddd['group'] == 1][['target', 'hand_angle']]
    y_obs_0 = ddd_0.hand_angle.values
    y_obs_1 = ddd_1.hand_angle.values

    ddd = dd.groupby(['group', 'target']).std() / np.sqrt(10)
    ddd.reset_index(inplace=True)
    ddd_0 = ddd[ddd['group'] == 0][['target', 'hand_angle']]
    ddd_1 = ddd[ddd['group'] == 1][['target', 'hand_angle']]
    y_obs_err_0 = ddd_0.hand_angle.values
    y_obs_err_1 = ddd_1.hand_angle.values

    x = np.arange(0, 11, 1)
    # y_obs_0 = np.nanmean(x_obs_0[-68:-1, :], 0)
    y_pred_0 = np.nanmean(x_pred_0[-68:-1, :], 0)
    y_pred_0_s = np.nanmean(x_pred_0_s[-68:-1, :], 0)
    y_pred_0_f = np.nanmean(x_pred_0_f[-68:-1, :], 0)
    # ax[0, 1].plot(x, y_obs_0, '.-')
    ax[0, 1].errorbar(x, y_obs_0, yerr=y_obs_err_0)
    ax[0, 1].plot(x, y_pred_0, '.-')
    ax[0, 1].plot(x, y_pred_0_s, '.-', color='k', alpha=0.2)
    ax[0, 1].plot(x, y_pred_0_f, '--', color='k', alpha=0.2)
    ax[0, 1].set_ylim(-10, 35)
    ax[0, 1].set_title('Controls', size=16)
    ax[0, 1].set_ylabel('Hand Angle', size=16)
    ax[0, 1].set_xlabel('Relative Target Angle', size=16)
    ax[0, 1].set_xticks(x)
    ax[0, 1].set_xticklabels([
        '-150', '-120', '-90', '-60', '-30', '0', '30', '60', '90', '120',
        '150'
    ])

    # y_obs_1 = np.nanmean(x_obs_1[-68:-1, :], 0)
    y_pred_1 = np.nanmean(x_pred_1[-68:-1, :], 0)
    y_pred_1_s = np.nanmean(x_pred_1_s[-68:-1, :], 0)
    y_pred_1_f = np.nanmean(x_pred_1_f[-68:-1, :], 0)
    # ax[1, 1].plot(x, y_obs_1, '.-')
    ax[1, 1].errorbar(x, y_obs_1, yerr=y_obs_err_1)
    ax[1, 1].plot(x, y_pred_1, '.-')
    ax[1, 1].plot(x, y_pred_1_s, '.-', color='k', alpha=0.2)
    ax[1, 1].plot(x, y_pred_1_f, '--', color='k', alpha=0.2)
    ax[1, 1].set_ylim(-10, 35)
    ax[1, 1].set_title('Surgeons', size=16)
    ax[1, 1].set_ylabel('Hand Angle', size=16)
    ax[1, 1].set_xlabel('Relative Target Angle', size=16)
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels([
        '-150', '-120', '-90', '-60', '-30', '0', '30', '60', '90', '120',
        '150'
    ])

    ax[0, 0].text(-0.1,
                  1.1,
                  'A',
                  transform=ax[0, 0].transAxes,
                  size=20,
                  weight='bold',
                  va='top',
                  ha='right')
    ax[0, 1].text(-0.1,
                  1.1,
                  'B',
                  transform=ax[0, 1].transAxes,
                  size=20,
                  weight='bold',
                  va='top',
                  ha='right')
    ax[1, 0].text(-0.1,
                  1.1,
                  'C',
                  transform=ax[1, 0].transAxes,
                  size=20,
                  weight='bold',
                  va='top',
                  ha='right')
    ax[1, 1].text(-0.1,
                  1.1,
                  'D',
                  transform=ax[1, 1].transAxes,
                  size=20,
                  weight='bold',
                  va='top',
                  ha='right')

    plt.tight_layout()
    plt.savefig('../figures/fig_results.pdf')

    names = [
        'alpha_s', 'beta_s', 'g_s', 'alpha_f', 'beta_f', 'g_f', 'beta_s_2',
        'beta_f_2'
    ]

    d0 = pd.read_csv('../fits/fit_group_0_boot.txt', names=names)
    d1 = pd.read_csv('../fits/fit_group_1_boot.txt', names=names)

    d0['group'] = 0
    d1['group'] = 1

    d0['g_s'] = d0['g_s'] / 60.0
    d1['g_s'] = d1['g_s'] / 60.0

    d = pd.concat((d0, d1), sort=False)

    d0['beta_s'] = 1 - d0['beta_s']
    d0['beta_f'] = 1 - d0['beta_f']
    d1['beta_s'] = 1 - d1['beta_s']
    d1['beta_f'] = 1 - d1['beta_f']

    p0[1] = 1 - p0[1]
    p0[4] = 1 - p0[4]
    p1[1] = 1 - p1[1]
    p1[4] = 1 - p1[4]

    p0[2] = p0[2] / 60.0
    p1[2] = p1[2] / 60.0

    c = ['#1f77b4', '#ff7f0e']
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7]) * 2
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.violinplot(d0.values[:10000, :8],
                  positions=x - 0.25,
                  showextrema=False,
                  points=1000)
    ax.violinplot(d1.values[:10000, :8],
                  positions=x + 0.25,
                  showextrema=False,
                  points=1000)
    ax.boxplot(d0.values[:10000, :8],
               positions=x - 0.25,
               whis=[2.5, 97.5],
               showfliers=False,
               widths=0.4,
               labels=None,
               patch_artist=False)
    ax.boxplot(d1.values[:10000, :8],
               positions=x + 0.25,
               whis=[2.5, 97.5],
               showfliers=False,
               widths=0.4,
               labels=None,
               patch_artist=False)
    ax.plot(x-0.25, p0, '.', color=c[0])
    ax.plot(x+0.25, p1, '.', color=c[1])
    # ax.plot(np.reshape(npcolor=c[1].repeat(x - 0.25, 10000), (8, 10000)).T[0:10000:10],
    #         d0.values[:10000, :8][0:10000:10],
    #         '.',
    #         alpha=0.1,
    #         color=c[0])
    # ax.plot(np.reshape(np.repeat(x + 0.25, 10000), (8, 10000)).T[0:10000:10],
    #         d1.values[:10000, :8][0:10000:10],
    #         '.',
    #         alpha=0.1,
    #         color=c[1])
    ax.set_xticks(x)
    ax.set_xticklabels([
        r'$\boldsymbol{\alpha_{s}}$', r'$\boldsymbol{\beta_{s}}$',
        r'$\boldsymbol{g_{s}}$', r'$\boldsymbol{\alpha_{f}}$',
        r'$\boldsymbol{\beta_{f}}$', r'$\boldsymbol{g_{f}}$',
        r'$\boldsymbol{\gamma_{s}}$', r'$\boldsymbol{\gamma_{f}}$'
    ], fontsize=16)
    ax.set_ylim(-.1, 1.1)
    ax.set_ylabel('Parameter Value', size=16)
    plt.tight_layout()
    plt.savefig('../figures/fig_params.pdf')


def inspect_boot_stats():
    names = [
        'alpha_s', 'beta_s', 'g_s', 'alpha_f', 'beta_f', 'g_f', 'beta_s_2',
        'beta_f_2'
    ]

    p0 = np.loadtxt('../fits/fit_group_0.txt')
    p1 = np.loadtxt('../fits/fit_group_1.txt')

    d0 = pd.read_csv('../fits/fit_group_0_boot.txt', names=names)
    d1 = pd.read_csv('../fits/fit_group_1_boot.txt', names=names)

    print(
        'alpha_s',
        str(
            bootstrap_t(p0[0], p1[0], d0.alpha_s.values, d1.alpha_s.values,
                        10000)))
    print(
        'beta_s',
        str(
            bootstrap_t(p0[1], p1[1], d0.beta_s.values, d1.beta_s.values,
                        10000)))
    print('g_s',
          str(bootstrap_t(p0[2], p1[2], d0.g_s.values, d1.g_s.values, 10000)))
    print(
        'alpha_f',
        str(
            bootstrap_t(p0[3], p1[3], d0.alpha_f.values, d1.alpha_f.values,
                        10000)))
    print(
        'beta_f',
        str(
            bootstrap_t(p0[4], p1[4], d0.beta_f.values, d1.beta_f.values,
                        10000)))
    print('g_f',
          str(bootstrap_t(p0[5], p1[5], d0.g_f.values, d1.g_f.values, 10000)))
    print(
        'beta_s_2',
        str(
            bootstrap_t(p0[6], p1[6], d0.beta_s_2.values, d1.beta_s_2.values,
                        10000)))
    print(
        'beta_f_2',
        str(
            bootstrap_t(p0[7], p1[7], d0.beta_f_2.values, d1.beta_f_2.values,
                        10000)))


# fit_state_space_with_g_func_2_state()
# fit_state_space_with_g_func_2_state_boot()
inspect_results_boot()
# inspect_boot_stats()



