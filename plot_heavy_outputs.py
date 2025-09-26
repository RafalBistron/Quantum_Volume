import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def __get_means_and_deviations(distributions_tab):
    distributions_array = np.array(distributions_tab)
    means = np.average(distributions_array,axis = 1)
    num = distributions_array.shape[1]
    deviations = np.sqrt(np.sum(np.abs(means.reshape(-1,1) - distributions_array)**2,axis = 1)/(num*(num - 1)) )
    return means, deviations


def __simple_exponent(x, a, b, c):
    return a*np.exp(-b*x) +c


def __two_exponents(x, a, b, c):
    return a*np.exp(-(3/4)*b*x)+(a/2)*np.exp(-b*x) + c


def show_heavy_outputs(heavy_outputs_tabs, T_tab, N):
    """Plot distribution of heavy output frequencies, from 'heavy_outputs_tabs' for 'N' qubit circuits with layers from 'T_tab', together with exponential fit."""
    fig,ax = plt.subplots(figsize = (8,6))
    color_tab = ['black','red','blue','orange']
    label_tab = ['perfect standard h_U', 'standard h_U','single parity h_U','double parity h_U']

    for i, hu_tab in enumerate(heavy_outputs_tabs):
        means, sigmas = __get_means_and_deviations(hu_tab)

        ax.errorbar(T_tab, means, yerr = sigmas, label = label_tab[i], c = color_tab[i],fmt = ".")
        v1 = ax.violinplot(hu_tab,T_tab)
        for part_name in ('cbars','cmins','cmaxes'):
            vp = v1[part_name]
            vp.set_edgecolor(color_tab[i])
        for pc in v1['bodies']:
            pc.set_color(color_tab[i])
            pc.set_edgecolor(color_tab[i])

        func = __simple_exponent if i < 3 else  __two_exponents
        p0 = [2,1,0.5]
        bounds =  ([0,0,0],[3,np.inf,1])
        fit_coef, fit_cov = curve_fit(func, T_tab, means, sigma=sigmas, p0=p0, maxfev=5000, bounds=bounds)

        ax.plot(T_tab, func(T_tab,*fit_coef),c = color_tab[i])
        ax.plot(T_tab,0*T_tab+2/3,c = 'black',linestyle = "-.")
        ax.fill_between(T_tab, func(T_tab,*(fit_coef + np.diag(fit_cov))),
                        func(T_tab,*(fit_coef - np.diag(fit_cov))), alpha=0.1,color = color_tab[i])

    plt.xlabel("T", fontsize=12)
    plt.title(f"Heavy outputs for QV circuits, N = {N}")
    plt.legend()
    plt.show()           