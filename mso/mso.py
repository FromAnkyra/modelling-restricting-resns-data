import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import NymphESN.nymphesn as nymph
import NymphESN.errorfuncs as errorfunc
import NymphESN.restrictedmatrix as rmatrix
import NymphESN.vis as vis
from typing import Tuple 

xs_0 = np.linspace(0, 50, num=25)
xs = np.linspace(0, 10, num=10)
xs_minus_1 = np.linspace(0, 50, num=500)
xs_minus_2 = np.linspace(0, 100, num=5000)

resolution_4 = np.linspace(0, 10, 80)

class MSO(Enum):
    one = 0.2
    two = 0.311
    three = 0.42
    four = 0.51
    five = 0.63
    six = 0.74
    seven = 0.85
    eight = 0.97

def generate_MSO(xs, params):
    f = lambda x : sum([np.sin(param*x) for param in params])
    ys = map(f, xs)
    return np.fromiter(ys, dtype=float)

def compare_plots(params, figname):
    fig, ax = plt.subplots()

    # ax.plot(xs_0, generate_MSO(xs_0, params), label="ticks=2")
    # ax.plot(xs, generate_MSO(xs, params), label="ticks=1")
    # ax.plot(xs_minus_1, generate_MSO(xs_minus_1, params), label="ticks=0.1")
    ax.plot(resolution_4, generate_MSO(resolution_4, params), 'b', linestyle='-', label="MSO")
    ax.plot(resolution_4, generate_MSO(resolution_4, params), '.g', label="sample points used")
    ax.plot(xs, generate_MSO(xs, params), '.m', label="y(t)")
    ax.legend()
    fig.savefig(figname)
    return 

# compare_plots([MSO.one.value], "mso/singles/single_sine_1.png")
# compare_plots([MSO.two.value], "mso/singles/single_sine_2.png")
# compare_plots([MSO.three.value], "mso/singles/single_sine_3.png")
# compare_plots([MSO.four.value], "mso/singles/single_sine_4.png")
# compare_plots([MSO.five.value], "mso/singles/single_sine_5.png")
# compare_plots([MSO.six.value], "mso/singles/single_sine_6.png")
# compare_plots([MSO.seven.value], "mso/singles/single_sine_7.png")
# compare_plots([MSO.eight.value], "mso/resolutions/singles/single_sine_8.png")

compare_plots([MSO.one.value, MSO.two.value], "/home/cw1647/phd/mso/simple_plots/sample_points_two.png")
# compare_plots([MSO.one.value, MSO.two.value, MSO.three.value], "/home/cw1647/phd/mso/simple_plots/three.png")
# compare_plots([MSO.one.value, MSO.two.value, MSO.three.value, MSO.four.value], "/home/cw1647/phd/mso/simple_plots/four.png")
# compare_plots([MSO.one.value, MSO.two.value, MSO.three.value, MSO.four.value, MSO.five.value, MSO.six.value, MSO.seven.value, MSO.eight.value], "/home/cw1647/phd/mso/simple_plots/eight.png")


compare_plots([MSO.one.value, MSO.two.value, MSO.three.value, MSO.four.value, MSO.five.value, MSO.six.value, MSO.seven.value, MSO.eight.value], "/home/cw1647/phd/mso/simple_plots/sample_points_eight.png")

# print("hey world")

# look at the training lengths

def run_MSO(esn: nymph.NymphESN, MSO, data_lengths: Tuple[int, int, int], error="nrmse"):
    offset = 1
    u = list(MSO)[:sum(data_lengths)] # only take the values up to your training lengths
    vtarget = np.asarray(u[1:] + [u[-1]] * offset) # replicate the last value
    esn.set_data_lengths(data_lengths[0], data_lengths[1], data_lengths[2])
    esn.set_input_stream(u)
    esn.run_full()
    esn.train_reservoir(vtarget[data_lengths[0] : data_lengths[0]+data_lengths[1]])
    # print(f"{esn.Wv=}")
    esn.get_output()
    if error=="nrmse":
        return esn.get_error(vtarget, errorfunc.ErrorFuncs.nrmse)
    elif error=="mse":
        return esn.get_error(vtarget, errorfunc.ErrorFuncs.mse)
    else: # both
        return esn.get_error(vtarget, errorfunc.ErrorFuncs.nrmse), esn.get_error(vtarget, errorfunc.ErrorFuncs.mse)
    

def run_MSO_rr(esn: nymph.NymphESN, MSO, data_lengths: Tuple[int, int, int], error="nrmse"):
    offset = 1
    u = list(MSO)[:sum(data_lengths)] # only take the values up to your training lengths
    vtarget = np.asarray(u[1:] + [u[-1]] * offset) # replicate the last value
    esn.set_data_lengths(data_lengths[0], data_lengths[1], data_lengths[2])
    esn.set_input_stream(u)
    esn.run_full()
    esn.train_ridge_regression(vtarget[data_lengths[0] : data_lengths[0]+data_lengths[1]])
    # print(f"{esn.Wv=}")
    esn.get_output()
    # print(f"{esn.vall=}")
    if error=="nrmse":
        return esn.get_error(vtarget, errorfunc.ErrorFuncs.nrmse)
    elif error=="mse":
        return esn.get_error(vtarget, errorfunc.ErrorFuncs.mse)
    else: # both
        return esn.get_error(vtarget, errorfunc.ErrorFuncs.nrmse), esn.get_error(vtarget, errorfunc.ErrorFuncs.mse)
# "optimal training length" function
# between say, 500 and 5000?
# get the mean of the nrmse over 50 runs
# graph the means over the different lengths
# also return the value where the differential is below a certain value (need to play around to know which)

