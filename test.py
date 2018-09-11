""" Feeds the stochastic weather generator for debugging purposes """
import datetime as dt

import numpy as np
from numpy import pi, cos, sin
import pandas as pd
import matplotlib.pyplot as plt

import sharp

def main():
    # Read in the CSV
    data = pd.read_csv('testdata_large.csv', index_col=2, parse_dates=True)
    t_in = data['TAVG']
    t_in = t_in.where(~np.isnan(t_in), (data['TMAX'] + data['TMIN'])/2)
    precip = data['PRCP']
    precip = precip.where(precip >= 0.254, 0).where(precip < 0.254, 1)

    # Clean up precip and dates
    df2 = data.reindex(pd.date_range(data.index[0], data.index[-1]))
    new_precip = df2['PRCP']
    new_precip = new_precip.where(precip >= 0.254, 0).where(precip < 0.254, 1)
    new_precip = new_precip.where(~np.isnan(df2['PRCP']), 0.5)

    # Run the generator
    outs = list()
    count = 100
    year = 365
    start = dt.datetime.now()
    model = sharp.SHArPGenerator(n_harmonics=4)
    model.fit_temperature(t_in.values, precip.values, data.index)
    model.fit_precip(precip.values, data.index, 5)
    for _ in range(count):
        #precip = np.fromiter(model.generate_precip(365, start_doy=df2.index[-365].dayofyear-1), int)
        outs.append(model.generate_temp(t_in.values[-365], new_precip[-365:], df2.index[-365:]))
    end = dt.datetime.now()
    print(end - start)

    # Plot the results
    _, ax = plt.subplots()
    for line in outs:
        ax.plot(df2.index[-365:], line, color='b', alpha=0.05)
    ax.plot(data.index[-365:], t_in[-365:], 'r', label='Observed')
    ax.set_title(f'{count} Temperature Simulations Generated with SHArP Prototype')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (℃)')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
