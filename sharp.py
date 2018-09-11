""" Has functions related to generating temperature stochastically
    Author: Alex Weech
"""
from typing import Sequence, Generator, List, Tuple, Dict

import numpy as np
from numpy import pi, cos, sin
import scipy.sparse as ssparse
import scipy.special as sfuncs
import scipy.optimize as sciroots
from scipy.optimize.optimize import OptimizeResult
import pandas as pd
from pandas import DatetimeIndex

RetTup = Tuple[np.ndarray, np.ndarray, DatetimeIndex]

TAU = 365

class SHArPGenerator:
    """ A stochastic weather generator that follows the equation
        `T_(k+1) = a*T_k + b_k + c_k * e_k`
        where a is the value of persistance; T is temperature; b is a climatology model composed
        of a mean value, periodic component, and a trend; c is the standard deviation of the
        model error; and e is a random normal distribution. Seperate models are constructed to
        describe wet days and dry days.

        Parameters
        ----------
        n_harmonics : Optional parameter that determines the number of harmonics used in b. Defaults to 2.

        flag : Optional parameter that determines whether to use a seasonally varying c (True) or a constant c (False). Defaults to True.

    """

    def __init__(self, n_harmonics: int = 2, flag: bool = True) -> None:
        self.n_harmonics = n_harmonics
        self.flag = flag
        self.c: float
        self.a: float
        self.dry_climo: np.ndarray
        self.wet_climo: np.ndarray
        self.c_dry_coeffs: np.ndarray
        self.c_wet_coeffs: np.ndarray
        self.init_day: int
        self.m: int
        self.precip_coeffs: Dict[str, np.ndarray]

    def _generate_b(self, pobo: np.ndarray, ks: np.ndarray) -> np.ndarray:
        """ Generate the b arrays """
        harmons = list()
        pobo_n = 1-pobo
        for n in range(self.n_harmonics+1):
            harmons.append(pobo_n * cos(n*2*pi*ks/TAU))
            harmons.append(pobo * cos(n*2*pi*ks/TAU))
            if n != 0:
                harmons.append(pobo_n * sin(n*2*pi*ks/TAU))
                harmons.append(pobo * sin(n*2*pi*ks/TAU))
        arr = np.array(harmons).T
        return np.concatenate([ks[:, np.newaxis], arr], axis=1)

    def _generate_kernel(self, pobo: np.ndarray, ks: np.ndarray,
                         t: np.ndarray) -> np.ndarray:
        """ Generates the common kernel a*T_k + b_k """
        term1 = t[:, np.newaxis]
        term2 = self._generate_b(pobo, ks)
        return np.concatenate([term1, term2], axis=1)

    def _clean(self, t: np.ndarray, p: np.ndarray, dates: DatetimeIndex) -> RetTup:
        """ Clean up the input for bad precip data and missing times """

        # Clean precip
        if ~np.all((p == 0) | (p == 1)):
            raise ValueError("Precipitation must be series of 0 and 1")

        # Clean time
        idx = pd.date_range(dates[0], dates[-1])
        if np.all(idx.values == dates.values):
            return t, p, dates
        data = {'t':t, 'p':p}
        temp_df = pd.DataFrame(data=data, index=dates)
        temp_df = temp_df.reindex(idx)
        return temp_df['t'].values, temp_df['p'].values, temp_df.index

    def _generate_bk(self, pobo: np.ndarray, ks: np.ndarray) -> np.ndarray:
        pobo_n = 1-pobo
        total_harm = self.dry_climo[1] * pobo_n + self.wet_climo[1] * pobo
        dry_coeffs = self.dry_climo[2:]
        wet_coeffs = self.wet_climo[2:]
        for n in range(0, len(dry_coeffs), 2):
            total_harm += dry_coeffs[n] * pobo_n * cos((n+1)*2*pi*ks/TAU)
            total_harm += dry_coeffs[n+1] * pobo_n * sin((n+1)*2*pi*ks/TAU)
        for n in range(0, len(wet_coeffs), 2):
            total_harm += wet_coeffs[n] * pobo * cos((n+1)*2*pi*ks/TAU)
            total_harm += wet_coeffs[n+1] * pobo * sin((n+1)*2*pi*ks/TAU)
        return ks*self.dry_climo[0] + total_harm

    def _generate_cj(self) -> np.ndarray:
        # Compute a dry and wet c for each day of the year
        doys = np.arange(TAU)
        total = np.zeros_like(doys, dtype=float)
        for n in range(0, len(self.c_dry_coeffs), 2):
            total += self.c_dry_coeffs[n] * cos(n*2*pi*doys/TAU)
            total += self.c_dry_coeffs[n+1] * sin(n*2*pi*doys/TAU)
        c_dry = np.sqrt(total)
        total = np.zeros_like(doys, dtype=float)
        for n in range(0, len(self.c_wet_coeffs), 2):
            total += self.c_wet_coeffs[n] * cos(n*2*pi*doys/TAU)
            total += self.c_wet_coeffs[n+1] * sin(n*2*pi*doys/TAU)
        c_wet = np.sqrt(total)

        return c_dry, c_wet

    def _fit_precip_int(self, p: np.ndarray, start_doy: int, pattern: Sequence[int],
                        m: int) -> OptimizeResult:

        doys = np.arange(TAU) + 1

        # Calculate n2 and n3
        nyears = np.zeros(TAU, dtype=int)
        n3 = np.zeros(TAU)
        n2 = np.zeros(TAU)
        for i in range(2, len(p)):
            idx = (i + start_doy) % TAU
            n3[idx] += p[i] == pattern[2] and p[i-1] == pattern[1] and p[i-2] == pattern[0]
            n2[idx] += p[i-1] == pattern[1] and p[i-2] == pattern[0]
            nyears[idx] += 1

        # Calculate logits in bins
        prob3 = n3 / nyears
        prob_binned = np.array([np.nanmean(ls) for ls in np.array_split(prob3, 26)])
        doys_binned = np.array([np.mean(ls) for ls in np.array_split(doys, 26)])
        logit3 = sfuncs.logit(prob_binned)

        # Solve for estimations using least squares
        cols = list()
        phi = np.vectorize(lambda i, t: cos((i-1)*pi*t/TAU) if i%2 == 1 else sin(i*pi*t/TAU))
        for i in range(1, m+1):
            cols.append(phi(i, doys_binned))

        matrix = np.column_stack(cols)
        est_coeffs = np.linalg.lstsq(matrix, logit3, rcond=None)[0]

        # Solve for all data using hybr
        def maxloglikefunc(a: np.ndarray) -> List[float]:
            rows = list()
            ip = np.arange(1, m+1)
            logit = np.nansum(a * phi(ip[:, np.newaxis], doys).T, axis=1)
            for i in range(1, m+1):
                rows.append(np.nansum(phi(i, doys) * (n3 - n2*sfuncs.expit(logit))))
            return rows

        def maxloglikejac(a: np.ndarray) -> List[List[float]]:
            rows: List[List[float]] = list()
            ip = np.arange(1, m+1)
            logit = np.nansum(a * phi(ip[:, np.newaxis], doys).T, axis=1)
            for i in range(1, m+1):
                rows.append(list())
                for j in range(1, m+1):
                    item = phi(i, doys)*phi(j, doys) * -n2*sfuncs.expit(logit)/(1+np.exp(logit))
                    rows[i-1].append(np.nansum(item))
            return rows

        return sciroots.root(maxloglikefunc, est_coeffs, jac=maxloglikejac)

    def fit_precip(self, p: np.ndarray, dates: DatetimeIndex, m: int):
        """ Fit SHArP Precipitation Model

            Parameters
            ----------
            p : Series of 1s and 0s where 1 indicates a wet day and 0 indicates a dry day

            dates : Day of year that corresponds to the zeroth element in p. Should be between 0 and 365.

            m : Number of coefficients to include in harmonics. Must be odd and should be less than 365.

            Returns
            -------
            self : An instance of self

            Raises
            ------
            ValueError
                * If p is not an array of 1s and 0s
                * If m is even
                * If the model could not be fit
        """
        if m % 2 != 1:
            raise ValueError('m must be odd (Woolhiser 2008)')
        if ~np.all((p == 0) | (p == 1)):
            raise ValueError("Precipitation must be series of 0 and 1")
        idx = pd.date_range(dates[0], dates[-1])
        if np.all(idx.values != dates.values):
            data = {'p':p}
            temp_df = pd.DataFrame(data=data, index=dates)
            temp_df = temp_df.reindex(idx)
            p = temp_df['p'].values

        patterns = {'p000':(0, 0, 0), 'p010':(0, 1, 0), 'p110':(1, 1, 0), 'p100':(1, 0, 0)}
        probs = dict()
        for name, pattern in patterns.items():
            status = self._fit_precip_int(p, dates[0].dayofyear-1, pattern, m)
            if not status.success:
                raise ValueError(f'Could not fit the model. Message: {status.message}')
            probs[name] = status.x
        self.precip_coeffs = probs
        self.m = m
        return self

    def generate_precip(self, n_days: int, start_doy: int = 0,
                        init_p: Sequence[int] = None) -> Generator[int, None, None]:
        """ Generate a precipitation sequence from SHArP

            Parameters
            ----------
            n_days : How many days to generate

            start_doy : A day of year between 0 and 365 to start on. Defaults to 0

            init_p : An initial precipitation pattern to seed generation with. Defaults to random values.

            Returns
            -------
            A generator that produces 0 for dry days and 1 for wet days

            Raises
            ------
            ValueError
                * If the model has not been fit
        """
        # Throw if not fit
        if self.precip_coeffs is None:
            raise ValueError('Precipitation model has not been fit')

        # Create the probabilties for each day
        probs = dict()
        doys = np.arange(TAU) + 1
        phi = np.vectorize(lambda i, t: cos((i-1)*pi*t/TAU) if i%2 == 1 else sin(i*pi*t/TAU))
        ip = np.arange(self.m)
        for name, coeffs in self.precip_coeffs.items():
            patt = np.sum(coeffs * phi(ip[:, np.newaxis], doys).T, axis=1)
            probs[name] = sfuncs.expit(patt)

        # Create a seed
        seed: List[int] = []
        if init_p is not None:
            seed.extend(init_p)
        else:
            seed.extend(np.random.random_integers(0, high=1, size=2))

        # Generate up to n_days
        i, j = seed[-2], seed[-1]
        for n in range(n_days):
            curr_doy = (start_doy + n) % TAU
            if i == 0 and j == 0:
                prob = probs['p000'][curr_doy]
            elif i == 0 and j == 1:
                prob = probs['p010'][curr_doy]
            elif i == 1 and j == 1:
                prob = probs['p110'][curr_doy]
            else:
                prob = probs['p100'][curr_doy]
            i = j
            if prob < np.random.ranf():
                j = 1
                yield 1
            else:
                j = 0
                yield 0


    def fit_temperature(self, t_in: np.ndarray, p: np.ndarray, dates: DatetimeIndex):
        """ Fit SHArP Temperature Model

            Parameters
            ----------
            t_in : Temperature series

            p : Series of 1s and 0s where 1 indicates a wet day and 0 indicates a dry day

            dates : Dates that correspond to items in t_in and p

            Returns
            -------
            self : An instance of self

            Raises
            ------
            ValueError
                If p is not an array of 1s and 0s
        """
        # Clean the inputs
        t_in, p, dates = self._clean(t_in, p, dates)
        self.init_day = dates[0]

        # Calculate the bk coefficients
        # Make lists we'll need
        t_nf = t_in[1:]
        t_nb = t_in[:-1]
        dates_nf = dates[1:]
        k = t_nf.shape[0]
        pobo = p[1:]
        pobo_n = 1 - pobo
        tobo = t_in[1:]
        ks = np.arange(k)
        ks_arr = ks[:, np.newaxis]
        kernel = self._generate_kernel(pobo, ks, t_nb)

        # Generate matrix
        row0 = np.nansum(t_nb[:, np.newaxis]*kernel, axis=0)
        sol0 = np.nansum(t_nb*tobo)
        row1 = np.nansum(ks_arr*kernel, axis=0)
        sol1 = np.nansum(ks*tobo)
        base_harm_rows = (np.nansum(pobo_n[:, np.newaxis]*kernel, axis=0),
                          np.nansum(pobo[:, np.newaxis]*kernel, axis=0))
        base_harm_sols = (np.nansum(pobo_n * tobo), np.nansum(pobo * tobo))
        rows = [row0, row1, *base_harm_rows]
        sols = [sol0, sol1, *base_harm_sols]
        for n in range(1, self.n_harmonics+1):
            rows.append(np.nansum((pobo_n * cos(n*2*pi*ks/TAU))[:, np.newaxis] * kernel, axis=0))
            rows.append(np.nansum((pobo * cos(n*2*pi*ks/TAU))[:, np.newaxis] * kernel, axis=0))
            rows.append(np.nansum((pobo_n * sin(n*2*pi*ks/TAU))[:, np.newaxis] * kernel, axis=0))
            rows.append(np.nansum((pobo * sin(n*2*pi*ks/TAU))[:, np.newaxis] * kernel, axis=0))

            sols.append(np.nansum(pobo_n * cos(n*2*pi*ks/TAU) * tobo))
            sols.append(np.nansum(pobo * cos(n*2*pi*ks/TAU) * tobo))
            sols.append(np.nansum(pobo_n * sin(n*2*pi*ks/TAU) * tobo))
            sols.append(np.nansum(pobo * sin(n*2*pi*ks/TAU) * tobo))

        # Solve matrix and extract coefficients
        matrix = np.array(rows)
        solutions = np.array(sols)
        coeffs = np.linalg.solve(matrix, solutions)
        a = coeffs[0]
        self.a = a
        alpha = coeffs[1]
        bases = coeffs[2:4]
        harm_coeffs = coeffs[4:]
        wet_coeffs = harm_coeffs[1::2]
        dry_coeffs = harm_coeffs[::2]
        self.dry_climo = [alpha, bases[0], *dry_coeffs]
        self.wet_climo = [alpha, bases[1], *wet_coeffs]

        # Generate D, B, and DT - B
        B = self._generate_bk(pobo, ks)
        B[0] += a*t_in[0]

        center = np.ones(k)
        off = np.zeros(k-1) - a
        D = ssparse.diags([center, off], [0, -1], format='csc')

        dtb = (D @ t_nf - B)

        # Generate ck
        if not self.flag:
            # No seasonal variation
            self.c = np.sqrt(np.nansum(dtb**2)/k)
        else:
            # With seasonal variation

            # Group by doy and compute variance
            dtb_df = pd.Series(data=dtb, index=dates_nf, name='dtb')
            pobo_bool = pobo.astype(bool)
            dry_dtb = dtb_df.where(~pobo_bool)
            wet_dtb = dtb_df.where(pobo_bool)
            est_func = lambda x: np.nansum(x**2) / x.size
            dry_ests = dry_dtb.groupby(dry_dtb.index.dayofyear).agg(est_func)
            wet_ests = wet_dtb.groupby(wet_dtb.index.dayofyear).agg(est_func)
            if dry_ests.shape[0] == 366:
                dry_ests = dry_ests[:-1]
            if wet_ests.shape[0] == 366:
                wet_ests = wet_ests[:-1]
            dry_ests = dry_ests.where(~np.isnan(dry_ests), 0)
            wet_ests = wet_ests.where(~np.isnan(wet_ests), 0)

            for i in range(1, len(dry_ests)):
                if dry_ests.index[i] - dry_ests.index[i-1] != 1:
                    print(dry_ests.index[i], dry_ests.index[i-1])

            # Compute coefficients
            doys = np.arange(TAU)
            rho_dry = np.nansum(dry_ests) / TAU
            c_dry_coeffs = [rho_dry, 0]
            for n in range(1, self.n_harmonics+1):
                c_dry_coeffs.append(2/TAU * np.nansum(dry_ests * cos(2*n*pi*doys/TAU)))
                c_dry_coeffs.append(2/TAU * np.nansum(dry_ests * sin(2*n*pi*doys/TAU)))
            self.c_dry_coeffs = c_dry_coeffs

            rho_wet = np.nansum(wet_ests) / TAU
            c_wet_coeffs = [rho_wet, 0]
            for n in range(1, self.n_harmonics+1):
                c_wet_coeffs.append(2/TAU * np.nansum(wet_ests * cos(2*n*pi*doys/TAU)))
                c_wet_coeffs.append(2/TAU * np.nansum(wet_ests * sin(2*n*pi*doys/TAU)))
            self.c_wet_coeffs = c_wet_coeffs
        return self

    def generate_temp(self, t0: float, p: np.ndarray,
                      dates: DatetimeIndex) -> np.ndarray:
        """ Generate a temperature sequence from SHArP

            Parameters
            ----------
            t0 : A seed temperature in the same units the model was fit with

            p : Series of 1s and 0s where 1 indicates a wet day and 0 indicates a dry day

            dates : Dates that correspond to items in p

            Returns
            -------
            t_out : A stochastic temperature sequence

            Raises
            ------
            ValueError
                * If the model has not been fit
                * If p contains NaNs
                * If dates are not each 1 day apart
        """
        if not self.dry_climo:
            raise ValueError('The model has not been fit')
        if np.any(np.isnan(p)):
            raise ValueError('p cannot contain NaNs')
        if ~np.all((dates[1:] - dates[:-1]).days == 1):
            raise ValueError('Dates must all be 1 day apart')

        # Compute time series
        k = len(dates)-1
        e = np.random.randn(k)
        t_out = np.empty(k+1)
        t_out[0] = t0
        pobo = p[1:]
        pobo_n = 1 - pobo
        ks = np.arange(k)
        date_diff = (dates[0] - self.init_day).days
        B = self._generate_bk(pobo, ks+date_diff)
        if not self.flag:
            for k in ks:
                t_out[k+1] = self.a*t_out[k] + B[k] + self.c*e[k]
        else:
            c_dry, c_wet = self._generate_cj()
            real_doys = dates.dayofyear.values
            real_doys[real_doys == 366] = 365
            for k in ks:
                t_out[k+1] = (self.a*t_out[k] + B[k] + c_wet[real_doys[k]-1]*e[k]*pobo[k]
                                                     + c_dry[real_doys[k]-1]*e[k]*pobo_n[k])

        return t_out

    def __repr__(self):
        return (f'{{dry_climo:{self.dry_climo},\nwet_climo:{self.wet_climo},\nc:{self.c}, '
                + f'a:{self.a}, precip_coeffs:{self.precip_coeffs},\n'
                + f'n_harmonics:{self.n_harmonics}, flag:{self.flag}, init_day:{self.init_day},\n'
                + f'c_dry_coeffs:{self.c_dry_coeffs},\nc_wet_coeffs:{self.c_wet_coeffs}}}')
