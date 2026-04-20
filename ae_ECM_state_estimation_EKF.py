
#A:E ECM STATE ESTIMATION EKF
#model:


#TODO: CHeck your matrices are correct for RC!!
#TODO: Check sign conventions!!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator


#Power Limit Design Limits
VMAX = 4.2
VMIN = 2.5
IMIN = -100
IMAX = 100
SOCMIN = 0.1
SOCMAX = 0.95
PMIN = -np.inf 
PMAX = np.inf 
P_DT = 10 #seconds. How long we want to be able to sustain the power limit for. 
SIM_DT = 1 #seconds. Timestep to use for simulating cell response when calculating power limits. Smaller = more accurate but more computation.


def initEKF(SOC0, SigmaX0, SigmaV, SigmaW, model_params, model_props, SOC_offset=0.0):
    #We assume the cell is at rest initially with neutral hysteresis state 
    
    # Initial state description
    ir1_0 = 0 # Initial current through R1. We assume the cell is at rest, so this is zero.
    ir2_0 = 0 # Initial current through R2. We assume the cell is at rest, so this is zero.
    ekfData = {}
    ekfData['ir1Ind'] = 0
    ekfData['ir2Ind'] = 1

    hk0 = 0
    ekfData['hkInd'] = 2

    #NOTE(CP): In a real implementation initial SOC0 would either have to be based on prior knowledge or 
    # get calculated from the first voltage measurement using the OCV curve (Assuming that that first voltage measuerement is the fully relaxed cell voltage (OCV))
    #The following function would get implemented. 
    #SOC0 = SOCfromOCVtemp(v0, T0, model) 
    SOC0 = SOC0-SOC_offset #NOTE(CP): Test an offset in initial SOC

    ekfData['zkInd'] = 3

    ekfData['xhat'] = np.array([ir1_0, ir2_0, hk0, SOC0]).reshape(-1, 1)  # initial state

    # Covariance values
    ekfData['SigmaX'] = SigmaX0
    ekfData['SigmaV'] = SigmaV
    ekfData['SigmaW'] = SigmaW
    ekfData['Qbump'] = 5 #How much we want to increase the uncertainty when we get a really bad measurement (several in a row)

    # previous value of current
    ekfData['priorI'] = 0
    ekfData['signIk'] = 0

    # store model data structure too
    ekfData['model_params'] = model_params
    ekfData['model_props'] = model_props

    return ekfData


def getPropECM(propName, model_props):
    return model_props[propName].iloc[0]


def getParamECM(paramName, T, z, model_params):
    Ts = np.sort(model_params["T_degC"].unique())
    T = np.clip(T, Ts[0], Ts[-1])

    vals = []
    for T0 in Ts:
        sub = model_params.loc[model_params["T_degC"] == T0, ["SOC", paramName]].sort_values("SOC")
        soc = sub["SOC"].to_numpy()
        par = sub[paramName].to_numpy()
        z0 = np.clip(z, soc[0], soc[-1])
        vals.append(np.interp(z0, soc, par))

    return float(np.interp(T, Ts, vals))


def dOCVfromSOCtemp(z, T, model_params, OCVparam):
    Ts = np.sort(model_params["T_degC"].unique())
    T = np.clip(T, Ts[0], Ts[-1])

    vals = []
    for T0 in Ts:
        sub = model_params.loc[
            model_params["T_degC"] == T0, ["SOC", OCVparam]
        ].sort_values("SOC")

        soc = sub["SOC"].to_numpy()
        ocv = sub[OCVparam].to_numpy()

        # Remove duplicate SOC values if present
        soc, idx = np.unique(soc, return_index=True)
        ocv = ocv[idx]

        # Gradient dOCV/dSOC for this temperature slice
        docvdz = np.gradient(ocv, soc)

        # Hold nearest gradient outside SOC range
        z0 = np.clip(z, soc[0], soc[-1])
        vals.append(np.interp(z0, soc, docvdz))

    # Interpolate gradient across temperature
    return float(np.interp(T, Ts, vals))


def iterEKF(vk, ik, Tk, deltat, ekfData):
    model_params = ekfData['model_params']
    model_props = ekfData['model_props']

    #Current state of charge estimate for looking up model parameters
    zk = ekfData['xhat'][ekfData['zkInd'], 0] 
    hk = ekfData['xhat'][ekfData['hkInd'], 0]
   
    # Load the cell model parameters
    Q = getPropECM('Qnom_Ah', model_props)
    R1 = getParamECM('R_R1_Ohm', Tk, zk, model_params)
    C1 = getParamECM('C_C1_F', Tk, zk, model_params)
    R2 = getParamECM('R_R2_Ohm', Tk, zk, model_params)
    C2 = getParamECM('C_C2_F', Tk, zk, model_params)
    R0 = getParamECM('R_R0_Ohm', Tk, zk, model_params)
    G = getParamECM('gamma', Tk, zk, model_params)

    R1C1 = np.exp(-deltat / (R1 * C1))
    R2C2 = np.exp(-deltat / (R2 * C2))

    # Get data stored in ekfData structure
    I = ekfData['priorI']
    SigmaX = ekfData['SigmaX']
    SigmaV = ekfData['SigmaV']
    SigmaW = ekfData['SigmaW']
    xhat = ekfData['xhat']
    ir1Ind = ekfData['ir1Ind']
    ir2Ind = ekfData['ir2Ind']
    hkInd = ekfData['hkInd']
    zkInd = ekfData['zkInd']

    #NOTE(CP): As long as the current is not negligible (less than C/100), we will use its sign. 
    # Otherwise, we will just assume the current is zero and not use its sign, to avoid noise causing sign flips.
    if abs(ik) > Q / 100: 
        ekfData['signIk'] = np.sign(ik)
    signIk = ekfData['signIk']

    # EKF Step 0: Compute Ahat[k-1], Bhat[k-1]
    nx = len(xhat)
    Ahat = np.zeros((nx, nx))
    Bhat = np.zeros((nx, 1))

    Ahat[zkInd, zkInd] = 1
    Bhat[zkInd, 0] = deltat / (3600 * Q)

    Ahat[ir1Ind, ir1Ind] = R1C1
    Bhat[ir1Ind, 0] = (1 - R1C1)

    Ahat[ir2Ind, ir2Ind] = R2C2
    Bhat[ir2Ind, 0] = (1 - R2C2)

    Ah = np.exp(-abs(I * G * deltat / (3600 * Q)))  # hysteresis factor
    Ahat[hkInd, hkInd] = Ah
    B = np.hstack((Bhat, 0 * Bhat))
    Bhat[hkInd, 0] = -abs(G * deltat / (3600 * Q)) * Ah * (1 - signIk * xhat[hkInd, 0])
    B[hkInd, 1] = 1 - Ah

    # Step 1a: State estimate time update
    xhat = Ahat @ xhat + B @ np.array([[I], [signIk]])

    zk = xhat[zkInd, 0]
    hk = xhat[hkInd, 0]

    # Load the cell model parameters
    Q = getPropECM('Qnom_Ah', model_props)
    R1 = getParamECM('R_R1_Ohm', Tk, zk, model_params)
    C1 = getParamECM('C_C1_F', Tk, zk, model_params)
    R2 = getParamECM('R_R2_Ohm', Tk, zk, model_params)
    C2 = getParamECM('C_C2_F', Tk, zk, model_params)
    R0 = getParamECM('R_R0_Ohm', Tk, zk, model_params)
    G = getParamECM('gamma', Tk, zk, model_params)
   
    R1C1 = np.exp(-deltat / (R1 * C1))
    R2C2 = np.exp(-deltat / (R2 * C2))

    # Step 1b: Error covariance time update
    # sigmaminus(k) = Ahat(k-1)*sigmaplus(k-1)*Ahat(k-1)' + ...
    # Bhat(k-1)*sigmawtilde*Bhat(k-1)'
    SigmaX = Ahat @ SigmaX @ Ahat.T + Bhat @ np.atleast_2d(SigmaW) @ Bhat.T if np.ndim(SigmaW) > 0 else Ahat @ SigmaX @ Ahat.T + Bhat * SigmaW * Bhat.T

    # Step 1c: Output estimate
    OCVdch = getParamECM('E_OCV_dch_V', Tk, zk, model_params)
    OCVch = getParamECM('E_OCV_ch_V', Tk, zk, model_params)
    OCV = ((1+hk)/2)*OCVch + ((1-hk)/2)*OCVdch
    yhat = (
        OCV
        + float(R1) * xhat[ir1Ind, 0]
        + float(R2) * xhat[ir2Ind, 0]
        + R0 * ik
    )

    # Step 2a: Estimator gain matrix
    Chat = np.zeros((1, nx))
    Chat[0, zkInd] = dOCVfromSOCtemp(zk, Tk, model_params, 'E_OCV_ch_V') * (1+hk)/2 + dOCVfromSOCtemp(zk, Tk, model_params, 'E_OCV_dch_V') * (1-hk)/2
    Chat[0, hkInd] = 0.5 * (OCVch - OCVdch)
    Chat[0, ir1Ind] = float(R1)
    Chat[0, ir2Ind] = float(R2)

    Dhat = 1
    SigmaY = Chat @ SigmaX @ Chat.T + Dhat * SigmaV * Dhat
    L = SigmaX @ Chat.T / SigmaY

    # Step 2b: State estimate measurement update
    r = vk - yhat  # residual. Use to check for sensor errors...
    if r**2 > 100 * SigmaY:
        L[:] = 0.0
    xhat = xhat + L * r
    xhat[hkInd, 0] = min(1, max(-1, xhat[hkInd, 0]))  # Help maintain robustness
    xhat[zkInd, 0] = min(1.05, max(-0.05, xhat[zkInd, 0]))

    # Step 2c: Error covariance measurement update (Joseph form)
    I4 = np.eye(nx)
    SigmaX = (I4 - L @ Chat) @ SigmaX @ (I4 - L @ Chat).T + L * SigmaV * L.T

    # % Q-bump code
    if r**2 > 4 * SigmaY:  # bad voltage estimate by 2 std. devs, bump Q
        print('Bumping SigmaX')
        SigmaX[zkInd, zkInd] = SigmaX[zkInd, zkInd] * ekfData['Qbump']

    _, S, Vt = np.linalg.svd(SigmaX)
    V = Vt.T
    HH = V @ np.diag(S) @ V.T
    SigmaX = (SigmaX + SigmaX.T + HH + HH.T) / 4  # Help maintain robustness

    # Save data in ekfData structure for next time...
    ekfData['priorI'] = ik
    ekfData['SigmaX'] = SigmaX
    ekfData['xhat'] = xhat
    ekfData['yhat'] = yhat
    ekfData['hk'] = hk
    ekfData['Q']  = Q
    ekfData['R1'] = R1
    ekfData['C1'] = C1
    ekfData['R2'] = R2
    ekfData['C2'] = C2
    ekfData['R0'] = R0
    ekfData['G']  = G

    zk = xhat[zkInd, 0]
    zkbnd = 3 * np.sqrt(SigmaX[zkInd, zkInd])

    return zk, zkbnd, ekfData


# Power limit calculation functions
def simCellDT(ik, ekfData, Tk, sim_dt=SIM_DT, DT=P_DT):
    """Simulate the cell at constant current ik for DT seconds (closed-form solution).

    Args:
        ik:      Constant current (A). Positive = charge, negative = discharge.
        ekfData: EKF state dict — provides current state estimate and cached ECM params.
        Tk:      Temperature (degC), used for OCV table lookup at the final SOC.
        sim_dt:  Simulation timestep (s). Smaller = more accurate decay factors.
        DT:      Total simulation horizon (s).

    Internally computes KDT = ceil(DT / sim_dt) steps of size sim_dt.
    The actual simulated time is sim_dt * KDT (>= DT due to ceiling rounding).

    State vector matches the EKF: x = [ir1, ir2, hk, SOC] (indices 0,1,2,3).
    ECM parameters are read from ekfData — no redundant model lookups.
    Voltage uses the same OCV / hysteresis model as iterEKF.
    """
    ir1Ind, ir2Ind, hkInd, zkInd = 0, 1, 2, 3

    KDT = int(np.ceil(DT / sim_dt))  # number of sim steps; actual duration = sim_dt * KDT

    x0 = ekfData['xhat'].flatten()
    Q  = ekfData['Q']
    R1 = ekfData['R1']
    C1 = ekfData['C1']
    R2 = ekfData['R2']
    C2 = ekfData['C2']
    R0 = ekfData['R0']
    G  = ekfData['G']
    model_params = ekfData['model_params']

    # Per-step discrete-time decay factors
    a1 = np.exp(-sim_dt / (R1 * C1))
    a2 = np.exp(-sim_dt / (R2 * C2))
    Ah = np.exp(-abs(ik * G * sim_dt / (3600 * Q)))

    # Closed-form over KDT steps: x_RC[KDT] = a^KDT * x_RC[0] + ik * (1 - a^KDT)
    # Derived from geometric series: (1-a) * sum_{j=0}^{KDT-1} a^j * ik = ik * (1 - a^KDT)
    a1k = a1 ** KDT
    a2k = a2 ** KDT
    Ahk = Ah ** KDT

    ir1_f = a1k * x0[ir1Ind] + ik * (1 - a1k)
    ir2_f = a2k * x0[ir2Ind] + ik * (1 - a2k)

    # hk[KDT] = Ah^KDT * hk[0] + sign(ik) * (1 - Ah^KDT)
    sign_ik = np.sign(ik) if abs(ik) > 1e-9 else 0.0
    hk_f  = Ahk * x0[hkInd] + sign_ik * (1 - Ahk)

    # SOC integrates current over the actual simulated time (sim_dt * KDT seconds)
    SOC_f = x0[zkInd] + ik * sim_dt * KDT / (3600 * Q)

    xDT = np.array([ir1_f, ir2_f, hk_f, SOC_f])

    # Terminal voltage — same form as iterEKF
    OCVdch = getParamECM('E_OCV_dch_V', Tk, np.clip(SOC_f, 0.0, 1.0), model_params)
    OCVch  = getParamECM('E_OCV_ch_V',  Tk, np.clip(SOC_f, 0.0, 1.0), model_params)
    OCV    = ((1 + hk_f) / 2) * OCVch + ((1 - hk_f) / 2) * OCVdch
    vDT    = OCV + R1 * ir1_f + R2 * ir2_f + R0 * ik 

    return vDT, xDT


def bisect(f, a, b, tol=1e-2):
    """Find x in [a, b] where f(x) = 0 by bisection.

    Requires f(a) and f(b) to have opposite signs.
    If they don't, returns the endpoint with the smaller |f(x)|.
    """
    max_iter = np.ceil(np.log2((b - a) / tol))
    
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return a if abs(fa) < abs(fb) else b
    for _ in range(int(max_iter)):
        if abs(b - a) < tol:
            return (a + b) / 2
        mid = (a + b) / 2
        fm = f(mid)
        if fa * fm <= 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return (a + b) / 2


def computePowerLimits(ekfData, Tk):
    """Find the maximum charge and discharge current for the next P_DT seconds.

    Returns (ilimit_charge, ilimit_discharge) in Amperes.
    Sign convention: positive = charging, negative = discharging (matches EKF).

    Discharge: most negative current where vDT >= VMIN and SOC_f >= SOCMIN.
    Charge:    most positive current where vDT <= VMAX and SOC_f <= SOCMAX.
    """
    def h_discharge(ik):
        # h <= 0  means all discharge constraints satisfied
        vDT, xDT = simCellDT(ik, ekfData, Tk)
        return max(VMIN - vDT, SOCMIN - xDT[3])

    def h_charge(ik):
        # h >= 0  means all charge constraints satisfied
        vDT, xDT = simCellDT(ik, ekfData, Tk)
        return min(VMAX - vDT, SOCMAX - xDT[3])

    # Discharge: find most negative ik in [IMIN, 0] where h_discharge crosses zero
    if h_discharge(IMIN) <= 0:
        ilimit_discharge = IMIN          # full discharge range available
    elif h_discharge(0.0) >= 0:
        ilimit_discharge = 0.0           # no discharge possible at this state
    else:
        ilimit_discharge = bisect(h_discharge, IMIN, 0.0)

    # Charge: find most positive ik in [0, IMAX] where h_charge crosses zero
    if h_charge(IMAX) >= 0:
        ilimit_charge = IMAX             # full charge range available
    elif h_charge(0.0) <= 0:
        ilimit_charge = 0.0              # no charging possible at this state
    else:
        ilimit_charge = bisect(h_charge, 0.0, IMAX)

    #NOTE(CP): Think about whether I can avoid having to recall simCellDT here by reusing the results from the bisection search.
    vDT_discharge, _ = simCellDT(ilimit_discharge, ekfData, Tk)
    vDT_charge, _    = simCellDT(ilimit_charge,    ekfData, Tk)

    #Calculate power limit
    plimit_charge = ilimit_charge * vDT_charge
    plimit_discharge = ilimit_discharge * vDT_discharge

    return plimit_charge, plimit_discharge

# load CellModel % loads "model" of cell
cell_props = 'Molicel_INR-21700-P45B_cellprops_2.2.csv'
cell_params = 'Molicel_INR-21700-P45B_ECM_2.2.csv'
working_dir = os.getcwd()
model_props = pd.read_csv(working_dir + '/' + cell_props) 
model_params = pd.read_csv(working_dir + '/' + cell_params)

# Load cell-test data. Contains variable "DYNData" of which the field
# "script1" is of interest. It has sub-fields time, current, voltage, soc.
data_file = 'MOLICEL_P45B_079_025degC_DC_WLTP_5C_Dch_1p5C_Ch_validation.csv'
#data_file = 'MOLICEL-INR21700-P45B_019_Aging_Block_004_0.csv'
#data_file = 'MOLICEL-INR21700-P45B_019_Aging_Block_004_0 - shortened.csv' 
#data_file = 'MOLICEL_P45B_002_010degC_DC1.csv'

data = pd.read_csv(working_dir + '/' + data_file)  # loads data from cell test

#downsample data for faster processing (optional)
data = data.iloc[::50, :].reset_index(drop=True)

time = data['t_s'].to_numpy()
#deltat = time[1] - time[0]
time = time - time[0]  # start time at 0
current = data['I_exp_A'].to_numpy()  # discharge < 0, charge > 0
voltage = data['V_exp_V'].to_numpy()
temperature = data['T_exp_degC'].to_numpy()
soc = data['SOC'].to_numpy()

#Add an offset to the initial SOC to test robustness of EKF.
SOC_offset = 0 #NOTE(CP): Test an offset in initial SOC

#Add noise to the voltage and current measurements to test robustness of EKF.
voltage_noise_std = 0.01  # standard deviation of voltage noise
current_noise_std = 0.1   # standard deviation of current noise

voltage = voltage + np.random.normal(0, voltage_noise_std, size=voltage.shape)
current = current + np.random.normal(0, current_noise_std, size=current.shape)

# Reserve storage for computed results, for plotting
sochat = np.zeros_like(soc)
socbound = np.zeros_like(soc)
plimit_charge_arr    = []
plimit_discharge_arr = []

# Covariance values
SigmaX0 = np.diag([1e-3, 1e-3, 1e-3, 1e-2])  # uncertainty of initial state
SigmaV = 2e-2  # uncertainty of voltage sensor squared, output equation
SigmaW = 1e1  # uncertainty of current sensor squared, state equation

# Create ekfData structure and initialize variables using first
# voltage measurement and first temperature measurement
ekfData = initEKF(soc[0], SigmaX0, SigmaV, SigmaW, model_params, model_props, SOC_offset)

# Now, enter loop for remainder of time, where we update the EKF
# once per sample interval
# add progressbar

hks = []
yhats = []

for k in tqdm(range(len(voltage)), desc="Running EKF"):
    deltat_k = time[k] - time[k-1] if k > 0 else 0.0
    vk = voltage[k]  # "measure" voltage
    ik = current[k]  # "measure" current
    Tk = temperature[k]  # "measure" temperature

    # Update SOC (and other model states)
    sochat[k], socbound[k], ekfData = iterEKF(vk, ik, Tk, deltat_k, ekfData)

    # Calculate charge/discharge current limits for the next P_DT seconds
    plimit_charge, plimit_discharge = computePowerLimits(ekfData, Tk)
    plimit_charge_arr.append(plimit_charge)
    plimit_discharge_arr.append(plimit_discharge)

    #Store variables for plotting
    hks.append(ekfData['hk'])
    yhats.append(ekfData['yhat'])

    
# Plot results
print('RMS SOC estimation error = %g%%' %
      np.sqrt(np.mean((100*(soc-sochat))**2)))

ind = np.where(np.abs(soc-sochat) > socbound)[0]
print('Percent of time error outside bounds = %g%%' %
      (len(ind)/len(soc)*100))

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
fig.suptitle('EKF State Estimation Results')
t = time / 60

ax1 = axes[0, 0]
ax1.plot(t, 100*sochat, label='Estimate')
ax1.plot(t, 100*soc, label='Truth')
ax1.plot(t, 100*(sochat+socbound), 'k--', linewidth=0.8)
ax1.plot(t, 100*(sochat-socbound), 'k--', linewidth=0.8, label='Bounds')
ax1.set_title('SOC estimation using EKF')
ax1.set_ylabel('SOC (%)')
ax1.legend()
ax1.grid(True)

ax2 = axes[0, 1]
ax2.plot(t, 100*(soc-sochat), label='Estimation error')
ax2.plot(t, 100*socbound, 'k--', linewidth=0.8)
ax2.plot(t, -100*socbound, 'k--', linewidth=0.8, label='Bounds')
ax2.set_title('SOC estimation errors using EKF')
ax2.set_ylabel('SOC error (%)')
ax2.set_ylim([-20, 20])
ax2.legend()
ax2.grid(True)

# ax3 = axes[1, 0]
# ax3.plot(t, hks)
# ax3.set_title('Hysteresis state hk over time')
# ax3.set_xlabel('Time (min)')
# ax3.set_ylabel('Hysteresis state hk')
# ax3.grid(True)

#plot current limits
ax3 = axes[1, 0]
ax3.plot(t, plimit_charge_arr, label='Charge current limit')
ax3.plot(t, plimit_discharge_arr, label='Discharge current limit')
ax3.set_title('Power limits over time')
ax3.set_xlabel('Time (min)')
ax3.set_ylabel('Power limit (A)')
ax3.legend()

ax4 = axes[1, 1]
ax4.plot(t, voltage, label='True voltage')
ax4.plot(t, yhats, label='Predicted voltage')
ax4.set_title('Voltage prediction vs true voltage')
ax4.set_xlabel('Time (min)')
ax4.set_ylabel('Voltage (V)')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()


