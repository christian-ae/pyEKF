#A:E ECM STATE ESTIMATION EKF
#model:

from pyexpat import model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator


def initEKF(SOC0, SigmaX0, SigmaV, SigmaW, model_params, model_props):
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
    SOC0 = SOC0-0.3

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

    #TODO: CHeck your matrices are correct for RC!!
    #TODO: Check sign conventions!!

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
    # if abs(ik) > Q / 100: 
    #     ekfData['signIk'] = np.sign(ik)
    # signIk = ekfData['signIk']

    # EKF Step 0: Compute Ahat[k-1], Bhat[k-1]
    nx = len(xhat)
    Ahat = np.zeros((nx, nx))
    Bhat = np.zeros((nx, 1))

    Ahat[zkInd, zkInd] = 1
    Bhat[zkInd, 0] = -deltat / (3600 * Q)

    Ahat[ir1Ind, ir1Ind] = R1C1
    Bhat[ir1Ind, 0] = (1 - R1C1)

    Ahat[ir2Ind, ir2Ind] = R2C2
    Bhat[ir2Ind, 0] = (1 - R2C2)

    Ah = np.exp(-abs(I * G * deltat / (3600 * Q)))  # hysteresis factor
    Ahat[hkInd, hkInd] = Ah
    B = np.hstack((Bhat, 0 * Bhat))
    Bhat[hkInd, 0] = -abs(G * deltat / (3600 * Q)) * Ah * (1 - np.sign(I) * xhat[hkInd, 0])
    B[hkInd, 1] = Ah - 1

    # Step 1a: State estimate time update
    xhat = Ahat @ xhat + B @ np.array([[I], [np.sign(I)]])

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
        - float(R1) * xhat[ir1Ind, 0]
        - float(R2) * xhat[ir2Ind, 0]
        - R0 * I
    )

    # Step 2a: Estimator gain matrix
    Chat = np.zeros((1, nx))
    Chat[0, zkInd] = dOCVfromSOCtemp(zk, Tk, model_params, 'E_OCV_ch_V') * (1+hk)/2 + dOCVfromSOCtemp(zk, Tk, model_params, 'E_OCV_dch_V') * (1-hk)/2
    Chat[0, hkInd] = 0.5 * (OCVch - OCVdch)
    Chat[0, ir1Ind] = -float(R1)
    Chat[0, ir2Ind] = -float(R2)

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

    zk = xhat[zkInd, 0]
    zkbnd = 3 * np.sqrt(SigmaX[zkInd, zkInd])

    return zk, zkbnd, ekfData

# load CellModel % loads "model" of cell
cell_props = 'Molicel_INR-21700-P45B_cellprops_2.2.csv'
cell_params = 'Molicel_INR-21700-P45B_ECM_2.2.csv'
working_dir = os.getcwd()
model_props = pd.read_csv(working_dir + '/' + cell_props) 
model_params = pd.read_csv(working_dir + '/' + cell_params)

# Load cell-test data. Contains variable "DYNData" of which the field
# "script1" is of interest. It has sub-fields time, current, voltage, soc.
data_file = 'MOLICEL_P45B_079_025degC_DC_WLTP_5C_Dch_1p5C_Ch_validation.csv'
data = pd.read_csv(working_dir + '/' + data_file)  # loads data from cell test

#downsample data for faster processing (optional)
data = data.iloc[::30, :].reset_index(drop=True)

time = data['t_s'].to_numpy()
deltat = time[1] - time[0]
time = time - time[0]  # start time at 0
current = -data['I_exp_A'].to_numpy()  # flip current sign so discharge < 0; charge > 0.
voltage = data['V_exp_V'].to_numpy()
temperature = data['T_exp_degC'].to_numpy()
soc = data['SOC'].to_numpy()

# Reserve storage for computed results, for plotting
sochat = np.zeros_like(soc)
socbound = np.zeros_like(soc)

# Covariance values
SigmaX0 = np.diag([1e-3, 1e-3, 1e-3, 1e-2])  # uncertainty of initial state
SigmaV = 2e-1  # uncertainty of voltage sensor, output equation
SigmaW = 1e1  # uncertainty of current sensor, state equation

# Create ekfData structure and initialize variables using first
# voltage measurement and first temperature measurement
ekfData = initEKF(soc[0], SigmaX0, SigmaV, SigmaW, model_params, model_props)

# Now, enter loop for remainder of time, where we update the EKF
# once per sample interval
# add progressbar

for k in tqdm(range(len(voltage)), desc="Running EKF"):
    vk = voltage[k]  # "measure" voltage
    ik = current[k]  # "measure" current
    Tk = temperature[k]  # "measure" temperature

    # Update SOC (and other model states)
    sochat[k], socbound[k], ekfData = iterEKF(vk, ik, Tk, deltat, ekfData)

# Plot results
plt.figure(1)
plt.clf()
plt.plot(time/60, 100*sochat, label='Estimate')
plt.plot(time/60, 100*soc, label='Truth')
plt.plot(time/60, 100*(sochat+socbound))
plt.plot(time/60, 100*(sochat-socbound))
plt.title('SOC estimation using EKF')
plt.xlabel('Time (min)')
plt.ylabel('SOC (%)')
plt.legend(['Estimate','Truth','Bounds'])
plt.grid(True)

print('RMS SOC estimation error = %g%%' %
      np.sqrt(np.mean((100*(soc-sochat))**2)))

plt.figure(2)
plt.clf()
plt.plot(time/60, 100*(soc-sochat), label='Estimation error')
plt.plot(time/60, 100*socbound)
plt.plot(time/60, -100*socbound)
plt.title('SOC estimation errors using EKF')
plt.xlabel('Time (min)')
plt.ylabel('SOC error (%)')
plt.ylim([-4,4])
plt.legend(['Estimation error','Bounds'])
plt.grid(True)

ind = np.where(np.abs(soc-sochat) > socbound)[0]
print('Percent of time error outside bounds = %g%%' %
      (len(ind)/len(soc)*100))

plt.show()