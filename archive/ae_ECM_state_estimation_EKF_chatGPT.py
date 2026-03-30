import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def initEKF(SOC0, SigmaX0, SigmaV, SigmaW, model_params, model_props):
    ekfData = {}

    # State indices for x = [SOC, h, V1, V2]^T
    ekfData["socInd"] = 0
    ekfData["hInd"] = 1
    ekfData["v1Ind"] = 2
    ekfData["v2Ind"] = 3

    # Initial state: neutral hysteresis, relaxed RC voltages
    h0 = 0.0
    V10 = 0.0
    V20 = 0.0

    # Optional initial SOC offset for testing
    #SOC0 = SOC0 - 0.1

    ekfData["xhat"] = np.array([SOC0, h0, V10, V20], dtype=float).reshape(-1, 1)

    ekfData["SigmaX"] = SigmaX0
    ekfData["SigmaV"] = SigmaV
    ekfData["SigmaW"] = SigmaW
    ekfData["Qbump"] = 5.0

    ekfData["priorI"] = 0.0
    ekfData["model_params"] = model_params
    ekfData["model_props"] = model_props

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

        soc, idx = np.unique(soc, return_index=True)
        ocv = ocv[idx]

        docvdz = np.gradient(ocv, soc)
        z0 = np.clip(z, soc[0], soc[-1])
        vals.append(np.interp(z0, soc, docvdz))

    return float(np.interp(T, Ts, vals))


def ocv_from_soc_h(z, h, T, model_params):
    OCVdch = getParamECM("E_OCV_dch_V", T, z, model_params)
    OCVch = getParamECM("E_OCV_ch_V", T, z, model_params)
    OCV = 0.5 * (1 + h) * OCVch + 0.5 * (1 - h) * OCVdch
    return OCV, OCVch, OCVdch


def iterEKF(vk, ik, Tk, deltat, ekfData):
    model_params = ekfData["model_params"]
    model_props = ekfData["model_props"]

    xhat = ekfData["xhat"]
    SigmaX = ekfData["SigmaX"]
    SigmaV = ekfData["SigmaV"]
    SigmaW = ekfData["SigmaW"]

    socInd = ekfData["socInd"]
    hInd = ekfData["hInd"]
    v1Ind = ekfData["v1Ind"]
    v2Ind = ekfData["v2Ind"]

    # Use previous current for state propagation, same as your original structure
    I = ekfData["priorI"]

    soc = float(xhat[socInd, 0])
    h = float(xhat[hInd, 0])

    # Model parameters at current estimated SOC
    Q_Ah = getPropECM("Qnom_Ah", model_props)
    Q_As = 3600.0 * Q_Ah

    R0 = getParamECM("R_R0_Ohm", Tk, soc, model_params)
    R1 = getParamECM("R_R1_Ohm", Tk, soc, model_params)
    C1 = getParamECM("C_C1_F", Tk, soc, model_params)
    R2 = getParamECM("R_R2_Ohm", Tk, soc, model_params)
    C2 = getParamECM("C_C2_F", Tk, soc, model_params)
    gamma = getParamECM("gamma", Tk, soc, model_params)

    tau1 = R1 * C1
    tau2 = R2 * C2

    # Exact discrete-time matrices for x = [SOC, h, V1, V2]^T
    a_h = np.exp(-gamma * abs(I) * deltat / Q_As)
    a1 = np.exp(-deltat / tau1)
    a2 = np.exp(-deltat / tau2)

    Ad = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, a_h, 0.0, 0.0],
        [0.0, 0.0, a1, 0.0],
        [0.0, 0.0, 0.0, a2],
    ], dtype=float)

    # Negative current = discharge current
    Bd = np.array([
        [deltat / Q_As],            # SOC decreases when I < 0
        [0.0],                      # hysteresis uses explicit sign(I) branch forcing below
        [R1 * (1.0 - a1)],          # V1 becomes negative during discharge
        [R2 * (1.0 - a2)],          # V2 becomes negative during discharge
    ], dtype=float)

    sign_I = np.sign(I)
    if abs(I) < Q_Ah / 100.0:
        sign_I = 0.0

    Bh = np.zeros((4, 1), dtype=float)
    Bh[hInd, 0] = a_h - 1.0

    # State prediction
    xhat_minus = Ad @ xhat + Bd * I + Bh * sign_I

    # Bound physical states a little for robustness
    xhat_minus[hInd, 0] = np.clip(xhat_minus[hInd, 0], -1.0, 1.0)
    xhat_minus[socInd, 0] = np.clip(xhat_minus[socInd, 0], -0.05, 1.05)

    soc_minus = float(xhat_minus[socInd, 0])
    h_minus = float(xhat_minus[hInd, 0])

    # Recompute params at predicted SOC for output linearization
    R0 = getParamECM("R_R0_Ohm", Tk, soc_minus, model_params)
    R1 = getParamECM("R_R1_Ohm", Tk, soc_minus, model_params)
    C1 = getParamECM("C_C1_F", Tk, soc_minus, model_params)
    R2 = getParamECM("R_R2_Ohm", Tk, soc_minus, model_params)
    C2 = getParamECM("C_C2_F", Tk, soc_minus, model_params)
    gamma = getParamECM("gamma", Tk, soc_minus, model_params)

    # Process covariance prediction
    if np.ndim(SigmaW) == 0:
        SigmaX = Ad @ SigmaX @ Ad.T + Bd * SigmaW * Bd.T
    else:
        SigmaX = Ad @ SigmaX @ Ad.T + Bd @ np.atleast_2d(SigmaW) @ Bd.T

    # Nonlinear output prediction
    OCV, OCVch, OCVdch = ocv_from_soc_h(soc_minus, h_minus, Tk, model_params)
    yhat = OCV + R0 * I + float(xhat_minus[v1Ind, 0]) + float(xhat_minus[v2Ind, 0])

    # EKF measurement Jacobian Ck = d g / d x
    Ck = np.zeros((1, 4), dtype=float)
    Ck[0, socInd] = (
        0.5 * (1.0 + h_minus) * dOCVfromSOCtemp(soc_minus, Tk, model_params, "E_OCV_ch_V")
        + 0.5 * (1.0 - h_minus) * dOCVfromSOCtemp(soc_minus, Tk, model_params, "E_OCV_dch_V")
    )
    Ck[0, hInd] = 0.5 * (OCVch - OCVdch)
    Ck[0, v1Ind] = 1.0
    Ck[0, v2Ind] = 1.0

    # Direct-feedthrough from current in voltage equation
    Dk = 1

    # Innovation covariance
    # Measurement noise is additive on voltage measurement, so noise feedthrough is 1
    SigmaY = Ck @ SigmaX @ Ck.T + SigmaV

    # Kalman gain
    L = SigmaX @ Ck.T / SigmaY

    # Measurement update
    r = vk - yhat
    if r**2 > 100.0 * SigmaY:
        L[:] = 0.0

    xhat = xhat_minus + L * r
    xhat[hInd, 0] = np.clip(xhat[hInd, 0], -1.0, 1.0)
    xhat[socInd, 0] = np.clip(xhat[socInd, 0], -0.05, 1.05)

    # Joseph-form covariance update
    I4 = np.eye(4)
    SigmaX = (I4 - L @ Ck) @ SigmaX @ (I4 - L @ Ck).T + L * SigmaV * L.T

    if r**2 > 4.0 * SigmaY:
        print("Bumping SigmaX")
        SigmaX[socInd, socInd] *= ekfData["Qbump"]

    # Symmetrize covariance for robustness
    _, S, Vt = np.linalg.svd(SigmaX)
    V = Vt.T
    HH = V @ np.diag(S) @ V.T
    SigmaX = (SigmaX + SigmaX.T + HH + HH.T) / 4.0

    ekfData["priorI"] = ik
    ekfData["SigmaX"] = SigmaX
    ekfData["xhat"] = xhat
    ekfData["yhat"] = yhat

    soc_est = float(xhat[socInd, 0])
    soc_bnd = 3.0 * np.sqrt(SigmaX[socInd, socInd])

    return soc_est, soc_bnd, ekfData

# ---------------------------
# Load model and data
# ---------------------------

cell_props = "Molicel_INR-21700-P45B_cellprops_2.2.csv"
cell_params = "Molicel_INR-21700-P45B_ECM_2.2.csv"
working_dir = os.getcwd()

model_props = pd.read_csv(os.path.join(working_dir, cell_props))
model_params = pd.read_csv(os.path.join(working_dir, cell_params))

#data_file = "MOLICEL-INR21700-P45B_019_Aging_Block_004_0 - shortened - shortened.csv"
data_file = "MOLICEL-INR21700-P45B_019_Aging_Block_004_0 - shortened.csv"
#data_file = "MOLICEL_P45B_079_025degC_DC_WLTP_5C_Dch_1p5C_Ch_validation.csv"
data = pd.read_csv(os.path.join(working_dir, data_file))

# Optional downsample
data = data.iloc[::40, :].reset_index(drop=True)

time = data["t_s"].to_numpy()
deltat = time[1] - time[0]
time = time - time[0]

# Keep your original current convention
current = data["I_exp_A"].to_numpy()
voltage = data["V_exp_V"].to_numpy()
temperature = data["T_exp_degC"].to_numpy()
soc_true = data["SOC"].to_numpy()

sochat = np.zeros_like(soc_true)
socbound = np.zeros_like(soc_true)

# Covariances
SigmaX0 = np.diag([1e-2, 1e-3, 1e-3, 1e-3])   # [SOC, h, V1, V2]
SigmaV = 2e-2 #Sensor noise variance (voltage measurement noise)
SigmaW = 1e-1 #Process noise variance (model uncertainty, unmodeled dynamics, etc.)

ekfData = initEKF(soc_true[0], SigmaX0, SigmaV, SigmaW, model_params, model_props)

v_errs = []
for k in tqdm(range(len(voltage)), desc="Running EKF"):
    vk = voltage[k]
    ik = current[k]
    Tk = temperature[k]
    sochat[k], socbound[k], ekfData = iterEKF(vk, ik, Tk, deltat, ekfData)

    #voltage error
    v_errs.append(vk - ekfData["yhat"])

# ---------------------------
# Plot results
# ---------------------------

plt.figure(1)
plt.clf()
plt.plot(time / 60, 100 * sochat, label="Estimate")
plt.plot(time / 60, 100 * soc_true, label="Truth")
plt.plot(time / 60, 100 * (sochat + socbound))
plt.plot(time / 60, 100 * (sochat - socbound))
plt.title("SOC estimation using EKF")
plt.xlabel("Time (min)")
plt.ylabel("SOC (%)")
plt.legend(["Estimate", "Truth", "Bounds"])
plt.grid(True)

print("RMS SOC estimation error = %g%%" % np.sqrt(np.mean((100 * (soc_true - sochat)) ** 2)))

plt.figure(2)
plt.clf()
plt.plot(time / 60, 100 * (soc_true - sochat), label="Estimation error")
plt.plot(time / 60, 100 * socbound)
plt.plot(time / 60, -100 * socbound)
plt.title("SOC estimation errors using EKF")
plt.xlabel("Time (min)")
plt.ylabel("SOC error (%)")
plt.ylim([-4, 4])
plt.legend(["Estimation error", "Bounds"])
plt.grid(True)

ind = np.where(np.abs(soc_true - sochat) > socbound)[0]
print("Percent of time error outside bounds = %g%%" % (len(ind) / len(soc_true) * 100))

plt.show()

#plot voltage errors
plt.figure(3)
plt.clf()
plt.plot(time / 60, v_errs, label="Voltage error")
plt.title("Voltage estimation errors using EKF")
plt.xlabel("Time (min)")
plt.ylabel("Voltage error (V)")
plt.legend()
plt.grid(True)
plt.show()
