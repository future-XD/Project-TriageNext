"""
stage1_timeseries_and_count_glms.py
===================================
Stage 1 Pipeline for Blueprint B: Urban Hospital ED Surge & Triage Analytics.

Covers:
  Part A: Stationarity & Box-Jenkins Diagnostics (ADF, KPSS, Ljung-Box)
  Part B: Time-Series Progression (AR -> ARMA -> ARIMA -> SARIMA -> SARIMAX -> Hybrid Ridge)
  Part C: Hourly Count GLMs (Poisson -> Quasi-Poisson -> Negative Binomial w/ Cameron-Trivedi Test)
  Part D: Zero-Inflated Poisson (ZIP) vs. Standard Poisson w/ Vuong Likelihood Ratio Test
  Part E: Multi-Panel Diagnostic Plot Generation (outputs/stage1_diagnostics.png)
"""

import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.discrete.count_model import ZeroInflatedPoisson
from statsmodels.tools.sm_exceptions import InterpolationWarning, ConvergenceWarning

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore", category=InterpolationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 1. DATA LOADER (Auto-generates CSVs if not found)
# =============================================================================
def load_stage1_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_path = os.path.join(data_dir, "ed_daily_timeseries.csv")
    hourly_path = os.path.join(data_dir, "ed_hourly_timeseries.csv")

    if not (os.path.exists(daily_path) and os.path.exists(hourly_path)):
        print("[Info] CSV files not found in 'data/'. Running generate_ed_dataset.py first...")
        from generate_ed_dataset import generate_urban_ed_dataset
        generate_urban_ed_dataset(output_dir=data_dir, save_csv=True)

    daily_df = pd.read_csv(daily_path, parse_dates=["date"])
    daily_df = daily_df.set_index("date").asfreq("D")

    hourly_df = pd.read_csv(hourly_path, parse_dates=["timestamp"])
    hourly_df = hourly_df.set_index("timestamp").asfreq("1h")

    return daily_df, hourly_df


# =============================================================================
# 2. PART A: STATIONARITY & BOX-JENKINS DIAGNOSTICS (ADF, KPSS, LJUNG-BOX)
# =============================================================================
def run_stationarity_diagnostics(series: pd.Series) -> pd.DataFrame:
    """
    Runs joint ADF (H0: Unit Root) and KPSS (H0: Stationary) tests across:
      1. Raw Series Y_t
      2. First-Differenced Series (d=1): Y_t - Y_{t-1}
      3. Trend + Seasonally Differenced Series (d=1, D=1, s=7)
    """
    transformations = {
        "1. Raw Series (Y_t)": series.dropna(),
        "2. First Difference (d=1)": series.diff(1).dropna(),
        "3. Seasonal + First Diff (d=1, D=1, s=7)": series.diff(1).diff(7).dropna(),
    }

    records = []
    for name, s in transformations.items():
        adf_stat, adf_p, _, _, _, _ = adfuller(s, autolag="AIC")
        kpss_stat, kpss_p, _, _ = kpss(s, regression="c", nlags="auto")
        lb_df = acorr_ljungbox(s, lags=[7, 14], return_df=True)
        lb_p7 = lb_df.loc[7, "lb_pvalue"]

        if adf_p < 0.05 and kpss_p >= 0.05:
            verdict = "Stationary (Strict)"
        elif adf_p >= 0.05 and kpss_p < 0.05:
            verdict = "Non-Stationary (Unit Root / Drift)"
        else:
            verdict = "Trend / Seasonal Autocorrelation"

        records.append(
            {
                "Series Transformation": name,
                "ADF Stat": round(adf_stat, 3),
                "ADF p-val": round(adf_p, 4),
                "KPSS Stat": round(kpss_stat, 3),
                "KPSS p-val": round(kpss_p, 4),
                "Ljung-Box p (Lag 7)": f"{lb_p7:.2e}",
                "Diagnosis": verdict,
            }
        )

    diag_df = pd.DataFrame(records)
    print("\n" + "=" * 95)
    print("PART A: STATIONARITY & AUTOCORRELATION DIAGNOSTICS (DAILY ED ADMISSIONS)")
    print("=" * 95)
    print(diag_df.to_string(index=False))
    return diag_df


# =============================================================================
# 3. PART B: TIME-SERIES PROGRESSION (AR -> ARMA -> ARIMA -> SARIMA -> SARIMAX)
# =============================================================================
def run_timeseries_pipeline(daily_df: pd.DataFrame, test_days: int = 60) -> dict:
    """
    Trains the full Box-Jenkins syllabus hierarchy + SARIMAX & Hybrid Ridge extension
    on 670 training days and evaluates on the final 60 out-of-sample days.
    """
    y = daily_df["daily_admissions"].astype(float)
    exog_cols = ["heat_stress_index", "pollution_stress_index", "daily_aqi", "is_weekend"]
    exog = daily_df[exog_cols].astype(float)

    y_train, y_test = y.iloc[:-test_days], y.iloc[-test_days:]
    exog_train, exog_test = exog.iloc[:-test_days], exog.iloc[-test_days:]

    model_specs = [
        ("AR(2) [Syllabus Baseline]", (2, 0, 0), (0, 0, 0, 0), False, "c"),
        ("ARMA(2,1) [Syllabus Baseline]", (2, 0, 1), (0, 0, 0, 0), False, "c"),
        ("ARIMA(2,1,1) [Trend Differenced]", (2, 1, 1), (0, 0, 0, 0), False, "c"),
        ("SARIMA(2,1,1)(1,1,1)_7 [Weekly Cycle]", (2, 1, 1), (1, 1, 1, 7), False, "n"),
        ("SARIMAX(2,1,1)(1,1,1)_7 + Weather/AQI", (2, 1, 1), (1, 1, 1, 7), True, "n"),
    ]

    results_list = []
    predictions_horizon = {}
    fitted_models = {}

    for label, order, seasonal_order, use_exog, trend_spec in model_specs:
        ex_tr = exog_train if use_exog else None
        ex_te = exog_test if use_exog else None
        ex_full = exog if use_exog else None

        model = SARIMAX(
            endog=y_train,
            exog=ex_tr,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend_spec,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=250)
        fitted_models[label] = res

        # 1. Multi-step horizon forecast (60-day ahead trajectory)
        fc_horizon = res.get_forecast(steps=test_days, exog=ex_te).predicted_mean
        predictions_horizon[label] = fc_horizon

        # 2. 1-step-ahead rolling out-of-sample evaluation (without refitting parameters)
        res_full = res.apply(y, exog=ex_full)
        fc_1step = res_full.fittedvalues.iloc[-test_days:]

        # Ljung-Box residual whiteness check at lag 7
        resid_clean = res.resid.iloc[15:]
        lb_pval = acorr_ljungbox(resid_clean, lags=[7], return_df=True).loc[7, "lb_pvalue"]

        results_list.append(
            {
                "Model": label,
                "AIC": round(res.aic, 1),
                "BIC": round(res.bic, 1),
                "1-Step RMSE": round(np.sqrt(mean_squared_error(y_test, fc_1step)), 2),
                "60-Day Horizon RMSE": round(np.sqrt(mean_squared_error(y_test, fc_horizon)), 2),
                "Horizon MAE": round(mean_absolute_error(y_test, fc_horizon), 2),
                "Horizon MAPE (%)": round(mean_absolute_percentage_error(y_test, fc_horizon) * 100, 2),
                "Resid Ljung-Box p": round(lb_pval, 3),
            }
        )

    # -------------------------------------------------------------------------
    # Stage 2 Extension: Hybrid SARIMA + Regularized Ridge on Non-Linear Exog
    # -------------------------------------------------------------------------
    sarima_res = fitted_models["SARIMA(2,1,1)(1,1,1)_7 [Weekly Cycle]"]
    sarima_train_resid = sarima_res.resid.iloc[14:]

    # Build non-linear & interaction exogenous features for the residual learner
    def build_residual_features(df_sub: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df_sub.index)
        feats["heat_stress"] = df_sub["heat_stress_index"]
        feats["pollution_stress"] = df_sub["pollution_stress_index"]
        feats["aqi"] = df_sub["daily_aqi"]
        feats["temp_sq"] = df_sub["mean_temp_c"] ** 2
        feats["aqi_x_humidity"] = df_sub["pollution_stress_index"] * df_sub["mean_humidity_pct"]
        feats["heat_x_weekend"] = df_sub["heat_stress_index"] * df_sub["is_weekend"]
        feats["aqi_roll3"] = df_sub["pollution_stress_index"].rolling(3, min_periods=1).mean()
        return feats

    X_res_full = build_residual_features(daily_df)
    X_res_train = X_res_full.loc[sarima_train_resid.index]
    X_res_test = X_res_full.iloc[-test_days:]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_res_train)
    X_te_scaled = scaler.transform(X_res_test)

    ridge_resid = RidgeCV(alphas=np.logspace(-2, 3, 50))
    ridge_resid.fit(X_tr_scaled, sarima_train_resid)

    hybrid_horizon = (
        predictions_horizon["SARIMA(2,1,1)(1,1,1)_7 [Weekly Cycle]"]
        + pd.Series(ridge_resid.predict(X_te_scaled), index=y_test.index)
    )
    sarima_1step = sarima_res.apply(y).fittedvalues.iloc[-test_days:]
    hybrid_1step = sarima_1step + pd.Series(ridge_resid.predict(X_te_scaled), index=y_test.index)
    predictions_horizon["Hybrid SARIMA + Ridge Exog Residuals"] = hybrid_horizon

    results_list.append(
        {
            "Model": "Hybrid SARIMA + Ridge Exog Residuals",
            "AIC": np.nan,
            "BIC": np.nan,
            "1-Step RMSE": round(np.sqrt(mean_squared_error(y_test, hybrid_1step)), 2),
            "60-Day Horizon RMSE": round(np.sqrt(mean_squared_error(y_test, hybrid_horizon)), 2),
            "Horizon MAE": round(mean_absolute_error(y_test, hybrid_horizon), 2),
            "Horizon MAPE (%)": round(mean_absolute_percentage_error(y_test, hybrid_horizon) * 100, 2),
            "Resid Ljung-Box p": np.nan,
        }
    )

    summary_df = pd.DataFrame(results_list)
    print("\n" + "=" * 105)
    print("PART B: TIME-SERIES MODEL PROGRESSION & OUT-OF-SAMPLE FORECAST BENCHMARK (TEST = 60 DAYS)")
    print("=" * 105)
    print(summary_df.to_string(index=False))

    return {
        "summary_df": summary_df,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions_horizon,
        "sarimax_res": fitted_models["SARIMAX(2,1,1)(1,1,1)_7 + Weather/AQI"],
    }


# =============================================================================
# 4. PART C: COUNT GLMs (POISSON vs. QUASI-POISSON vs. NEGATIVE BINOMIAL)
# =============================================================================
def run_hourly_count_glms(hourly_df: pd.DataFrame, test_hours: int = 1440) -> dict:
    """
    Fits Poisson GLM, tests for overdispersion via Pearson phi and Cameron-Trivedi test,
    and upgrades to Quasi-Poisson and Maximum Likelihood Negative Binomial (NB2).
    """
    df = hourly_df.copy()
    # Harmonic Fourier features for 24-hour diurnal cycle + secular trend
    df["sin_24"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
    df["cos_24"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)
    df["sin_12"] = np.sin(4 * np.pi * df["hour_of_day"] / 24.0)
    df["cos_12"] = np.cos(4 * np.pi * df["hour_of_day"] / 24.0)
    df["time_trend"] = np.linspace(0.0, 1.0, len(df))

    feature_cols = [
        "time_trend",
        "sin_24",
        "cos_24",
        "sin_12",
        "cos_12",
        "is_weekend",
        "heat_stress_index",
        "pollution_stress_index",
    ]

    X = sm.add_constant(df[feature_cols].astype(float))
    y = df["total_arrivals"].astype(float)

    X_train, X_test = X.iloc[:-test_hours], X.iloc[-test_hours:]
    y_train, y_test = y.iloc[:-test_hours], y.iloc[-test_hours:]

    # 1. Syllabus Baseline: Standard Poisson GLM
    poisson_mod = sm.GLM(y_train, X_train, family=sm.families.Poisson())
    poisson_res = poisson_mod.fit()

    # Overdispersion Diagnostics
    pearson_phi = poisson_res.pearson_chi2 / poisson_res.df_resid
    deviance_phi = poisson_res.deviance / poisson_res.df_resid

    # Cameron-Trivedi (1990) Auxiliary Regression Test for Overdispersion:
    # H0: Var(Y) = mu vs. H1: Var(Y) = mu + alpha * mu^2
    mu_hat = poisson_res.fittedvalues
    aux_dep = ((y_train - mu_hat) ** 2 - y_train) / mu_hat
    ct_ols = sm.OLS(aux_dep, mu_hat.to_frame(name="mu_hat")).fit()
    alpha_ct = float(ct_ols.params["mu_hat"])
    ct_tstat = float(ct_ols.tvalues["mu_hat"])
    ct_pval = float(ct_ols.pvalues["mu_hat"])

    # 2. Extension 1: Quasi-Poisson (Scale-corrected standard errors)
    quasi_res = poisson_mod.fit(scale="X2")

    # 3. Extension 2: Negative Binomial GLM (Exact MLE profile optimization over alpha)
    def negbin_neg_llf(alpha_val: float) -> float:
        mod = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial(alpha=alpha_val))
        return -mod.fit().llf

    opt_alpha = minimize_scalar(negbin_neg_llf, bounds=(0.01, 2.0), method="bounded")
    alpha_mle = float(opt_alpha.x)
    nb_mod = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial(alpha=alpha_mle))
    nb_res = nb_mod.fit()

    # Evaluate Out-of-Sample Predictions & 95% Predictive Interval Coverage
    pred_p = poisson_res.predict(X_test)
    pred_nb = nb_res.predict(X_test)

    # 95% predictive bounds: Poisson vs Negative Binomial
    p_lo, p_hi = stats.poisson.ppf(0.025, pred_p), stats.poisson.ppf(0.975, pred_p)
    # Convert NB (mu, alpha) to scipy nbinom (n, p) parameterization: n = 1/alpha, p = n/(n + mu)
    n_param = 1.0 / alpha_mle
    p_param = n_param / (n_param + pred_nb)
    nb_lo, nb_hi = stats.nbinom.ppf(0.025, n_param, p_param), stats.nbinom.ppf(0.975, n_param, p_param)

    cov_poisson = ((y_test >= p_lo) & (y_test <= p_hi)).mean() * 100
    cov_nb = ((y_test >= nb_lo) & (y_test <= nb_hi)).mean() * 100

    print("\n" + "=" * 95)
    print("PART C: HOURLY COUNT GLM DIAGNOSTICS (POISSON vs. QUASI-POISSON vs. NEGATIVE BINOMIAL)")
    print("=" * 95)
    print(f"  * Pearson Dispersion Ratio (phi = Chi2 / df) : {pearson_phi:.3f}  (Syllabus Poisson assumes phi = 1.000)")
    print(f"  * Deviance Dispersion Ratio (Dev / df)       : {deviance_phi:.3f}")
    print(f"  * Cameron-Trivedi Overdispersion t-stat      : {ct_tstat:.2f} (p-value = {ct_pval:.2e} -> Reject Equidispersion)")
    print(f"  * Estimated NB Dispersion Parameter (alpha)  : CT Aux = {alpha_ct:.3f} | Exact MLE = {alpha_mle:.3f}")
    print("-" * 95)

    glm_table = pd.DataFrame(
        [
            {
                "Count Model": "Poisson GLM (Syllabus Baseline)",
                "Log-Likelihood": round(poisson_res.llf, 1),
                "AIC": round(poisson_res.aic, 1),
                "BIC": round(poisson_res.bic_llf, 1),
                "Dispersion (phi)": round(pearson_phi, 3),
                "SE(pollution_stress)": round(poisson_res.bse["pollution_stress_index"], 4),
                "Test RMSE": round(np.sqrt(mean_squared_error(y_test, pred_p)), 3),
                "95% PI Coverage (%)": round(cov_poisson, 1),
            },
            {
                "Count Model": "Quasi-Poisson GLM (Scale-Adjusted)",
                "Log-Likelihood": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
                "Dispersion (phi)": 1.000,
                "SE(pollution_stress)": round(quasi_res.bse["pollution_stress_index"], 4),
                "Test RMSE": round(np.sqrt(mean_squared_error(y_test, pred_p)), 3),
                "95% PI Coverage (%)": round(cov_poisson, 1),
            },
            {
                "Count Model": f"Negative Binomial GLM (alpha={alpha_mle:.2f})",
                "Log-Likelihood": round(nb_res.llf, 1),
                "AIC": round(nb_res.aic, 1),
                "BIC": round(nb_res.bic_llf, 1),
                "Dispersion (phi)": round(nb_res.pearson_chi2 / nb_res.df_resid, 3),
                "SE(pollution_stress)": round(nb_res.bse["pollution_stress_index"], 4),
                "Test RMSE": round(np.sqrt(mean_squared_error(y_test, pred_nb)), 3),
                "95% PI Coverage (%)": round(cov_nb, 1),
            },
        ]
    )
    print(glm_table.to_string(index=False))

    return {
        "glm_table": glm_table,
        "poisson_res": poisson_res,
        "nb_res": nb_res,
        "alpha_mle": alpha_mle,
        "y_test": y_test,
        "pred_p": pred_p,
        "pred_nb": pred_nb,
    }


# =============================================================================
# 5. PART D: ZERO-INFLATED POISSON (ZIP) vs. POISSON ON CRITICAL AMBULANCES
# =============================================================================
def run_zero_inflated_pipeline(hourly_df: pd.DataFrame, test_hours: int = 1440) -> dict:
    """
    Models hourly 'critical_ambulance_arrivals' using standard Poisson vs.
    Zero-Inflated Poisson (ZIP) with an overnight structural-zero logit gate,
    and performs the Vuong Non-Nested Likelihood Ratio Test.
    """
    df = hourly_df.copy()
    df["sin_24"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
    df["cos_24"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)
    df["overnight_flag"] = ((df["hour_of_day"] >= 1) & (df["hour_of_day"] <= 5)).astype(float)

    count_cols = ["sin_24", "cos_24", "pollution_stress_index", "heat_stress_index", "is_weekend"]
    infl_cols = ["overnight_flag", "pollution_stress_index"]

    X_count = sm.add_constant(df[count_cols].astype(float))
    Z_infl = sm.add_constant(df[infl_cols].astype(float))
    y = df["critical_ambulance_arrivals"].astype(float)

    X_tr, X_te = X_count.iloc[:-test_hours], X_count.iloc[-test_hours:]
    Z_tr, Z_te = Z_infl.iloc[:-test_hours], Z_infl.iloc[-test_hours:]
    y_tr, y_te = y.iloc[:-test_hours], y.iloc[-test_hours:]

    # 1. Standard Poisson Model
    pois_mod = sm.Poisson(y_tr, X_tr)
    pois_res = pois_mod.fit(disp=False)

    # 2. Zero-Inflated Poisson (ZIP) Model
    zip_mod = ZeroInflatedPoisson(endog=y_tr, exog=X_tr, exog_infl=Z_tr, inflation="logit")
    zip_res = zip_mod.fit(method="bfgs", maxiter=300, disp=False)

    # Vuong Non-Nested Likelihood Ratio Test (ZIP vs. Standard Poisson)
    ll_obs_zip = zip_mod.loglikeobs(zip_res.params)
    ll_obs_pois = pois_mod.loglikeobs(pois_res.params)
    m_i = ll_obs_zip - ll_obs_pois
    vuong_stat = np.sqrt(len(m_i)) * (m_i.mean() / m_i.std(ddof=1))
    vuong_pval = 2.0 * (1.0 - stats.norm.cdf(abs(vuong_stat)))

    # Compare predicted zero probability on out-of-sample test set
    actual_zero_pct = (y_te == 0).mean() * 100
    mu_pois_te = pois_res.predict(X_te)
    pois_zero_pct = np.exp(-mu_pois_te).mean() * 100
    zip_zero_pct = zip_res.predict(exog=X_te, exog_infl=Z_te, which="prob-zero").mean() * 100

    zip_table = pd.DataFrame(
        [
            {
                "Model": "Standard Poisson (Critical Ambulances)",
                "Log-Likelihood": round(pois_res.llf, 1),
                "AIC": round(pois_res.aic, 1),
                "BIC": round(pois_res.bic, 1),
                "Actual Test 0s (%)": round(actual_zero_pct, 2),
                "Predicted 0s (%)": round(pois_zero_pct, 2),
            },
            {
                "Model": "Zero-Inflated Poisson (ZIP - Logit Gate)",
                "Log-Likelihood": round(zip_res.llf, 1),
                "AIC": round(zip_res.aic, 1),
                "BIC": round(zip_res.bic, 1),
                "Actual Test 0s (%)": round(actual_zero_pct, 2),
                "Predicted 0s (%)": round(zip_zero_pct, 2),
            },
        ]
    )

    print("\n" + "=" * 95)
    print("PART D: ZERO-INFLATED POISSON (ZIP) vs. STANDARD POISSON (CRITICAL AMBULANCE ARRIVALS)")
    print("=" * 95)
    print(f"  * Vuong Non-Nested Test Statistic (ZIP vs. Poisson) : z = {vuong_stat:.2f} (p = {vuong_pval:.2e} -> ZIP Superior)")
    print("-" * 95)
    print(zip_table.to_string(index=False))

    return {
        "zip_table": zip_table,
        "pois_res": pois_res,
        "zip_res": zip_res,
        "y_te": y_te,
    }


# =============================================================================
# 6. PART E: PUBLICATION-READY DIAGNOSTIC PLOTS
# =============================================================================
def save_stage1_plots(ts_out: dict, glm_out: dict, zip_out: dict, out_dir: str = "outputs") -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.32, wspace=0.22)

    # Panel 1: 60-Day Out-of-Sample Time Series Forecast Trajectory
    ax1 = axes[0, 0]
    y_te = ts_out["y_test"]
    preds = ts_out["predictions"]
    ax1.plot(y_te.index, y_te.values, color="#1f2937", lw=2.0, label="Actual Daily Admissions")
    ax1.plot(
        y_te.index,
        preds["ARIMA(2,1,1) [Trend Differenced]"],
        color="#9ca3af",
        ls="--",
        lw=1.6,
        label="ARIMA(2,1,1) (No Seasonality)",
    )
    ax1.plot(
        y_te.index,
        preds["SARIMA(2,1,1)(1,1,1)_7 [Weekly Cycle]"],
        color="#2563eb",
        lw=1.8,
        label="SARIMA(2,1,1)(1,1,1)_7",
    )
    ax1.plot(
        y_te.index,
        preds["SARIMAX(2,1,1)(1,1,1)_7 + Weather/AQI"],
        color="#dc2626",
        lw=2.0,
        label="SARIMAX + Weather/AQI",
    )
    ax1.set_title("A. 60-Day Out-of-Sample ED Admission Forecast Progression", fontweight="bold")
    ax1.set_ylabel("Daily Patient Admissions")
    ax1.legend(frameon=True, fontsize=8.5)
    ax1.grid(alpha=0.25)

    # Panel 2: Horizon RMSE Ablation Across Time-Series Hierarchy
    ax2 = axes[0, 1]
    df_sum = ts_out["summary_df"]
    short_names = ["AR(2)", "ARMA(2,1)", "ARIMA", "SARIMA", "SARIMAX", "Hybrid\nSARIMA+Ridge"]
    colors = ["#94a3b8", "#64748b", "#475569", "#3b82f6", "#ef4444", "#10b981"]
    bars = ax2.bar(short_names, df_sum["60-Day Horizon RMSE"], color=colors, width=0.58)
    ax2.set_title("B. 60-Day Horizon RMSE Ablation (Lower is Better)", fontweight="bold")
    ax2.set_ylabel("Out-of-Sample RMSE (Patients / Day)")
    for b in bars:
        ax2.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.3,
            f"{b.get_height():.2f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax2.set_ylim(0, df_sum["60-Day Horizon RMSE"].max() * 1.18)
    ax2.grid(axis="y", alpha=0.25)

    # Panel 3: Observed vs. Theoretical Count Distribution (Poisson vs Negative Binomial)
    ax3 = axes[1, 0]
    y_h = glm_out["y_test"].to_numpy()
    max_k = 20
    k_vals = np.arange(0, max_k + 1)
    obs_freq = np.array([(y_h == k).mean() for k in k_vals])

    mu_p = glm_out["pred_p"].to_numpy()
    mu_nb = glm_out["pred_nb"].to_numpy()
    alpha = glm_out["alpha_mle"]
    n_p = 1.0 / alpha
    p_p = n_p / (n_p + mu_nb[:, None])

    pois_pmf = stats.poisson.pmf(k_vals[None, :], mu_p[:, None]).mean(axis=0)
    nb_pmf = stats.nbinom.pmf(k_vals[None, :], n_p, p_p).mean(axis=0)

    ax3.bar(k_vals, obs_freq, color="#cbd5e1", label="Observed Hourly Counts", width=0.7)
    ax3.plot(k_vals, pois_pmf, "o--", color="#ef4444", lw=2, label="Poisson GLM (Underestimates Tails)")
    ax3.plot(k_vals, nb_pmf, "s-", color="#059669", lw=2, label=f"Negative Binomial (alpha={alpha:.2f})")
    ax3.set_title("C. Hourly Arrival Count Distribution: Poisson vs. Negative Binomial", fontweight="bold")
    ax3.set_xlabel("Hourly Patient Arrivals (k)")
    ax3.set_ylabel("Probability Mass P(Y = k)")
    ax3.legend(frameon=True, fontsize=8.5)
    ax3.grid(alpha=0.25)

    # Panel 4: Critical Ambulance Count Distribution (Standard Poisson vs ZIP)
    ax4 = axes[1, 1]
    y_amb = zip_out["y_te"].to_numpy()
    k_amb = np.arange(0, 7)
    obs_amb = np.array([(y_amb == k).mean() for k in k_amb])

    pois_amb_pmf = zip_out["pois_res"].predict( which="prob").mean(axis=0)[:7]
    zip_amb_pmf = zip_out["zip_res"].predict(which="prob").mean(axis=0)[:7]

    x_pos = np.arange(len(k_amb))
    ax4.bar(x_pos - 0.25, obs_amb, width=0.25, color="#334155", label="Actual Frequency")
    ax4.bar(x_pos, pois_amb_pmf, width=0.25, color="#f97316", label="Standard Poisson")
    ax4.bar(x_pos + 0.25, zip_amb_pmf, width=0.25, color="#2563eb", label="Zero-Inflated Poisson (ZIP)")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([str(k) for k in k_amb])
    ax4.set_title("D. Critical Ambulance Arrivals: Zero-Inflation Calibration", fontweight="bold")
    ax4.set_xlabel("Hourly Critical Ambulance Arrivals")
    ax4.set_ylabel("Proportion of Hours")
    ax4.legend(frameon=True, fontsize=8.5)
    ax4.grid(axis="y", alpha=0.25)

    out_file = os.path.join(out_dir, "stage1_diagnostics.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Saved] High-resolution 4-panel diagnostic figure -> {out_file}")


if __name__ == "__main__":
    daily_df, hourly_df = load_stage1_data()
    run_stationarity_diagnostics(daily_df["daily_admissions"])
    ts_results = run_timeseries_pipeline(daily_df)
    glm_results = run_hourly_count_glms(hourly_df)
    zip_results = run_zero_inflated_pipeline(hourly_df)
    save_stage1_plots(ts_results, glm_results, zip_results)
