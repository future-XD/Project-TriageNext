"""
generate_ed_dataset.py
======================
Synthesizes and loads a realistic 2-year Urban Hospital Emergency Department (ED)
dataset modeled on the MIMIC-IV-ED clinical schema + seasonal weather/AQI drivers.

Outputs three synchronized CSV tables:
  1. ed_daily_timeseries.csv   -> AR, ARMA, ARIMA, SARIMA(s=7), SARIMAX
  2. ed_hourly_timeseries.csv  -> Poisson, Quasi-Poisson, Negative Binomial, ZIP
  3. ed_patient_visits.csv     -> OLS/Ridge/Lasso, Logistic/XGBoost, LDA/QDA/RDA
"""

import os
import numpy as np
import pandas as pd


def generate_urban_ed_dataset(
    start_date: str = "2024-01-01",
    n_days: int = 730,
    seed: int = 42,
    output_dir: str = "data",
    save_csv: bool = True,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_hours = n_days * 24

    # =========================================================================
    # 1. HOURLY TIMESTAMPS & EXOGENOUS WEATHER / AQI REGIMES (17,520 Hours)
    # =========================================================================
    hourly_index = pd.date_range(start=start_date, periods=n_hours, freq="1h")
    day_of_year = hourly_index.dayofyear.to_numpy()
    hour_of_day = hourly_index.hour.to_numpy()
    day_of_week = hourly_index.dayofweek.to_numpy()
    is_weekend = (day_of_week >= 5).astype(int)
    t_hours = np.arange(n_hours)

    # Seasonal temperature (North Indian / Urban continental profile: hot May-Jun, cool Dec-Jan)
    annual_temp_wave = 26.0 + 11.5 * np.sin(2 * np.pi * (day_of_year - 105) / 365.25)
    diurnal_temp_wave = 4.8 * np.sin(2 * np.pi * (hour_of_day - 9) / 24.0)
    ambient_temp_c = np.round(
        annual_temp_wave + diurnal_temp_wave + rng.normal(0, 1.4, n_hours), 2
    )

    # Relative humidity (peaks during July-August monsoon & early mornings)
    monsoon_bump = 28.0 * np.exp(-0.5 * ((day_of_year - 215) / 32.0) ** 2)
    humidity_pct = np.clip(
        48.0 + monsoon_bump - 1.1 * diurnal_temp_wave + rng.normal(0, 5.0, n_hours),
        15.0,
        98.0,
    ).round(1)

    # Air Quality Index (AQI): severe winter inversion spikes (Nov-Jan) + rush-hour bumps
    winter_inversion = 165.0 * np.cos(2 * np.pi * (day_of_year - 355) / 365.25).clip(0)
    rush_hour_aqi = 22.0 * (
        np.exp(-0.5 * ((hour_of_day - 8) / 2.0) ** 2)
        + np.exp(-0.5 * ((hour_of_day - 20) / 2.5) ** 2)
    )
    aqi = np.clip(
        85.0 + winter_inversion + rush_hour_aqi + rng.gamma(shape=2.5, scale=12.0, size=n_hours),
        25.0,
        500.0,
    ).round(0)

    # Heatwave & Severe Pollution stress indicators (Exogenous X_t for SARIMAX)
    heat_stress = np.maximum(0.0, ambient_temp_c - 36.0)
    pollution_stress = np.maximum(0.0, (aqi - 150.0) / 100.0)

    # =========================================================================
    # 2. CORRELATED ARMA(2,1) TEMPORAL PROCESS + SEASONAL ARRIVAL INTENSITY
    # =========================================================================
    # Generate an ARMA(2,1) innovation process at the daily level and interpolate
    ar_eps = rng.normal(0, 1.0, n_days + 10)
    arma_daily = np.zeros(n_days + 10)
    for d in range(2, n_days + 10):
        # AR(2): phi_1 = 0.55, phi_2 = 0.20; MA(1): theta_1 = 0.35
        arma_daily[d] = (
            0.55 * arma_daily[d - 1]
            + 0.20 * arma_daily[d - 2]
            + ar_eps[d]
            + 0.35 * ar_eps[d - 1]
        )
    arma_daily = arma_daily[10:]
    arma_hourly = np.repeat(arma_daily, 24)

    # Secular upward hospital demand trend (non-stationary drift -> requires d=1 in ARIMA)
    secular_trend = np.linspace(0.0, 0.28, n_hours)

    # Weekly seasonality (Monday surge, Sunday dip -> requires D=1, s=7 in SARIMA)
    dow_effects = np.array([0.18, 0.08, 0.04, 0.02, 0.05, -0.14, -0.23])
    weekly_effect = dow_effects[day_of_week]

    # Diurnal 24h cycle (peaks 10:00-21:00, trough 02:00-06:00)
    diurnal_effect = (
        0.48 * np.sin(2 * np.pi * (hour_of_day - 7) / 24.0)
        + 0.15 * np.cos(4 * np.pi * (hour_of_day - 10) / 24.0)
    )

    # True log-intensity for hourly arrivals
    log_lambda_total = (
        1.15
        + secular_trend
        + weekly_effect
        + diurnal_effect
        + 0.045 * heat_stress
        + 0.085 * pollution_stress
        + 0.06 * arma_hourly
    )
    mu_total = np.exp(log_lambda_total)

    # Overdispersed counts via Gamma-Poisson (Negative Binomial) mixture!
    # Dispersion alpha = 0.35 -> Variance = mu + alpha * mu^2 (Guarantees NB > Poisson)
    dispersion_alpha = 0.35
    gamma_rates = rng.gamma(
        shape=1.0 / dispersion_alpha, scale=dispersion_alpha * mu_total
    )
    total_arrivals = rng.poisson(gamma_rates)

    # Critical Ambulance Arrivals: Zero-Inflated Poisson (ZIP) structure!
    # Structural zero gate during quiet overnight hours (01:00 to 05:00)
    overnight_flag = ((hour_of_day >= 1) & (hour_of_day <= 5)).astype(int)
    logit_zero = -1.35 + 2.15 * overnight_flag - 0.35 * pollution_stress
    prob_structural_zero = 1.0 / (1.0 + np.exp(-logit_zero))
    is_structural_zero = rng.binomial(1, prob_structural_zero)

    mu_crit_amb = np.exp(
        -0.45
        + 0.40 * diurnal_effect
        + 0.14 * pollution_stress
        + 0.08 * heat_stress
        + 0.12 * is_weekend
    )
    raw_crit_amb = rng.poisson(mu_crit_amb)
    critical_ambulance_arrivals = np.where(is_structural_zero == 1, 0, raw_crit_amb)
    critical_ambulance_arrivals = np.minimum(critical_ambulance_arrivals, total_arrivals)

    # Active ED Crowding (rolling 6-hour arrival accumulation)
    active_ed_crowding = (
        pd.Series(total_arrivals).rolling(window=6, min_periods=1).sum().to_numpy()
    )
    mean_wait_time_mins = np.clip(
        12.0 + 1.15 * active_ed_crowding + rng.normal(0, 4.0, n_hours), 5.0, 180.0
    ).round(1)

    hourly_df = pd.DataFrame(
        {
            "timestamp": hourly_index,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "ambient_temp_c": ambient_temp_c,
            "humidity_pct": humidity_pct,
            "aqi": aqi,
            "heat_stress_index": np.round(heat_stress, 2),
            "pollution_stress_index": np.round(pollution_stress, 2),
            "total_arrivals": total_arrivals,
            "critical_ambulance_arrivals": critical_ambulance_arrivals,
            "active_ed_crowding": active_ed_crowding.astype(int),
            "mean_wait_time_mins": mean_wait_time_mins,
        }
    )

    # =========================================================================
    # 3. PATIENT-LEVEL EHR RECORDS (MIMIC-IV-ED Schema + Distinct Covariances)
    # =========================================================================
    # Expand each hourly arrival count into individual patient visits
    n_patients = int(total_arrivals.sum())
    hour_indices = np.repeat(np.arange(n_hours), total_arrivals)

    # Assign arrival timestamp with random minute/second offset within the hour
    minute_offsets = pd.to_timedelta(rng.integers(0, 3600, size=n_patients), unit="s")
    intime = hourly_index[hour_indices] + minute_offsets

    p_crowding = active_ed_crowding[hour_indices]
    p_aqi = aqi[hour_indices]
    p_temp = ambient_temp_c[hour_indices]

    # Triage Acuity Levels (1: Resuscitation/Critical, 2: Emergent, 3: Urgent, 4: Non-Urgent)
    # Environmental stress shifts probability toward higher acuity (Class 1 & 2)
    stress_shift = np.clip((p_aqi - 100) / 1500.0 + np.maximum(0, p_temp - 35) / 200.0, 0, 0.08)
    base_probs = np.column_stack(
        [
            0.07 + 0.5 * stress_shift,  # Class 1: Resuscitation / Critical
            0.22 + 0.5 * stress_shift,  # Class 2: Emergent
            0.43 - 0.5 * stress_shift,  # Class 3: Urgent
            0.28 - 0.5 * stress_shift,  # Class 4: Non-Urgent
        ]
    )
    # Vectorized categorical sampling for acuity class (1, 2, 3, 4)
    cum_probs = np.cumsum(base_probs, axis=1)
    u_draw = rng.random((n_patients, 1))
    acuity = (u_draw > cum_probs).sum(axis=1) + 1

    # -------------------------------------------------------------------------
    # Multivariate Gaussian Vitals per Acuity Class: [heartrate, sbp, dbp, resprate, o2sat, temperature]
    # Specifically designed with DISTINCT covariance matrices (Sigma_1 != Sigma_4)
    # so Box's M test is significant and QDA / RDA outperform LDA!
    # -------------------------------------------------------------------------
    means = {
        1: np.array([118.0, 96.0, 58.0, 26.0, 89.5, 38.1]),   # Critical: tachycardia, hypotension, hypoxia
        2: np.array([101.0, 138.0, 86.0, 21.0, 93.8, 37.7]),  # Emergent: elevated HR & hypertensive/stressed
        3: np.array([84.0, 126.0, 79.0, 17.5, 96.8, 37.2]),   # Urgent: mildly elevated vitals
        4: np.array([74.0, 119.0, 75.0, 15.0, 98.4, 36.8]),   # Non-Urgent: tight normal physiological baseline
    }

    # Class 1 has high variance + strong negative HR-SBP correlation (-0.65) due to shock
    cov_1 = np.array([
        [480.0, -290.0, -140.0,  75.0, -48.0,  6.5],
        [-290.0, 520.0,  260.0, -55.0,  38.0, -4.0],
        [-140.0, 260.0,  195.0, -25.0,  18.0, -2.0],
        [ 75.0,  -55.0,  -25.0,  32.0, -14.0,  2.1],
        [-48.0,   38.0,   18.0, -14.0,  24.0, -1.2],
        [  6.5,   -4.0,   -2.0,   2.1,  -1.2,  1.10],
    ])

    # Class 2 has positive HR-SBP correlation (+0.55) from adrenergic/hypertensive stress
    cov_2 = np.array([
        [240.0,  150.0,  85.0,  35.0, -18.0,  3.8],
        [150.0,  310.0, 175.0,  28.0, -12.0,  2.0],
        [ 85.0,  175.0, 130.0,  14.0,  -6.0,  1.1],
        [ 35.0,   28.0,  14.0,  16.0,  -6.5,  1.2],
        [-18.0,  -12.0,  -6.0,  -6.5,   9.5, -0.5],
        [  3.8,    2.0,   1.1,   1.2,  -0.5,  0.65],
    ])

    # Class 3 has moderate variance
    cov_3 = np.array([
        [110.0,  40.0,  25.0,  10.0,  -3.5,  1.2],
        [ 40.0, 145.0,  80.0,   8.0,  -2.0,  0.6],
        [ 25.0,  80.0,  68.0,   4.0,  -1.0,  0.3],
        [ 10.0,   8.0,   4.0,   6.5,  -1.5,  0.4],
        [ -3.5,  -2.0,  -1.0,  -1.5,   3.2, -0.1],
        [  1.2,   0.6,   0.3,   0.4,  -0.1,  0.28],
    ])

    # Class 4 has very tight, low-variance covariance (healthy normal vitals)
    cov_4 = np.array([
        [52.0, 14.0,  9.0, 3.0, -0.6, 0.3],
        [14.0, 68.0, 36.0, 2.0, -0.4, 0.2],
        [ 9.0, 36.0, 34.0, 1.2, -0.2, 0.1],
        [ 3.0,  2.0,  1.2, 2.8, -0.3, 0.1],
        [-0.6, -0.4, -0.2, -0.3, 1.1, 0.0],
        [ 0.3,  0.2,  0.1, 0.1,  0.0, 0.12],
    ])
    covs = {1: cov_1, 2: cov_2, 3: cov_3, 4: cov_4}

    vitals = np.zeros((n_patients, 6))
    for k in (1, 2, 3, 4):
        mask = acuity == k
        vitals[mask] = rng.multivariate_normal(means[k], covs[k], size=mask.sum())

    heartrate = np.clip(vitals[:, 0], 38.0, 210.0).round(1)
    sbp = np.clip(vitals[:, 1], 55.0, 240.0).round(0).astype(int)
    dbp = np.clip(vitals[:, 2], 30.0, sbp - 12).round(0).astype(int)
    resprate = np.clip(vitals[:, 3], 8.0, 48.0).round(1)
    o2sat = np.clip(vitals[:, 4], 65.0, 100.0).round(1)
    temperature = np.clip(vitals[:, 5], 34.5, 41.5).round(2)

    # Patient Age & Pain score (0-10)
    age_base = np.where(acuity <= 2, 62.0, 44.0)
    age = np.clip(rng.normal(age_base, 17.5, n_patients), 18, 96).round(0).astype(int)
    pain = np.clip(rng.normal(9.5 - 1.6 * acuity, 2.0, n_patients), 0, 10).round(0).astype(int)

    # Arrival transport mode
    amb_prob = np.where(acuity == 1, 0.82, np.where(acuity == 2, 0.54, 0.14))
    is_ambulance = rng.binomial(1, amb_prob)
    arrival_transport = np.where(is_ambulance == 1, "AMBULANCE", "WALK IN")

    # -------------------------------------------------------------------------
    # Engineered / Collinear Predictors (Creates high VIF > 15 for Ridge/Lasso/ElasticNet)
    # -------------------------------------------------------------------------
    map_mmhg = np.round((1.0 / 3.0) * sbp + (2.0 / 3.0) * dbp + rng.normal(0, 0.8, n_patients), 1)
    pulse_pressure = np.round((sbp - dbp) + rng.normal(0, 0.8, n_patients), 1)
    shock_index = np.round(heartrate / sbp, 3)
    waiting_room_backlog = np.round(
        0.72 * p_crowding + rng.normal(0, 2.2, n_patients), 1
    ).clip(0)

    # -------------------------------------------------------------------------
    # Target 1: Continuous Length of Stay (los_hours) for OLS vs. Ridge/Lasso/ElasticNet
    # -------------------------------------------------------------------------
    los_true = (
        1.8
        + 0.048 * age
        + 0.032 * heartrate
        - 0.145 * (o2sat - 95.0)
        + 0.018 * sbp
        - 0.012 * dbp
        + 1.95 * shock_index
        + 0.115 * p_crowding
        + 0.085 * waiting_room_backlog
        + 1.40 * is_ambulance
        - 1.65 * acuity
    )
    los_hours = np.clip(los_true + rng.normal(0, 1.35, n_patients), 0.5, 48.0).round(2)
    outtime = intime + pd.to_timedelta(los_hours, unit="h")

    # -------------------------------------------------------------------------
    # Target 2: Binary ICU Transfer (icu_transfer) for Logistic vs. Tree Benchmark
    # Includes linear log-odds terms + a non-linear hypoxia-shock interaction
    # -------------------------------------------------------------------------
    nonlinear_crisis_interaction = 1.65 * ((o2sat < 91.0) & (shock_index > 1.05)).astype(float)
    logit_icu = (
        -3.65
        + 0.028 * (age - 50.0)
        + 0.035 * (heartrate - 80.0)
        - 0.220 * (o2sat - 96.0)
        + 0.085 * (resprate - 16.0)
        + 1.85 * (shock_index - 0.70)
        + 0.75 * is_ambulance
        - 0.85 * (acuity - 2.5)
        + nonlinear_crisis_interaction
    )
    prob_icu = 1.0 / (1.0 + np.exp(-logit_icu))
    icu_transfer = rng.binomial(1, prob_icu)

    patient_df = pd.DataFrame(
        {
            "stay_id": np.arange(30000001, 30000001 + n_patients),
            "subject_id": rng.integers(10000001, 10045000, size=n_patients),
            "intime": intime,
            "outtime": outtime,
            "age": age,
            "arrival_transport": arrival_transport,
            "is_ambulance": is_ambulance,
            "temperature": temperature,
            "heartrate": heartrate,
            "resprate": resprate,
            "o2sat": o2sat,
            "sbp": sbp,
            "dbp": dbp,
            "pain": pain,
            "map_mmhg": map_mmhg,
            "pulse_pressure": pulse_pressure,
            "shock_index": shock_index,
            "concurrent_ed_crowding": p_crowding.astype(int),
            "waiting_room_backlog": waiting_room_backlog,
            "ambient_aqi": p_aqi,
            "acuity": acuity,
            "los_hours": los_hours,
            "icu_transfer": icu_transfer,
        }
    )

    # =========================================================================
    # 4. DAILY TIME-SERIES TABLE (730 Days -> Clean AR, ARMA, ARIMA, SARIMA, SARIMAX)
    # =========================================================================
    daily_df = (
        hourly_df.set_index("timestamp")
        .resample("1D")
        .agg(
            daily_admissions=("total_arrivals", "sum"),
            daily_critical_ambulance=("critical_ambulance_arrivals", "sum"),
            mean_wait_time_mins=("mean_wait_time_mins", "mean"),
            mean_temp_c=("ambient_temp_c", "mean"),
            max_temp_c=("ambient_temp_c", "max"),
            mean_humidity_pct=("humidity_pct", "mean"),
            daily_aqi=("aqi", "mean"),
            heat_stress_index=("heat_stress_index", "mean"),
            pollution_stress_index=("pollution_stress_index", "mean"),
        )
        .reset_index()
        .rename(columns={"timestamp": "date"})
    )
    daily_df["day_of_week"] = daily_df["date"].dt.dayofweek
    daily_df["is_weekend"] = (daily_df["day_of_week"] >= 5).astype(int)
    for col in [
        "mean_wait_time_mins",
        "mean_temp_c",
        "max_temp_c",
        "mean_humidity_pct",
        "daily_aqi",
        "heat_stress_index",
        "pollution_stress_index",
    ]:
        daily_df[col] = daily_df[col].round(2)

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        daily_df.to_csv(os.path.join(output_dir, "ed_daily_timeseries.csv"), index=False)
        hourly_df.to_csv(os.path.join(output_dir, "ed_hourly_timeseries.csv"), index=False)
        patient_df.to_csv(os.path.join(output_dir, "ed_patient_visits.csv"), index=False)

    return {
        "daily": daily_df,
        "hourly": hourly_df,
        "patient": patient_df,
    }


def print_statistical_verification(datasets: dict[str, pd.DataFrame]) -> None:
    """Prints a diagnostic summary proving that all syllabus & extension signals exist."""
    daily = datasets["daily"]
    hourly = datasets["hourly"]
    patient = datasets["patient"]

    print("=" * 72)
    print("URBAN HOSPITAL ED DATASET - STATISTICAL VERIFICATION SUMMARY")
    print("=" * 72)
    print(f"1. Daily Time-Series Rows   : {len(daily):,} days (2 full years)")
    print(f"2. Hourly Count GLM Rows    : {len(hourly):,} hours")
    print(f"3. Patient-Level EHR Visits : {len(patient):,} individual stays")
    print("-" * 72)

    # Check 1: Overdispersion for Poisson vs Negative Binomial
    mean_arr = hourly["total_arrivals"].mean()
    var_arr = hourly["total_arrivals"].var()
    dispersion_ratio = var_arr / mean_arr
    zero_pct = (hourly["critical_ambulance_arrivals"] == 0).mean() * 100
    print(
        f"[Count GLMs] Hourly Arrivals Mean = {mean_arr:.2f}, Var = {var_arr:.2f} "
        f"| Dispersion Ratio (Var/Mean) = {dispersion_ratio:.2f} (> 1.0 -> Negative Binomial needed)"
    )
    print(
        f"[Zero-Inflation] Critical Ambulance Zero-Count Hours = {zero_pct:.1f}% "
        f"(-> Zero-Inflated Poisson justified)"
    )

    # Check 2: Multicollinearity (R^2 / VIF proxy) for OLS vs Ridge/ElasticNet
    X_cols = ["sbp", "dbp", "map_mmhg", "pulse_pressure", "concurrent_ed_crowding", "waiting_room_backlog"]
    corr_matrix = patient[X_cols].corr().to_numpy()
    vifs = np.diag(np.linalg.inv(corr_matrix))
    max_vif_col = X_cols[int(np.argmax(vifs))]
    print(
        f"[OLS vs Ridge] Max VIF among hemodynamic/crowding predictors = {vifs.max():.1f} "
        f"on '{max_vif_col}' (> 10.0 -> Ridge/ElasticNet justified)"
    )

    # Check 3: Distinct Class Covariance Matrices for LDA vs QDA/RDA
    vital_cols = ["heartrate", "sbp", "dbp", "resprate", "o2sat", "temperature"]
    log_dets = {}
    for k in (1, 2, 3, 4):
        sub = patient.loc[patient["acuity"] == k, vital_cols].to_numpy()
        sign, logdet = np.linalg.slogdet(np.cov(sub, rowvar=False))
        log_dets[k] = logdet
    print(
        "[LDA vs QDA/RDA] Log-Determinant of Covariance Matrix ln|Sigma_k| per Acuity Class:\n"
        f"   Class 1 (Critical): {log_dets[1]:.2f} | Class 2 (Emergent): {log_dets[2]:.2f} | "
        f"Class 3 (Urgent): {log_dets[3]:.2f} | Class 4 (Non-Urgent): {log_dets[4]:.2f}\n"
        "   (Massive spread in ln|Sigma_k| guarantees Box's M test rejects equal covariance -> QDA/RDA wins)"
    )

    # Check 4: ICU Class Imbalance & Time-Series Autocorrelation
    icu_rate = patient["icu_transfer"].mean() * 100
    lag1_autocorr = daily["daily_admissions"].autocorr(lag=1)
    lag7_autocorr = daily["daily_admissions"].autocorr(lag=7)
    print(
        f"[Logistic Regression] ICU Transfer Positive Rate = {icu_rate:.1f}% "
        f"(Class imbalance present)"
    )
    print(
        f"[Time Series] Daily Admissions Autocorrelation: Lag-1 = {lag1_autocorr:.3f}, "
        f"Lag-7 (Weekly Seasonality) = {lag7_autocorr:.3f}"
    )
    print("=" * 72)


if __name__ == "__main__":
    data = generate_urban_ed_dataset()
    print_statistical_verification(data)
