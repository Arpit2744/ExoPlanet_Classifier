import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# ====================================================================
# FINAL MODEL FEATURE DEFINITION (27 FEATURES)
# ====================================================================
FINAL_MODEL_FEATURES = [
    'TransitEpoch_BKJD',
    'ImpactParameter',
    'TransitDuration_hrs',
    'EquilibriumTemperature_K',
    'TransitSignal_to_Noise',
    'StellarEffectiveTemperature_K',
    'StellarSurfaceGravity',
    'Kepler_bandmag',
    'orbital_period_uncertainty',
    'transit_duration_uncertainty',
    'transit_depth_uncertainty',
    'planetary_radius_uncertainty',
    'insolation_flux_uncertainty',
    'impact_parameter_uncertainty',
    'stellar_temp_uncertainty',
    'stellar_gravity_uncertainty',
    'stellar_radius_uncertainty',
    'transit_epoch_uncertainty',
    'transit_frequency',
    'planet_to_star_radius_ratio',
    'planet_to_star_density_ratio',
    'transit_duration_to_period_ratio',
    'orbital_speed_estimate',
    'combined_uncertainty_score',
    'log_OrbitalPeriod_days',
    'log_InsolationFlux_Earthflux',
    'log_TransitDepth_ppm'
]

# Mapping dictionary for raw data columns (Kepler KOI or TESS TOI)
RAW_COLUMN_MAPPING = {
    # Base Features
    'koi_period': 'OrbitalPeriod_days', 'pl_orbper': 'OrbitalPeriod_days', 
    'koi_time0bk': 'TransitEpoch_BKJD', 'pl_tranmid': 'TransitEpoch_BKJD',
    'koi_impact': 'ImpactParameter', 'pl_imppar': 'ImpactParameter',
    'koi_duration': 'TransitDuration_hrs', 'pl_trandurh': 'TransitDuration_hrs',
    'koi_teq': 'EquilibriumTemperature_K', 'pl_eqt': 'EquilibriumTemperature_K',
    'koi_model_snr': 'TransitSignal_to_Noise', 'snr': 'TransitSignal_to_Noise',
    'koi_steff': 'StellarEffectiveTemperature_K', 'st_teff': 'StellarEffectiveTemperature_K',
    'koi_slogg': 'StellarSurfaceGravity', 'st_logg': 'StellarSurfaceGravity',
    'koi_kepmag': 'Kepler_bandmag', 'st_tmag': 'Kepler_bandmag', 
    'koi_srad': 'StellarRadius_Solarradii', 'st_rad': 'StellarRadius_Solarradii',
    'koi_prad': 'PlanetaryRadius_Earthradii', 'pl_rade': 'PlanetaryRadius_Earthradii',
    'koi_depth': 'TransitDepth_ppm', 'pl_trandep': 'TransitDepth_ppm',
    'koi_insol': 'InsolationFlux_Earthflux', 'pl_insol': 'InsolationFlux_Earthflux',
    
    # Uncertainty Columns (used to create the final uncertainty features)
    'koi_period_err1': 'OrbitalPeriodUpper_days', 'koi_period_err2': 'OrbitalPeriodLower_days',
    'pl_orbpererr1': 'OrbitalPeriodUpper_days', 'pl_orbpererr2': 'OrbitalPeriodLower_days',
    
    'koi_time0bk_err1': 'TransitEpoch_Upper', 'koi_time0bk_err2': 'TransitEpoch_Lower',
    'pl_tranmiderr1': 'TransitEpoch_Upper', 'pl_tranmiderr2': 'TransitEpoch_Lower',
    
    'koi_duration_err1': 'TransitDuration_Upper', 'koi_duration_err2': 'TransitDuration_Lower',
    'pl_trandurherr1': 'TransitDuration_Upper', 'pl_trandurherr2': 'TransitDuration_Lower',
    
    'koi_depth_err1': 'TransitDepth_Upper', 'koi_depth_err2': 'TransitDepth_Lower',
    'pl_trandeperr1': 'TransitDepth_Upper', 'pl_trandeperr2': 'TransitDepth_Lower',
    
    'koi_prad_err1': 'PlanetaryRadius_Upper', 'koi_prad_err2': 'PlanetaryRadius_Lower',
    'pl_radeerr1': 'PlanetaryRadius_Upper', 'pl_radeerr2': 'PlanetaryRadius_Lower',
    
    'koi_insol_err1': 'InsolationFlux_Upper', 'koi_insol_err2': 'InsolationFlux_Lower',
    'pl_insolerr1': 'InsolationFlux_Upper', 'pl_insolerr2': 'InsolationFlux_Lower',
    
    'koi_impact_err1': 'ImpactParameter_Upper', 'koi_impact_err2': 'ImpactParameter_Lower',
    'pl_impparerr1': 'ImpactParameter_Upper', 'pl_impparerr2': 'ImpactParameter_Lower',

    'koi_steff_err1': 'StellarEffectiveTemp_Upper', 'koi_steff_err2': 'StellarEffectiveTemp_Lower',
    'st_tefferr1': 'StellarEffectiveTemp_Upper', 'st_tefferr2': 'StellarEffectiveTemp_Lower',

    'koi_slogg_err1': 'Stellar_SurfaceGravity_Upper', 'koi_slogg_err2': 'Stellar_SurfaceGravity_Lower',
    'st_loggerr1': 'Stellar_SurfaceGravity_Upper', 'st_loggerr2': 'Stellar_SurfaceGravity_Lower',

    'koi_srad_err1': 'StellarRadius_Upper', 'koi_srad_err2': 'StellarRadius_Lower',
    'st_raderr1': 'StellarRadius_Upper', 'st_raderr2': 'StellarRadius_Lower',
    
    'koi_teq_err1': 'EquilibriumTemperature_Upper', 'koi_teq_err2': 'EquilibriumTemperature_Lower',
    'pl_eqterr1': 'EquilibriumTemperature_Upper', 'pl_eqterr2': 'EquilibriumTemperature_Lower',
}

# Define all the base features that need to be available for calculations
BASE_FEATURES_USED_IN_ENG = [
    'OrbitalPeriod_days', 'TransitDepth_ppm', 'InsolationFlux_Earthflux',
    'StellarRadius_Solarradii', 'PlanetaryRadius_Earthradii', 'StellarSurfaceGravity',
    'TransitDuration_hrs', 'TransitEpoch_BKJD', 'ImpactParameter', 
    'EquilibriumTemperature_K', 'TransitSignal_to_Noise', 'StellarEffectiveTemperature_K', 
    'Kepler_bandmag'
]

def preprocess_data(raw_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the full preprocessing and feature engineering pipeline to raw 
    TESS or KOI data to prepare it for LightGBM prediction.
    """
    df = raw_data_df.copy()

    # 1. RENAME COLUMNS and ensure all necessary base columns exist
    df = df.rename(columns=RAW_COLUMN_MAPPING, errors='ignore')

    # Ensure base features required for engineering exist (fill non-existing with NaN)
    for col in BASE_FEATURES_USED_IN_ENG:
        if col not in df.columns:
            df[col] = np.nan

    # Define features used in multiple steps
    BASE_PERIOD = 'OrbitalPeriod_days'
    BASE_RADIUS = 'PlanetaryRadius_Earthradii'
    BASE_SRAD = 'StellarRadius_Solarradii'
    BASE_TDEP = 'TransitDepth_ppm'
    BASE_TDUR = 'TransitDuration_hrs'
    BASE_INSOL = 'InsolationFlux_Earthflux'
    
    epsilon = 1e-6 # For safe division

    # 2. FEATURE ENGINEERING (Recreate all custom features)
    
    # A. Calculate Uncertainty Features (Upper - Lower)
    UNCERT_MAPS = {
        'orbital_period_uncertainty': ('OrbitalPeriodUpper_days', 'OrbitalPeriodLower_days'),
        'transit_duration_uncertainty': ('TransitDuration_Upper', 'TransitDuration_Lower'),
        'transit_depth_uncertainty': ('TransitDepth_Upper', 'TransitDepth_Lower'),
        'planetary_radius_uncertainty': ('PlanetaryRadius_Upper', 'PlanetaryRadius_Lower'),
        'insolation_flux_uncertainty': ('InsolationFlux_Upper', 'InsolationFlux_Lower'),
        'impact_parameter_uncertainty': ('ImpactParameter_Upper', 'ImpactParameter_Lower'),
        'stellar_temp_uncertainty': ('StellarEffectiveTemp_Upper', 'StellarEffectiveTemp_Lower'),
        'stellar_gravity_uncertainty': ('Stellar_SurfaceGravity_Upper', 'Stellar_SurfaceGravity_Lower'),
        'stellar_radius_uncertainty': ('StellarRadius_Upper', 'StellarRadius_Lower'),
        'transit_epoch_uncertainty': ('TransitEpoch_Upper', 'TransitEpoch_Lower'),
    }

    uncert_feature_names = []
    for new_col, (upper, lower) in UNCERT_MAPS.items():
        if upper in df.columns and lower in df.columns:
            df[new_col] = np.abs(df[upper] - df[lower])
        else:
            df[new_col] = np.nan
        uncert_feature_names.append(new_col)
        
    # B. Combined Uncertainty Score (Mean of all 10 uncertainties)
    df['combined_uncertainty_score'] = df[uncert_feature_names].mean(axis=1)

    # C. Composite & Ratio Features (Ensure base columns exist before calculation)
    df['transit_frequency'] = 1 / (df[BASE_PERIOD] + epsilon)
    df['planet_to_star_radius_ratio'] = df[BASE_RADIUS] / (df[BASE_SRAD] * 109.2 + epsilon)
    df['planet_to_star_density_ratio'] = (df[BASE_RADIUS]**3) / (df[BASE_SRAD]**3 + epsilon)
    df['transit_duration_to_period_ratio'] = df[BASE_TDUR] / (df[BASE_PERIOD] + epsilon)
    df['orbital_speed_estimate'] = df[BASE_SRAD] / (df[BASE_TDUR] + epsilon)

    # D. Logarithmic Transforms (np.log1p)
    df['log_OrbitalPeriod_days'] = np.log1p(df[BASE_PERIOD]) 
    df['log_InsolationFlux_Earthflux'] = np.log1p(df[BASE_INSOL])
    df['log_TransitDepth_ppm'] = np.log1p(df[BASE_TDEP])
    
    # 3. FINAL FILTERING AND IMPUTATION
    # Ensure all final features exist (fill with NaN if missing after all calcs)
    for feature in FINAL_MODEL_FEATURES:
        if feature not in df.columns:
            df[feature] = np.nan

    df_final = df[FINAL_MODEL_FEATURES]

    # Impute remaining missing values using the median.
    imputer = SimpleImputer(strategy='median')
    df_final_imputed = pd.DataFrame(imputer.fit_transform(df_final), columns=FINAL_MODEL_FEATURES)

    return df_final_imputed
