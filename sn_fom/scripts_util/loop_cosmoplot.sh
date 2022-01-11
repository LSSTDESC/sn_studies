#!/bin/bash


# 10 years
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_2_2.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_2_2_Ny_40


#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_Ny_40_Om_w0 --config  config_cosmoSN_deep_rolling_2_2_mini.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_Ny_40

#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_universal_10.csv --runtype universal --outName cosmoSN_universal_10_Ny_40

#nominal

#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype universal --outName cosmoSN_universal_yearly_Ny_40
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40

# bias 2 sigmas
#python sn_studies/sn_fom/cosmo_plot.py --fileDir fake/Fit_bias_bias_2_sigma_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype universal --outName cosmoSN_universal_yearly_Ny_40_bias_2_sigma
#python sn_studies/sn_fom/cosmo_plot.py --fileDir fake/Fit_bias_bias_2_sigma_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40_bias_2_sigma
#python sn_studies/sn_fom/cosmo_plot.py --fileDir fake/Fit_bias_bias_2_sigma_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40_bias_2_sigma

# photz 0.02
python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_photz_02_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype universal  --outName cosmoSN_universal_yearly_Ny_40_photz_02
python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_photz_02_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40_photz_02
python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_photz_02_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40_photz_02

# photz 0.002
python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_photz_002_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype universal  --outName cosmoSN_universal_yearly_Ny_40_photz_002
python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_photz_002_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40_photz_002
python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_photz_002_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40_photz_002

# NSN+1sigma
python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_nsn_bias_plus_1_sigma_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype universal  --outName cosmoSN_universal_yearly_Ny_40_nsn_plus_sigma
python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_nsn_bias_plus_1_sigma_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40_nsn_plus_sigma
python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_nsn_bias_plus_1_sigma_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40_nsn_plus_sigma

# NSN-1sigma
python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_nsn_bias_minus_1_sigma_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype universal  --outName cosmoSN_universal_yearly_Ny_40_nsn_minus_sigma
python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_nsn_bias_minus_1_sigma_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40_nsn_minus_sigma
python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit/Fit_bias_nsn_bias_minus_1_sigma_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40_nsn_minus_sigma
