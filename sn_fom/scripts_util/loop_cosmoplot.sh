#!/bin/bash


# 10 years
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_10years/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_2_2.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_2_2_Ny_40 --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_10years
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_10years/Fit_bias_Ny_40_Om_w0 --config  config_cosmoSN_deep_rolling_2_2_mini.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_Ny_40 --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_10years
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_10years/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_universal_10.csv --runtype universal --outName cosmoSN_universal_10_Ny_40 --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_10years

#nominal - yearly

#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_x1c_1sigma/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype universal --outName cosmoSN_universal_yearly_Ny_40  --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_yearly_x1c_1sigma
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_x1c_1sigma/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40 --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_yearly_x1c_1sigma
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_x1c_1sigma/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40 --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_yearly_x1c_1sigma

#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_zspectro_x1c_1sigma/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype universal --outName cosmoSN_universal_yearly_Ny_40 --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_yearly_zspectro_x1c_1sigma
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_zspectro_x1c_1sigma/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40 --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_yearly_zspectro_x1c_1sigma
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_zspectro_x1c_1sigma/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40  --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_yearly_zspectro_x1c_1sigma

python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_zspectro_x1c_1sigma_uy_800/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype universal --outName cosmoSN_universal_yearly_Ny_40 --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_yearly_zspectro_x1c_1sigma_uy_800
python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_zspectro_x1c_1sigma_uy_800/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40 --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_yearly_zspectro_x1c_1sigma_uy_800
python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_zspectro_x1c_1sigma_uy_800/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40  --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_yearly_zspectro_x1c_1sigma_uy_800

#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_new_x1c_2sigma/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype universal --outName cosmoSN_universal_yearly_Ny_40  --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_yearly_new_x1c_2sigma
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_new_x1c_2sigma/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40 --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_yearly_new_x1c_2sigma
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_new_x1c_2sigma/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40 --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_yearly_new_x1c_2sigma

#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_nosyste/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype yearly --outName cosmoSN_universal_yearly_Ny_40 --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_yearly_nosyste
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_nosyste/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40 --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_yearly_nosyste
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_nosyste/Fit_bias_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40  --outDir /home/philippe/LSST/sn_dd_opti/cosmo_files_yearly_nosyste



# bias 2 sigmas
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_10years_WFD/Fit_bias_bias_2sigma_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype universal --outName cosmoSN_universal_yearly_Ny_40_bias_2_sigma
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_10years_WFD/Fit_bias_bias_2sigma_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40_bias_2_sigma
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly_10years_WFD/Fit_bias_bias_2sigma_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40_bias_2_sigma

# photz 0.02
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_photz_02_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype universal  --outName cosmoSN_universal_yearly_Ny_40_photz_02
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_photz_02_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40_photz_02
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_photz_02_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40_photz_02

# photz 0.002
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_photz_002_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype universal  --outName cosmoSN_universal_yearly_Ny_40_photz_002
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_photz_002_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40_photz_002
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_photz_002_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40_photz_002

# photz 0.01
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_photz_01_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype universal  --outName cosmoSN_universal_yearly_Ny_40_photz_01
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_photz_01_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40_photz_01
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_photz_01_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40_photz_01

# photz 0.015
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_photz_015_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype universal  --outName cosmoSN_universal_yearly_Ny_40_photz_015
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_photz_015_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40_photz_015
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_photz_015_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40_photz_015

# NSN+1sigma
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_nsn_bias_plus_1_sigma_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype universal  --outName cosmoSN_universal_yearly_Ny_40_nsn_plus_sigma
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_nsn_bias_plus_1_sigma_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40_nsn_plus_sigma
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_nsn_bias_plus_1_sigma_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40_nsn_plus_sigma

# NSN-1sigma
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_nsn_bias_minus_1_sigma_Ny_40_Om_w0 --config config_cosmoSN_universal_yearly.csv --runtype universal  --outName cosmoSN_universal_yearly_Ny_40_nsn_minus_sigma
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_nsn_bias_minus_1_sigma_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_2_2_mini_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40_nsn_minus_sigma
#python sn_studies/sn_fom/cosmo_plot.py --fileDir cosmofit_yearly/Fit_bias_nsn_bias_minus_1_sigma_Ny_40_Om_w0 --config config_cosmoSN_deep_rolling_0.80_0.80_yearly.csv --runtype deep_rolling --outName cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40_nsn_minus_sigma
