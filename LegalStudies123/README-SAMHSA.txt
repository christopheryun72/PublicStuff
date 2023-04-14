- EDA_pt1_samhsa.pdf: Part 1 of EDA process
    - Preliminary Initial feature selection
    - Separation of data into normal distributed,
    non-normal binary distributed, non-normal multinomial
    distributed data.
    - Translation of feature values to NaN (ex: 'M'->np.nan)
    - Imputation of np.nan values to imputed values
    - Outputs the three subset datasets described above
    (normal distributed denoted by 'main_samhsa',
    non-normal binary distributed denoted by 'ill_dist_bin_samhsa.csv',
    non-normal multinomial distributed denoted by 'ill_dist_non_samhsa.csv').
    These datasets are used as input into the second EDA SAMHSA script
    described below
- EDA_pt2_samhsa.pdf: Part 2 of EDA process
    - Takes as input the three SAMHSA data subsets outputted from
    the EDA-pt1-samhsa.ipynb notebook.
    - Normalization of non-normal multinomial distributed features
    - Combining all three subsets into one SAMHSA dataset
    - Compute correlation matrices and identifying which features
    aren't collinear but are still correlated.
    - Multiple graphs grouped by year and grouped by state
- state_classes.npy: Numpy file containing labelencoder classes for the 'State' feature
