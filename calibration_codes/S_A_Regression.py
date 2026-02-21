import pandas as pd
import statsmodels.api as sm

# Load the experiment results CSV file.
results_df = pd.read_csv("experiment_results.csv")

# Identify predictor columns (all except 'objective')
parameter_columns = [col for col in results_df.columns if col != "objective"]

# Prepare the data for regression.
X = results_df[parameter_columns]
y = results_df["objective"]

# Drop rows with missing data.
X = X.dropna()
y = y[X.index]

# Add constant term.
X_const = sm.add_constant(X)

# Fit OLS model.
model = sm.OLS(y, X_const).fit()

# Save full regression summary (optional but recommended)
with open("OLS_summary.txt", "w") as f:
    f.write(model.summary().as_text())

# Filtering criteria
p_threshold = 0.05
coef_threshold = 1

filtered_params = []

for param in parameter_columns:
    coef = model.params[param]
    pval = model.pvalues[param]
    tval = model.tvalues[param]          # <-- t-statistic
    stderr = model.bse[param]            # <-- standard error
    
    if (pval < p_threshold) and (abs(coef) > coef_threshold):
        filtered_params.append({
            "Parameter": param,
            "Coefficient": coef,
            "Std_Error": stderr,
            "t_statistic": tval,
            "Abs_Coefficient": abs(coef),
            "P_value": pval
        })

# Create DataFrame
filtered_df = pd.DataFrame(filtered_params)

# Sort by absolute coefficient magnitude
filtered_df = filtered_df.sort_values(by="Abs_Coefficient", ascending=False)

# Save table
filtered_df.to_csv("filtered_parameters_SA.csv", index=False)

print("\nFiltered Parameters (Sensitivity Analysis):")
print(filtered_df)
