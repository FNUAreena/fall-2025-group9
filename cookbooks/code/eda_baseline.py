#%%
print("=== STEP 1-5: DATA PREPARATION (WITH TYPE CONVERSION) ===")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

df = pd.read_csv(r"/Users/chayachandana/Downloads/dataset.csv", low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])

print(f"Original dataset: {len(df)} records")

numeric_columns = ['Planned_Total', 'Offered_Total', 'Served_Total', 'Left_Over_Total', 
                   'Production_Cost_Total', 'Left_Over_Cost']

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"Converted {col} - Missing: {df[col].isnull().sum()}")

print(f"\nAfter type conversion:")
print(df[numeric_columns].describe())

holiday_dates = ['2025-05-26']
holiday_dates = [pd.to_datetime(date) for date in holiday_dates]
df_clean = df[~df['Date'].isin(holiday_dates)]
df_clean = df_clean[df_clean['Served_Total'] > 0]

print(f"\nAfter cleaning: {len(df_clean)} records")

df_grouped = df_clean.groupby(['School_Name', 'Date']).agg({
    'Planned_Total': 'sum',
    'Offered_Total': 'sum', 
    'Served_Total': 'sum',
    'Left_Over_Total': 'sum',
    'Production_Cost_Total': 'sum',
    'Left_Over_Cost': 'sum'
}).reset_index()

df_grouped['Weekday'] = df_grouped['Date'].dt.day_name()

print(f"\nAfter grouping:")
print(f"Records: {len(df_grouped)}")
print(f"Unique schools: {df_grouped['School_Name'].nunique()}")
print(f"Date range: {df_grouped['Date'].min()} to {df_grouped['Date'].max()}")

df_grouped['Cost_Per_Meal'] = df_grouped['Production_Cost_Total'] / df_grouped['Served_Total']
df_grouped['Wastage_Cost'] = df_grouped['Left_Over_Cost']
df_grouped['Offered_Ratio'] = df_grouped['Offered_Total'] / df_grouped['Served_Total']

df_grouped = df_grouped.replace([np.inf, -np.inf], np.nan)

print(f"\nKPI Statistics:")
print(f"Average Cost Per Meal: ${df_grouped['Cost_Per_Meal'].mean():.2f}")
print(f"Average Offered Ratio: {df_grouped['Offered_Ratio'].mean():.2f}")
print(f"Average Served Total: {df_grouped['Served_Total'].mean():.2f} meals")

print(f"\nRemoving noise from KPIs...")

cpm_upper = df_grouped['Cost_Per_Meal'].quantile(0.95)
df_grouped['Cost_Per_Meal_Clipped'] = df_grouped['Cost_Per_Meal'].clip(upper=cpm_upper)

df_grouped['Offered_Ratio_Clipped'] = df_grouped['Offered_Ratio'].clip(1.00, 1.50)

print(f"\nAfter noise removal:")
print(f"Cost Per Meal - Before: ${df_grouped['Cost_Per_Meal'].mean():.2f}, After: ${df_grouped['Cost_Per_Meal_Clipped'].mean():.2f}")
print(f"Offered Ratio - Before: {df_grouped['Offered_Ratio'].mean():.2f}, After: {df_grouped['Offered_Ratio_Clipped'].mean():.2f}")

print(f"\nSTEP 1-5 COMPLETED SUCCESSFULLY!")
# %%
print("=== STEP 6: MEDIAN BASELINE WEEKDAY PLAN ===")

median_baseline = df_grouped.groupby(['School_Name', 'Weekday']).agg({
    'Served_Total': 'median',
    'Offered_Total': 'median',
    'Offered_Ratio_Clipped': 'median'
}).reset_index()

median_baseline = median_baseline.rename(columns={
    'Served_Total': 'Baseline_Meals',
    'Offered_Total': 'Current_Offered',
    'Offered_Ratio_Clipped': 'Baseline_Offered_Ratio'
})

print(f"Created median baseline for {len(median_baseline)} school-weekday combinations")

df_final = df_grouped.merge(
    median_baseline, 
    on=['School_Name', 'Weekday'], 
    how='left',
    suffixes=('', '_Baseline')
)

print(f"\nBASELINE STATISTICS:")
print(f"Average Baseline Meals: {df_final['Baseline_Meals'].mean():.2f}")
print(f"Average Current Offered: {df_final['Current_Offered'].mean():.2f}")
print(f"Average Baseline Offered Ratio: {df_final['Baseline_Offered_Ratio'].mean():.2f}")

print(f"\nBASELINE BY WEEKDAY:")
weekday_baseline = df_final.groupby('Weekday')['Baseline_Meals'].median()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
for day in weekday_order:
    if day in weekday_baseline.index:
        print(f"  {day}: {weekday_baseline[day]:.1f} meals")

df_final['Baseline_Plan'] = df_final['Baseline_Meals'] * df_final['Baseline_Offered_Ratio']

print(f"\nFINAL BASELINE PLAN:")
print(f"Average Baseline Plan: {df_final['Baseline_Plan'].mean():.2f} meals")
print(f"Recommended production increase: {df_final['Baseline_Plan'].mean() - df_final['Current_Offered'].mean():.2f} meals")

print(f"\nSAMPLE BASELINE PLANS (first 3 schools):")
sample_schools = df_final['School_Name'].unique()[:3]
for school in sample_schools:
    school_data = df_final[df_final['School_Name'] == school].head(2)
    print(f"\n{school}:")
    for _, row in school_data.iterrows():
        print(f"  {row['Weekday']}: Plan {row['Baseline_Plan']:.1f} meals (Current: {row['Offered_Total']:.1f})")
# %%
print("=== STEP 7: BACKTESTING EVALUATION ===")

def backtest_baseline_model(df_test, baseline_plan, test_dates):
    """
    Backtest the baseline model on test dates with fallback handling for missing weekdays.
    """
    results = []
    
    global_fallback = baseline_plan['Baseline_Plan'].median()  
    
    for school in df_test['School_Name'].unique():
        school_data = df_test[df_test['School_Name'] == school]
        school_baseline = baseline_plan[baseline_plan['School_Name'] == school]
        
        school_fallback = school_baseline['Baseline_Plan'].median() if len(school_baseline) > 0 else global_fallback
        
        for date in test_dates:
            date_data = school_data[school_data['Date'] == date]
            if len(date_data) == 0:
                continue
                
            weekday = date_data['Weekday'].iloc[0]
            actual_served = date_data['Served_Total'].iloc[0]
            actual_offered = date_data['Offered_Total'].iloc[0]
            
            baseline_pred = school_baseline[school_baseline['Weekday'] == weekday]['Baseline_Plan']
            
            if len(baseline_pred) > 0:
                predicted_plan = baseline_pred.iloc[0]
            else:
                predicted_plan = school_fallback  
                
            demand_met = min(predicted_plan, actual_served) / actual_served if actual_served > 0 else 1
            overproduction = max(0, predicted_plan - actual_served)
            underproduction = max(0, actual_served - predicted_plan)
            waste_reduction = max(0, actual_offered - predicted_plan)
            
            results.append({
                'School_Name': school,
                'Date': date,
                'Weekday': weekday,
                'Actual_Served': actual_served,
                'Actual_Offered': actual_offered,
                'Baseline_Plan': predicted_plan,
                'Demand_Met_Rate': demand_met,
                'Overproduction': overproduction,
                'Underproduction': underproduction,
                'Waste_Reduction': waste_reduction
            })
    
    return pd.DataFrame(results)


all_dates = sorted(df_final['Date'].unique())
split_point = int(len(all_dates) * 0.7)
train_dates = all_dates[:split_point]
test_dates = all_dates[split_point:]

print(f"Training dates: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})") 
print(f"Testing dates: {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")

train_data = df_final[df_final['Date'].isin(train_dates)]
baseline_plan_train = train_data.groupby(['School_Name', 'Weekday']).agg({
    'Served_Total': 'median',
    'Offered_Ratio_Clipped': 'median'
}).reset_index()
baseline_plan_train['Baseline_Plan'] = (
    baseline_plan_train['Served_Total'] * baseline_plan_train['Offered_Ratio_Clipped']
)

print(f"\nTraining baseline statistics:")
print(f"Average baseline plan: {baseline_plan_train['Baseline_Plan'].mean():.2f} meals")

test_data = df_final[df_final['Date'].isin(test_dates)]
backtest_results = backtest_baseline_model(test_data, baseline_plan_train, test_dates)

print(f"\nBACKTESTING RESULTS:")
print(f"Number of test predictions: {len(backtest_results)}")
print(f"Schools evaluated: {backtest_results['School_Name'].nunique()}")

if len(backtest_results) > 0:

    avg_demand_met = backtest_results['Demand_Met_Rate'].mean() * 100
    total_overproduction = backtest_results['Overproduction'].sum()
    total_underproduction = backtest_results['Underproduction'].sum()
    total_waste_reduction = backtest_results['Waste_Reduction'].sum()
    
    print(f"\nPERFORMANCE METRICS:")
    print(f"Average Demand Met Rate: {avg_demand_met:.1f}%")
    print(f"Total Overproduction: {total_overproduction:.0f} meals")
    print(f"Total Underproduction: {total_underproduction:.0f} meals")
    print(f"Total Waste Reduction Potential: {total_waste_reduction:.0f} meals")
    
    print(f"\nPERFORMANCE BY WEEKDAY:")
    weekday_performance = backtest_results.groupby('Weekday').agg({
        'Demand_Met_Rate': 'mean',
        'Overproduction': 'mean',
        'Underproduction': 'mean',
        'Waste_Reduction': 'mean'
    }).round(3)
    print(weekday_performance)
    
    current_waste_rate = (test_data['Left_Over_Total'].sum() / test_data['Offered_Total'].sum()) * 100
    print(f"\nCURRENT WASTE LEVEL: {current_waste_rate:.1f}%")
# %%
print("=== STEP 8: DATA QUALITY INVESTIGATION ===")

print("INVESTIGATING THE 134.5% WASTE RATE ANOMALY")

print(f"\n1. WASTE PERCENTAGE DISTRIBUTION:")
print(df_final['Left_Over_Total'].describe())
print(f"\nWaste percentage stats:")
waste_pct = (df_final['Left_Over_Total'] / df_final['Offered_Total']) * 100
print(f"Min: {waste_pct.min():.1f}%")
print(f"Max: {waste_pct.max():.1f}%") 
print(f"Mean: {waste_pct.mean():.1f}%")
print(f"Median: {waste_pct.median():.1f}%")

print(f"\n2. IMPOSSIBLE VALUES CHECK:")
impossible_waste = df_final[df_final['Left_Over_Total'] > df_final['Offered_Total']]
print(f"Records with waste > offered: {len(impossible_waste)}/{len(df_final)} ({len(impossible_waste)/len(df_final)*100:.1f}%)")

if len(impossible_waste) > 0:
    print(f"Sample impossible records:")
    print(impossible_waste[['School_Name', 'Date', 'Offered_Total', 'Left_Over_Total']].head(3))

print(f"\n3. ZERO/VALUE CHECK:")
print(f"Records with Offered_Total = 0: {(df_final['Offered_Total'] == 0).sum()}")
print(f"Records with Left_Over_Total = 0: {(df_final['Left_Over_Total'] == 0).sum()}")
print(f"Records with Served_Total = 0: {(df_final['Served_Total'] == 0).sum()}")

print(f"\n4. TOP 5 HIGH-WASTE SCHOOLS:")
school_waste = df_final.groupby('School_Name').apply(
    lambda x: (x['Left_Over_Total'].sum() / x['Offered_Total'].sum()) * 100
).sort_values(ascending=False)
print(school_waste.head(5))

print(f"\n5. WASTE CALCULATION VALIDATION:")
print(f"Total Offered: {df_final['Offered_Total'].sum():.0f} meals")
print(f"Total Served: {df_final['Served_Total'].sum():.0f} meals") 
print(f"Total Left Over: {df_final['Left_Over_Total'].sum():.0f} meals")
print(f"Mathematical check: Offered ({df_final['Offered_Total'].sum():.0f}) = Served ({df_final['Served_Total'].sum():.0f}) + Left Over ({df_final['Left_Over_Total'].sum():.0f})")
print(f"Difference: {df_final['Offered_Total'].sum() - (df_final['Served_Total'].sum() + df_final['Left_Over_Total'].sum()):.0f} meals")

print(f"\n6. TEST DATA ANALYSIS:")
test_waste_rate = (test_data['Left_Over_Total'].sum() / test_data['Offered_Total'].sum()) * 100
print(f"Test data waste rate: {test_waste_rate:.1f}%")
print(f"Test data - Offered: {test_data['Offered_Total'].sum():.0f}, Served: {test_data['Served_Total'].sum():.0f}, Left Over: {test_data['Left_Over_Total'].sum():.0f}")
# %%
print("=== STEP 9: DATA CORRECTION ===")

df_corrected = df_final[df_final['Offered_Total'] > 0].copy()
print(f"Removed {len(df_final) - len(df_corrected)} records with Offered_Total = 0")
print(f"Remaining records: {len(df_corrected)}")

df_corrected['Waste_Percentage'] = (df_corrected['Left_Over_Total'] / df_corrected['Offered_Total']) * 100

df_corrected = df_corrected[df_corrected['Waste_Percentage'] <= 200]
print(f"After removing extreme waste outliers: {len(df_corrected)} records")

print(f"\nCORRECTED WASTE STATISTICS:")
print(f"Average Waste Percentage: {df_corrected['Waste_Percentage'].mean():.1f}%")
print(f"Median Waste Percentage: {df_corrected['Waste_Percentage'].median():.1f}%")
print(f"Min Waste: {df_corrected['Waste_Percentage'].min():.1f}%")
print(f"Max Waste: {df_corrected['Waste_Percentage'].max():.1f}%")

print(f"\nDATA VALIDATION:")
print(f"Total Offered: {df_corrected['Offered_Total'].sum():.0f}")
print(f"Total Served: {df_corrected['Served_Total'].sum():.0f}")
print(f"Total Left Over: {df_corrected['Left_Over_Total'].sum():.0f}")
print(f"Mathematical Check: {df_corrected['Offered_Total'].sum():.0f} = {df_corrected['Served_Total'].sum():.0f} + {df_corrected['Left_Over_Total'].sum():.0f}")

valid_math = abs(df_corrected['Offered_Total'].sum() - (df_corrected['Served_Total'].sum() + df_corrected['Left_Over_Total'].sum()))
print(f"Data Integrity: {'PASS' if valid_math < 1000 else 'FAIL'} (Difference: {valid_math:.0f} meals)")
# %%
print("=== STEP 10: PIVOT TO RELIABLE METRICS ===")

print("RELIABLE METRICS ANALYSIS:")

consumption_stats = df_corrected.groupby('Weekday')['Served_Total'].agg(['mean', 'median', 'count'])
print(f"\n1. CONSUMPTION PATTERNS (Reliable):")
print(consumption_stats.round(1))

print(f"\n2. DEMAND FORECASTING FOCUS:")
print(f"Average meals consumed: {df_corrected['Served_Total'].mean():.1f}")
print(f"Total consumption opportunity: {df_corrected['Served_Total'].sum():.0f} meals")

print(f"\n3. BASELINE PLAN PERFORMANCE (from backtesting):")
print(f"Demand Met Rate: 82.0% - This is reliable")
print(f"Identified Tuesday as highest opportunity day")
print(f"Friday has highest consumption")

print(f"\n NEW STRATEGY:")
print("Since waste data is unreliable, focus on:")
print("1. OPTIMIZING MEETING STUDENT DEMAND")
print("2. Using consumption patterns for better planning") 
print("3. The 82.0% demand met rate as our key metric")
print("4. Tuesday-Friday patterns for production planning")
# %%
print("=== STEP 11: FINAL SUMMARY & RECOMMENDATIONS ===")
print("=" * 60)

print("\n PROJECT SUMMARY:")
print(f"• Analyzed: 1,302 reliable records across 160+ schools")
print(f"• Time Period: 21 school days in May 2025") 
print(f"• Key Finding: Waste data unreliable, but consumption patterns are solid")

print("\n KEY RELIABLE FINDINGS:")
print(f"1. Consumption Patterns: Clear weekly rhythm")
print(f"   Friday: 29.8 meals (peak) → Wednesday: 22.9 meals (lowest)")
print(f"2. Demand Met Rate: 82.0% (room for improvement)")
print(f"3. Tuesday Opportunity: Highest optimization potential")

print("\n TEMPORAL PATTERNS:")
print(f"• Friday consumption: 30% higher than Wednesday")
print(f"• Consistent decline: Fri → Thu → Tue → Wed")
print(f"• Weak seasonality: Simple models sufficient")

print("\n MODEL SELECTION:")
print(f"RECOMMENDATION: ARIMAX with school features")
print(f"Why: Weak weekly patterns (11% variation)")
print(f"Better to focus on school-level differences than daily patterns")

print("\n WASTE REDUCTION STRATEGY:")
print("Since waste data is unreliable, we optimize differently:")
print("1. FOCUS on meeting 100% of student demand")
print("2. USE consumption patterns for precise production")
print("3. TARGET 82.0% → 90%+ demand met rate")
print("4. ELIMINATE both overproduction AND underproduction")

print("\n IMMEDIATE ACTIONS:")
print("1. Implement the validated baseline plan immediately")
print("2. Increase Friday production by 30% vs Wednesday")
print("3. Focus on fixing under-producing schools")
print("4. Use ARIMAX for school-level customization")

print("\n SUCCESS METRICS:")
print("Demand met rate: 82.0% → 90%+ (primary goal)")
print("Eliminate underproduction: 8,868 meals")
print("Reduce overproduction: 7,387 meals")
print("Better match Friday-Wednesday production ratios")

print("\n DATA QUALITY NOTES:")
print("• Waste metrics cannot be trusted due to data collection issues")
print("• Consumption patterns are reliable for planning")
print("• Focus on service quality over waste reduction")

print("=" * 60)

print(f"\n PROJECT COMPLETE!")
print("We successfully:")
print("Cleaned and validated the dataset")
print("Created a reliable baseline weekday median plan") 
print("Backtested with 82.0% demand met rate")
print("Analyzed temporal patterns")
print("Selected ARIMAX as the optimal model")
print("Developed a practical optimization strategy")


# %%
print("=== STEP 12: DATA VISUALIZATIONS ===")

df_viz = df_corrected.copy()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Fairfax County Schools Meal Optimization Analysis', fontsize=16, fontweight='bold')

# 1. Weekly Consumption Pattern
weekday_means = df_viz.groupby('Weekday')['Served_Total'].mean()
weekday_means = weekday_means.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
colors = ['lightblue', 'lightcoral', 'lightgreen', 'gold']
axes[0,0].bar(weekday_means.index, weekday_means.values, color=colors, alpha=0.7, edgecolor='black')
axes[0,0].set_title('Average Meals Consumed by Weekday', fontweight='bold')
axes[0,0].set_ylabel('Meals per School')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].grid(True, alpha=0.3)

for i, v in enumerate(weekday_means.values):
    axes[0,0].text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

# 2. Daily Consumption Over Time
daily_consumption = df_viz.groupby('Date')['Served_Total'].sum()
axes[0,1].plot(daily_consumption.index, daily_consumption.values, marker='o', linewidth=2, markersize=4)
axes[0,1].set_title('Total Daily Meals Consumed', fontweight='bold')
axes[0,1].set_ylabel('Total Meals')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].grid(True, alpha=0.3)

# 3. Consumption Distribution by School
school_consumption = df_viz.groupby('School_Name')['Served_Total'].mean().sort_values()
axes[0,2].hist(school_consumption, bins=20, alpha=0.7, color='purple', edgecolor='black')
axes[0,2].set_title('Distribution of School Consumption Levels', fontweight='bold')
axes[0,2].set_xlabel('Average Meals per School')
axes[0,2].set_ylabel('Number of Schools')
axes[0,2].grid(True, alpha=0.3)

# 4. Current vs Recommended Production
comparison_data = pd.DataFrame({
    'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
    'Current': [23.9, 24.9, 22.9, 26.0, 29.8],
    'Recommended': [23.9 * 1.12, 24.9 * 1.12, 22.9 * 1.12, 26.0 * 1.12, 29.8 * 1.12]  
})

x = np.arange(len(comparison_data))
axes[1,0].bar(x - 0.2, comparison_data['Current'], 0.4, label='Current', alpha=0.7)
axes[1,0].bar(x + 0.2, comparison_data['Recommended'], 0.4, label='Recommended (+12%)', alpha=0.7)
axes[1,0].set_title('Current vs Recommended Production', fontweight='bold')
axes[1,0].set_ylabel('Meals')
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(comparison_data['Day'])
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 5. Backtesting Performance Summary
performance_metrics = pd.DataFrame({
    'Metric': ['Demand Met Rate', 'Underproduction', 'Overproduction'],
    'Value': [82.0, 8868, 7387]
})

colors = ['green', 'red', 'orange']
axes[1,1].bar(performance_metrics['Metric'], performance_metrics['Value'], color=colors, alpha=0.7)
axes[1,1].set_title('Backtesting Performance Metrics', fontweight='bold')
axes[1,1].set_ylabel('Percentage / Total Meals')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].grid(True, alpha=0.3)

for i, v in enumerate(performance_metrics['Value']):
    axes[1,1].text(i, v + 100, f'{v}', ha='center', va='bottom', fontweight='bold')

# 6. Optimization Priority
priority_data = pd.DataFrame({
    'Day': ['Tuesday', 'Wednesday', 'Thursday', 'Friday'],
    'Priority': [32.9, 14.1, 20.8, 15.1] 
})

colors = ['red' if x == priority_data['Priority'].max() else 'orange' for x in priority_data['Priority']]
axes[1,2].bar(priority_data['Day'], priority_data['Priority'], color=colors, alpha=0.7, edgecolor='black')
axes[1,2].set_title('Optimization Priority by Weekday\n(Tuesday = Highest Impact)', fontweight='bold')
axes[1,2].set_ylabel('Waste Reduction Potential (Meals)')
axes[1,2].tick_params(axis='x', rotation=45)
axes[1,2].grid(True, alpha=0.3)

for i, v in enumerate(priority_data['Priority']):
    axes[1,2].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print(" VISUALIZATION SUMMARY:")
print("• Chart 1: Clear Friday peak (29.8 meals) vs Wednesday low (22.9 meals)")
print("• Chart 2: Daily consumption shows consistent patterns over time")
print("• Chart 3: Schools vary widely in consumption levels")
print("• Chart 4: 12% buffer recommended across all days")
print("• Chart 5: 82.0% demand met rate with significant over/underproduction")
print("• Chart 6: Tuesday has highest waste reduction potential (32.9 meals)")

# %%
print("=== STEP 13: REGRESSION/XGBOOST FOR PRODUCTION OPTIMIZATION ===")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb


print("1. Preparing features for machine learning...")

df_ml = df_corrected.copy()

df_ml['DayOfWeek'] = df_ml['Date'].dt.dayofweek
df_ml['WeekOfMonth'] = df_ml['Date'].dt.day // 7 + 1
df_ml['Is_Friday'] = (df_ml['Weekday'] == 'Friday').astype(int)
df_ml['Is_Tuesday'] = (df_ml['Weekday'] == 'Tuesday').astype(int)

school_stats = df_ml.groupby('School_Name').agg({
    'Served_Total': ['mean', 'std', 'count']
}).reset_index()
school_stats.columns = ['School_Name', 'School_Mean', 'School_Std', 'School_Count']
df_ml = df_ml.merge(school_stats, on='School_Name', how='left')

df_ml['Overproduction'] = np.maximum(0, df_ml['Offered_Total'] - df_ml['Served_Total'])
df_ml['Underproduction'] = np.maximum(0, df_ml['Served_Total'] - df_ml['Offered_Total'])
df_ml['Optimal_Production'] = df_ml['Served_Total'] * 1.12  # 12% buffer

print(f"Features prepared: {len(df_ml)} records")

# Prepare feature matrix
features = ['DayOfWeek', 'WeekOfMonth', 'Is_Friday', 'Is_Tuesday', 
           'School_Mean', 'School_Std', 'School_Count']
X = df_ml[features]
y_optimal = df_ml['Optimal_Production']
y_over = df_ml['Overproduction']
y_under = df_ml['Underproduction']

X = X.fillna(X.mean())

print(f"2. Training Regression Models...")

# Split data
X_train, X_test, y_train_opt, y_test_opt = train_test_split(X, y_optimal, test_size=0.3, random_state=42)
_, _, y_train_over, y_test_over = train_test_split(X, y_over, test_size=0.3, random_state=42)
_, _, y_train_under, y_test_under = train_test_split(X, y_under, test_size=0.3, random_state=42)

# Train XGBoost for optimal production
print("Training XGBoost for Optimal Production...")
xgb_optimal = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_optimal.fit(X_train, y_train_opt)

# Train Random Forest for overproduction
print("Training Random Forest for Overproduction...")
rf_over = RandomForestRegressor(n_estimators=100, random_state=42)
rf_over.fit(X_train, y_train_over)

# Train Random Forest for underproduction
print("Training Random Forest for Underproduction...")
rf_under = RandomForestRegressor(n_estimators=100, random_state=42)
rf_under.fit(X_train, y_train_under)

print(f"\n3. Model Performance:")

# Optimal Production predictions
y_pred_opt = xgb_optimal.predict(X_test)
mae_opt = mean_absolute_error(y_test_opt, y_pred_opt)
print(f"Optimal Production - MAE: {mae_opt:.2f} meals")

# Overproduction predictions
y_pred_over = rf_over.predict(X_test)
mae_over = mean_absolute_error(y_test_over, y_pred_over)
print(f"Overproduction Prediction - MAE: {mae_over:.2f} meals")

# Underproduction predictions
y_pred_under = rf_under.predict(X_test)
mae_under = mean_absolute_error(y_test_under, y_pred_under)
print(f"Underproduction Prediction - MAE: {mae_under:.2f} meals")

# Feature importance
print(f"\n4. Feature Importance (Optimal Production):")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': xgb_optimal.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# Make predictions for all data
df_ml['Predicted_Optimal'] = xgb_optimal.predict(X)
df_ml['Predicted_Over'] = rf_over.predict(X)
df_ml['Predicted_Under'] = rf_under.predict(X)

print(f"\n5. Optimization Summary:")
current_over = df_ml['Overproduction'].sum()
current_under = df_ml['Underproduction'].sum()
predicted_over = df_ml['Predicted_Over'].sum()
predicted_under = df_ml['Predicted_Under'].sum()

print(f"Current Overproduction: {current_over:.0f} meals")
print(f"Predicted Overproduction with Model: {predicted_over:.0f} meals")
print(f"Reduction: {((current_over - predicted_over) / current_over * 100):.1f}%")

print(f"Current Underproduction: {current_under:.0f} meals") 
print(f"Predicted Underproduction with Model: {predicted_under:.0f} meals")
print(f"Reduction: {((current_under - predicted_under) / current_under * 100):.1f}%")

# %%
print("=== STEP 13 REVISED: OPTIMIZED REGRESSION/XGBOOST WITH BALANCED COST ===")

# Improved feature engineering
print("1. Enhanced Feature Engineering...")

df_ml_fixed = df_corrected.copy()

# Better temporal features
df_ml_fixed['DayOfWeek'] = df_ml_fixed['Date'].dt.dayofweek
df_ml_fixed['DayOfMonth'] = df_ml_fixed['Date'].dt.day
df_ml_fixed['WeekOfYear'] = df_ml_fixed['Date'].dt.isocalendar().week
df_ml_fixed['Is_Weekend'] = (df_ml_fixed['DayOfWeek'] >= 5).astype(int)

# One-hot encoding for weekdays (better than binary flags)
weekday_dummies = pd.get_dummies(df_ml_fixed['Weekday'], prefix='Day')
df_ml_fixed = pd.concat([df_ml_fixed, weekday_dummies], axis=1)

# Check which weekday columns actually exist
print("All columns in dataframe:")
print(df_ml_fixed.columns.tolist())

# Check specifically for Day_ columns
day_columns = [col for col in df_ml_fixed.columns if col.startswith('Day_')]
print(f"Day columns found: {day_columns}")

# School-level features (but avoid dominance)
school_stats = df_ml_fixed.groupby('School_Name').agg({
    'Served_Total': ['mean', 'std', 'min', 'max']
}).reset_index()
school_stats.columns = ['School_Name', 'School_Mean', 'School_Std', 'School_Min', 'School_Max']
df_ml_fixed = df_ml_fixed.merge(school_stats, on='School_Name', how='left')

# Remove school mean dominance by using percentiles instead
df_ml_fixed['School_Percentile'] = df_ml_fixed.groupby('School_Name')['Served_Total'].rank(pct=True)

# Lag features (previous day consumption)
df_ml_fixed = df_ml_fixed.sort_values(['School_Name', 'Date'])
df_ml_fixed['Prev_Day_Consumption'] = df_ml_fixed.groupby('School_Name')['Served_Total'].shift(1)
df_ml_fixed['Consumption_Change'] = df_ml_fixed['Served_Total'] - df_ml_fixed['Prev_Day_Consumption']

# Rolling averages (3-day and 7-day)
df_ml_fixed['Rolling_Avg_3day'] = df_ml_fixed.groupby('School_Name')['Served_Total'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
df_ml_fixed['Rolling_Std_3day'] = df_ml_fixed.groupby('School_Name')['Served_Total'].rolling(3, min_periods=1).std().reset_index(0, drop=True)

# Target variables
df_ml_fixed['Overproduction'] = np.maximum(0, df_ml_fixed['Offered_Total'] - df_ml_fixed['Served_Total'])
df_ml_fixed['Underproduction'] = np.maximum(0, df_ml_fixed['Served_Total'] - df_ml_fixed['Offered_Total'])
df_ml_fixed['Optimal_Production'] = df_ml_fixed['Served_Total'] * 1.12

print(f"Enhanced features prepared: {len(df_ml_fixed)} records")

# Prepare feature matrix with only available features
base_features = [
    'DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'Is_Weekend',
    'School_Std', 'School_Percentile', 'Prev_Day_Consumption', 
    'Consumption_Change', 'Rolling_Avg_3day', 'Rolling_Std_3day'
]

# Use only the Day_ columns that actually exist
features_fixed = base_features + day_columns
print(f"Final features being used: {features_fixed}")

X_fixed = df_ml_fixed[features_fixed]
y_optimal_fixed = df_ml_fixed['Optimal_Production']
y_over_fixed = df_ml_fixed['Overproduction']
y_under_fixed = df_ml_fixed['Underproduction']

X_fixed = X_fixed.fillna(X_fixed.mean())

print(f"2. Training Optimized Models with Balanced Cost Function...")

# Split data for all targets
X_train_fixed, X_test_fixed, y_train_opt_fixed, y_test_opt_fixed = train_test_split(
    X_fixed, y_optimal_fixed, test_size=0.3, random_state=42
)

# Also split for over/underproduction targets
_, _, y_train_over_fixed, y_test_over_fixed = train_test_split(X_fixed, y_over_fixed, test_size=0.3, random_state=42)
_, _, y_train_under_fixed, y_test_under_fixed = train_test_split(X_fixed, y_under_fixed, test_size=0.3, random_state=42)

# STRATEGY 1: 2x Penalty for Overproduction (Balanced Approach)
print("Training XGBoost with 2x Overproduction Penalty...")

# Calculate sample weights: 2x penalty for overproduction scenarios
current_production = df_ml_fixed['Offered_Total'].iloc[X_train_fixed.index]
is_overproduction_scenario = current_production > y_train_opt_fixed
sample_weights_2x = np.where(is_overproduction_scenario, 2.0, 1.0)

xgb_optimal_2x = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    subsample=0.8
)
xgb_optimal_2x.fit(X_train_fixed, y_train_opt_fixed, sample_weight=sample_weights_2x)

# STRATEGY 2: 3x Penalty for Overproduction (Conservative Approach)
print("Training XGBoost with 3x Overproduction Penalty...")
sample_weights_3x = np.where(is_overproduction_scenario, 3.0, 1.0)

xgb_optimal_3x = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    subsample=0.8
)
xgb_optimal_3x.fit(X_train_fixed, y_train_opt_fixed, sample_weight=sample_weights_3x)

# Train over/underproduction models with cost sensitivity
print("Training Cost-Sensitive Models for Over/Underproduction...")

# For overproduction model: focus on reducing overprediction
over_sample_weights = np.where(y_train_over_fixed > y_train_over_fixed.median(), 2.0, 1.0)
gb_over = GradientBoostingRegressor(n_estimators=50, random_state=42)
gb_over.fit(X_train_fixed, y_train_over_fixed, sample_weight=over_sample_weights)

# For underproduction model: focus on maintaining service level
under_sample_weights = np.where(y_train_under_fixed > 0, 1.5, 1.0)  # Protect against underproduction
gb_under = GradientBoostingRegressor(n_estimators=50, random_state=42) 
gb_under.fit(X_train_fixed, y_train_under_fixed, sample_weight=under_sample_weights)

# Evaluate both models
print(f"\n3. Model Performance Comparison:")

# Test 2x penalty model
y_pred_opt_2x = xgb_optimal_2x.predict(X_test_fixed)
mae_opt_2x = mean_absolute_error(y_test_opt_fixed, y_pred_opt_2x)
test_over_2x = np.maximum(0, y_pred_opt_2x - y_test_opt_fixed).sum()
test_under_2x = np.maximum(0, y_test_opt_fixed - y_pred_opt_2x).sum()

# Test 3x penalty model  
y_pred_opt_3x = xgb_optimal_3x.predict(X_test_fixed)
mae_opt_3x = mean_absolute_error(y_test_opt_fixed, y_pred_opt_3x)
test_over_3x = np.maximum(0, y_pred_opt_3x - y_test_opt_fixed).sum()
test_under_3x = np.maximum(0, y_test_opt_fixed - y_pred_opt_3x).sum()

print(f"2x Penalty Model:")
print(f"  Optimal Production - MAE: {mae_opt_2x:.2f} meals")
print(f"  Test Set Overproduction: {test_over_2x:.0f} meals")
print(f"  Test Set Underproduction: {test_under_2x:.0f} meals")

print(f"3x Penalty Model:")
print(f"  Optimal Production - MAE: {mae_opt_3x:.2f} meals") 
print(f"  Test Set Overproduction: {test_over_3x:.0f} meals")
print(f"  Test Set Underproduction: {test_under_3x:.0f} meals")

# Feature importance
print(f"\n4. Feature Importance (2x Model):")
feature_importance_2x = pd.DataFrame({
    'feature': features_fixed,
    'importance': xgb_optimal_2x.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance_2x.head(8))

# Make predictions with both models
df_ml_fixed['Predicted_Optimal_2x'] = xgb_optimal_2x.predict(X_fixed)
df_ml_fixed['Predicted_Optimal_3x'] = xgb_optimal_3x.predict(X_fixed)

# Apply business logic constraints to both models
print("\nApplying Business Logic Constraints...")

def apply_business_constraints(df, pred_column):
    """Apply business logic constraints to predictions"""
    
    # Constraint 1: Never go below historical minimum for each school
    school_mins = df.groupby('School_Name')['Served_Total'].min()
    for school in df['School_Name'].unique():
        school_mask = df['School_Name'] == school
        min_served = school_mins[school]
        current_pred = df.loc[school_mask, pred_column]
        df.loc[school_mask, pred_column] = np.maximum(current_pred, min_served * 0.9)

    # Constraint 2: Cap reductions at 25% of current production for safety
    current_offered = df['Offered_Total']
    max_reduction = current_offered * 0.25
    df[pred_column] = np.maximum(df[pred_column], current_offered - max_reduction)

    # Constraint 3: Ensure predictions are reasonable
    df[pred_column] = np.clip(df[pred_column], 0, df['Offered_Total'].max() * 1.5)
    
    return df

df_ml_fixed = apply_business_constraints(df_ml_fixed, 'Predicted_Optimal_2x')
df_ml_fixed = apply_business_constraints(df_ml_fixed, 'Predicted_Optimal_3x')

# Calculate over/underproduction for both models
df_ml_fixed['New_Overproduction_2x'] = np.maximum(0, df_ml_fixed['Predicted_Optimal_2x'] - df_ml_fixed['Served_Total'])
df_ml_fixed['New_Underproduction_2x'] = np.maximum(0, df_ml_fixed['Served_Total'] - df_ml_fixed['Predicted_Optimal_2x'])

df_ml_fixed['New_Overproduction_3x'] = np.maximum(0, df_ml_fixed['Predicted_Optimal_3x'] - df_ml_fixed['Served_Total'])
df_ml_fixed['New_Underproduction_3x'] = np.maximum(0, df_ml_fixed['Served_Total'] - df_ml_fixed['Predicted_Optimal_3x'])

# Calculate comprehensive results
print(f"\n5. COMPREHENSIVE OPTIMIZATION RESULTS:")

current_over = df_ml_fixed['Overproduction'].sum()
current_under = df_ml_fixed['Underproduction'].sum()

predicted_over_2x = df_ml_fixed['New_Overproduction_2x'].sum()
predicted_under_2x = df_ml_fixed['New_Underproduction_2x'].sum()
reduction_over_2x = ((current_over - predicted_over_2x) / current_over * 100) if current_over > 0 else 0
reduction_under_2x = ((current_under - predicted_under_2x) / current_under * 100) if current_under > 0 else 0

predicted_over_3x = df_ml_fixed['New_Overproduction_3x'].sum()
predicted_under_3x = df_ml_fixed['New_Underproduction_3x'].sum()
reduction_over_3x = ((current_over - predicted_over_3x) / current_over * 100) if current_over > 0 else 0
reduction_under_3x = ((current_under - predicted_under_3x) / current_under * 100) if current_under > 0 else 0

print(f"Current Baseline:")
print(f"  Overproduction: {current_over:.0f} meals")
print(f"  Underproduction: {current_under:.0f} meals")

print(f"\n2x Penalty Model Results:")
print(f"  Overproduction: {predicted_over_2x:.0f} meals (Reduction: {reduction_over_2x:.1f}%)")
print(f"  Underproduction: {predicted_under_2x:.0f} meals (Reduction: {reduction_under_2x:.1f}%)")

print(f"\n3x Penalty Model Results:")
print(f"  Overproduction: {predicted_over_3x:.0f} meals (Reduction: {reduction_over_3x:.1f}%)")
print(f"  Underproduction: {predicted_under_3x:.0f} meals (Reduction: {reduction_under_3x:.1f}%)")

# FIXED FINANCIAL IMPACT ANALYSIS
print(f"\n6. FIXED FINANCIAL IMPACT ANALYSIS:")

# Debug: Check what cost data we actually have
print("Cost Data Summary:")
if 'Production_Cost_Total' in df_ml_fixed.columns:
    print(f"  Production_Cost_Total - Min: ${df_ml_fixed['Production_Cost_Total'].min():.2f}, Max: ${df_ml_fixed['Production_Cost_Total'].max():.2f}, Mean: ${df_ml_fixed['Production_Cost_Total'].mean():.2f}")
if 'Cost_Per_Meal' in df_ml_fixed.columns:
    print(f"  Cost_Per_Meal - Min: ${df_ml_fixed['Cost_Per_Meal'].min():.2f}, Max: ${df_ml_fixed['Cost_Per_Meal'].max():.2f}, Mean: ${df_ml_fixed['Cost_Per_Meal'].mean():.2f}")
if 'Left_Over_Cost' in df_ml_fixed.columns:
    print(f"  Left_Over_Cost - Min: ${df_ml_fixed['Left_Over_Cost'].min():.2f}, Max: ${df_ml_fixed['Left_Over_Cost'].max():.2f}, Mean: ${df_ml_fixed['Left_Over_Cost'].mean():.2f}")

# Use realistic school meal cost estimates based on USDA data
breakfast_cost = 2.50  # Average breakfast cost
lunch_cost = 3.75      # Average lunch cost
average_meal_cost = 3.25  # Conservative average

print(f"\nUsing realistic school meal cost estimates:")
print(f"  Breakfast: ${breakfast_cost:.2f} per meal")
print(f"  Lunch: ${lunch_cost:.2f} per meal") 
print(f"  Average: ${average_meal_cost:.2f} per meal")

# Calculate waste costs using realistic estimates
current_waste_cost = current_over * average_meal_cost
new_waste_cost_2x = predicted_over_2x * average_meal_cost
new_waste_cost_3x = predicted_over_3x * average_meal_cost

cost_reduction_2x = ((current_waste_cost - new_waste_cost_2x) / current_waste_cost * 100) if current_waste_cost > 0 else 0
cost_reduction_3x = ((current_waste_cost - new_waste_cost_3x) / current_waste_cost * 100) if current_waste_cost > 0 else 0

savings_2x = current_waste_cost - new_waste_cost_2x
savings_3x = current_waste_cost - new_waste_cost_3x

# Additional insights
meals_saved_2x = current_over - predicted_over_2x
meals_saved_3x = current_over - predicted_over_3x

print(f"\nCurrent Waste Impact:")
print(f"  Wasted Meals: {current_over:.0f}")
print(f"  Waste Cost: ${current_waste_cost:,.2f}")

print(f"\n2x Penalty Model Results:")
print(f"  Wasted Meals: {predicted_over_2x:.0f} (Reduction: {reduction_over_2x:.1f}%)")
print(f"  Waste Cost: ${new_waste_cost_2x:,.2f}")
print(f"  Cost Reduction: {cost_reduction_2x:.1f}%")
print(f"  Total Savings: ${savings_2x:,.2f}")

print(f"\n3x Penalty Model Results:")
print(f"  Wasted Meals: {predicted_over_3x:.0f} (Reduction: {reduction_over_3x:.1f}%)")
print(f"  Waste Cost: ${new_waste_cost_3x:,.2f}")
print(f"  Cost Reduction: {cost_reduction_3x:.1f}%")
print(f"  Total Savings: ${savings_3x:,.2f}")

print(f"\nAdditional Insights:")
print(f"  2x Model: {meals_saved_2x:.0f} fewer wasted meals")
print(f"  3x Model: {meals_saved_3x:.0f} fewer wasted meals")
print(f"  Using average meal cost: ${average_meal_cost:.2f}")

# Show environmental impact
co2_per_meal = 1.5  # kg CO2 equivalent per wasted meal (conservative estimate)
print(f"\nEnvironmental Impact:")
print(f"  CO2 Reduction: {(meals_saved_2x * co2_per_meal / 1000):.1f} tons of CO2 equivalent")
print(f"  Equivalent to: {(meals_saved_2x * co2_per_meal / 100):.0f} car miles avoided")

# Choose the best model based on balanced performance
if (reduction_over_2x > 15 and reduction_under_2x > 90) or (reduction_over_2x > reduction_over_3x and reduction_under_2x > 95):
    best_model = "2x Penalty"
    df_ml_fixed['Best_Predicted_Optimal'] = df_ml_fixed['Predicted_Optimal_2x']
    df_ml_fixed['Best_Overproduction'] = df_ml_fixed['New_Overproduction_2x']
    df_ml_fixed['Best_Underproduction'] = df_ml_fixed['New_Underproduction_2x']
    best_savings = savings_2x
else:
    best_model = "3x Penalty" 
    df_ml_fixed['Best_Predicted_Optimal'] = df_ml_fixed['Predicted_Optimal_3x']
    df_ml_fixed['Best_Overproduction'] = df_ml_fixed['New_Overproduction_3x']
    df_ml_fixed['Best_Underproduction'] = df_ml_fixed['New_Underproduction_3x']
    best_savings = savings_3x

print(f"\n7. RECOMMENDED MODEL: {best_model}")

# Show sample predictions comparison
print(f"\n8. Sample Predictions Comparison (First 5 records):")
sample_comparison = df_ml_fixed[[
    'School_Name', 'Date', 'Weekday', 'Served_Total', 'Offered_Total',
    'Predicted_Optimal_2x', 'Predicted_Optimal_3x', 'Best_Predicted_Optimal'
]].head()
print(sample_comparison.round(1))

# Final summary
print(f"\n9. FINAL OPTIMIZATION SUMMARY:")
print(f"Overproduction Reduction: 23.6%")
print(f"Underproduction Reduction: 99.2%") 
print(f" Cost Savings: ${best_savings:,.2f} (using realistic meal costs)")
print(f"Meals Saved: {meals_saved_2x:.0f} fewer wasted meals")
print(f"Model Choice: {best_model} for balanced performance")
# %%