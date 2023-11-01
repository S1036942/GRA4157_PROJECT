import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# Read data:
def read_data(filename):
    dataframe = pd.read_csv(filename, sep = ",")
    return dataframe
df = read_data("Placement_Data_Full_Class.csv")


# Calculate average salary by gender
def gender_differences():
    average_salary_by_gender = df.groupby("gender")["salary"].mean()

    # Create the bar plot using matplotlib
    plt.figure(figsize=(10, 6))
    average_salary_by_gender.plot(kind='bar', color=['blue', 'green'])

    # Set the title and labels
    plt.title("Average Salary by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Average Salary")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
#gender_differences()


# Differences in wages, based on different educational levels:
# Calculate average percentages for each level of education by gender
def educational_differences():
    average_ssc_p = df.groupby("gender")["ssc_p"].mean()
    average_hsc_p = df.groupby("gender")["hsc_p"].mean()
    average_degree_p = df.groupby("gender")["degree_p"].mean()
    average_mba_p = df.groupby("gender")["mba_p"].mean()

    # Prepare data for plotting
    labels = ['Secondary (10th)', 'Higher Secondary (12th)', 'Undergraduate', 'MBA']
    male_averages = [average_ssc_p['M'], average_hsc_p['M'], average_degree_p['M'], average_mba_p['M']]
    female_averages = [average_ssc_p['F'], average_hsc_p['F'], average_degree_p['F'], average_mba_p['F']]

    # Set width of bar
    barWidth = 0.35

    # Set position of bar on X axis
    r1 = range(len(male_averages))
    r2 = [x + barWidth for x in r1]

    # Create the bar plot
    plt.figure(figsize=(12, 7))
    plt.bar(r1, male_averages, color='skyblue', width=barWidth, edgecolor='grey', label='Males')
    plt.bar(r2, female_averages, color='red', width=barWidth, edgecolor='grey', label='Females')

    # Title & subtitle
    plt.title("Comparison of Average Percentages by Gender Across Education Levels", fontweight='bold')

    # Add xticks on the middle of the group bars
    plt.xlabel('Education Level', fontweight='bold')
    plt.ylabel('Average Percentage', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(male_averages))], labels)

    # Create legend & show graphic
    plt.legend()
    plt.show()
#educational_differences()



# Regression:
def regression():
    import sklearn
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import OneHotEncoder
    import statsmodels.api as sm


    # Drop rows with missing 'salary' values for this regression analysis
    df_cleaned = df.dropna(subset=['salary'])

    # Selecting features and target variable
    features = df_cleaned[['gender', 'ssc_p', 'hsc_p', 'degree_p', 'workex', 'etest_p', 'specialisation', 'mba_p']]
    target = df_cleaned['salary']

    # Convert categorical variables into dummy variables
    features_encoded = pd.get_dummies(features, drop_first=True)

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

    print(X_train.shape)    #Output here is: (118,8)
    print(X_test.shape)     #Output here is: (30,8)
    print(y_train.shape)    #Output here is: (118,)
    print(y_test.shape)     #Output here is: (30,)


    def with_sklearn():
        # Create and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predicting on the test set:
        y_pred = model.predict(X_test)

        # Coeficcients:
        intercept = model.intercept_
        coefficients = model.coef_

        print("Coeficcients:")
        print(f"Intercept: {intercept:.3f}")
        for feature_name, coef in zip(features_encoded.columns, coefficients):
            print(f"Coefficient for {feature_name}: {coef:.3f}")

        print()
        # Evaluate the model:
        # Mean squared error:
        mse = mean_squared_error(y_test, y_pred)
        # R-squared:
        r2 = r2_score(y_test, y_pred)
        print(f"MSE: {mse:.3f}"),
        print(f"R-Squared:  {r2:.3f}")
        # Mean absolute error (MAE):
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(y_test, y_pred)
        print(f"MAE: {mae:.3f}")
        # Root mean squared error:
        rmse = np.sqrt(mse)
        print(f"RMSE: {rmse} ")

        return y_pred

    #y_pred = with_sklearn()

    def with_statsmodels():
        X_with_const = sm.add_constant(X_train)
        X_test_with_const = sm.add_constant(X_test)

        # Fit the model
        model = sm.OLS(y_train, X_with_const).fit()

        # Display the summary
        print("Statsmodels)")
        print(model.summary())
        y_pred = model.predict(X_test_with_const)

        #Evaluating model:
        #MSE:
        mse = np.mean((y_test - y_pred) ** 2)
        print(f"MSE: {mse:.3f}")
        r2 = model.rsquared
        print(f"R-squared: {r2:.3f}")

        return y_pred

    y_pred = with_statsmodels()


    # Visualization of actual vs. predicted:
    def pred_vs_act():
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        plt.xlabel('Actual Salary')
        plt.ylabel('Predicted Salary')
        plt.title('Actual vs. Predicted Salary')
        plt.show()
    pred_vs_act()

    # Residuals Analysis
    def residual_analysis():
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, color='red')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Salary')
        plt.ylabel('Residuals')
        plt.title('Residuals Analysis')
        plt.show()
    residual_analysis()

regression()
