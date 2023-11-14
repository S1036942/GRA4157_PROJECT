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
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import OneHotEncoder
    import statsmodels.api as sm


    # Drop rows with missing 'salary' values for this regression analysis
    df_cleaned = df.dropna(subset=['salary'])

    # Remove outliers:
    Q1 = df_cleaned['salary'].quantile(0.25)
    Q3 = df_cleaned['salary'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = df_cleaned[(df_cleaned['salary'] >= lower_bound) & (df_cleaned['salary'] <= upper_bound)]




    # Selecting features and target variable
    features = df_cleaned[['gender', 'ssc_p', 'hsc_p', 'degree_p', 'workex', 'etest_p', 'specialisation', 'mba_p']]
    #features = df_cleaned[['gender', 'workex', 'specialisation', 'mba_p']]
    target = df_cleaned['salary']

    # Convert categorical variables into dummy variables
    features_encoded = pd.get_dummies(features, drop_first=True)

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.35, random_state=42)



    def shape():
        print(X_train.shape)    #Output here is: (118,8)
        print(X_test.shape)     #Output here is: (30,8)
        print(y_train.shape)    #Output here is: (118,)
        print(y_test.shape)     #Output here is: (30,)
    shape()

    def with_statsmodels():
        X_train_sm = sm.add_constant(X_train)
        X_test_sm = sm.add_constant(X_test)

        
        #Make sure dummy variables are int:
        X_train_sm['gender_M'] = X_train_sm['gender_M'].astype(int)
        X_train_sm['workex_Yes'] = X_train_sm['workex_Yes'].astype(int)
        X_train_sm['specialisation_Mkt&HR'] = X_train_sm['specialisation_Mkt&HR'].astype(int)

        X_test_sm['gender_M'] = X_test_sm['gender_M'].astype(int)
        X_test_sm['workex_Yes'] = X_test_sm['workex_Yes'].astype(int)
        X_test_sm['specialisation_Mkt&HR'] = X_test_sm['specialisation_Mkt&HR'].astype(int)
        
        # Fit the model
        model = sm.OLS(y_train, X_train_sm).fit()

        # Display the summary
        print("Statsmodels)")
        print(model.summary())
        
        #Prediciton:
        y_pred = model.predict(X_test_sm)

        #Evaluating model:
        #MSE:
        mse = np.mean((y_test - y_pred) ** 2)
        print(f"MSE: {mse:.3f}")
        r2 = model.rsquared
        print(f"R-squared: {r2:.3f}")
        #print(y_pred)

        # Predict one data point:
        #predict for one data point:
        def predict_one_data_sm():
            new_data_features = {'gender':'M', 'ssc_p':'70.00', 'hsc_p':'60.00', 'degree_p':'70.00', 'workex':'No', 'etest_p':'66.00', 'specialisation':'Mkt&HR', 'mba_p':'60.00'}
            #new_data_features = {'gender':'M', 'workex':'Yes', 'specialisation':'Mkt&HR', 'mba_p':'88.0'}
            new_data_df = pd.DataFrame([new_data_features])
            # Convert string representations of numbers to float
            for column in ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']:
            #for column in ['mba_p']:
                new_data_df[column] = pd.to_numeric(new_data_df[column])

            # One-hot encode the data
            new_data_encoded = pd.get_dummies(new_data_df, drop_first=True)

            # Ensure the new data has the same columns as the training data
            # If a column is missing in new_data_encoded, add it with a value of 0
            for col in features_encoded.columns:  # Assuming features_encoded is from your training data
                if col not in new_data_encoded.columns:
                    new_data_encoded[col] = 0

            # Ensure the columns are in the same order as the training data
            new_data_encoded = new_data_encoded[features_encoded.columns]

            # Add a constant for the intercept
            new_data_encoded['const'] = 1
            new_data_with_const = new_data_encoded

            # Predict using the model
            prediction = model.predict(new_data_with_const)
            #print(prediction)
            print("Predicted value:", prediction[0], "dollars")
        #predict_one_data_sm()

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
    #residual_analysis()

regression()


"""
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


        #predict for one data point:
        def predict_one_data():
            new_data_features = {'gender':'M', 'ssc_p':'91.00', 'hsc_p':'60.00', 'degree_p':'70.00', 'workex':'Yes', 'etest_p':'66.00', 'specialisation':'Mkt&HR', 'mba_p':'88.00'}
            #new_data_features_list = [value for value in new_data_features_dict.values()]
            new_data_df = pd.DataFrame(columns=df_cleaned.columns)
            new_data_df = pd.concat([new_data_df, pd.DataFrame([new_data_features])], ignore_index=True)

            # Convert new values to numeric:
            for column in ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']:
                new_data_df[column] = pd.to_numeric(new_data_df[column])
            # Fix for the dummy variables
            new_data_encoded = pd.get_dummies(new_data_df, drop_first=True)
            
            for col in features_encoded.columns:
                if col not in new_data_encoded.columns:
                    new_data_encoded[col] = 0

            
            prediction = model.predict(new_data_encoded)
            print(f"Prediction: {prediction[0]:.2f} dollars")
        predict_one_data()

        return y_pred
    #y_pred = with_sklearn()
    """
#regression()
