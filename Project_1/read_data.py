import pandas as pd
import numpy as np

def read_file(file):
    f = open(file, "r", encoding = "ISO-8859-1")
    line1 = f.readline()
    header_list = line1.strip().split("\t")
    for i in range(len(header_list)):
        header_list[i] = header_list[i].strip('"')


    data_dict = {}
    for h in header_list:
        data_dict[h] = []


    for line in f:
        list = line.strip().split()

        postal_code = list[0]
        place = list[1]
        merged_place = postal_code + " " + place
        
        new_list = []
        if len(list) == 9:
            house_type = list[5]
            quarter = list[6]
            avergage_price = list[7]
            number_sold = list[8]
        elif len(list) == 8:
            house_type = list[4]
            quarter = list[5]
            avergage_price = list[6]
            number_sold = list[7]
    
        elif len(list) == 7:
            house_type = list[3]
            quarter = list[4]
            avergage_price = list[5]
            number_sold = list[6]
        else:
            house_type = list[2]
            quarter = list[3]
            avergage_price = list[4]
            number_sold = list[5]

        new_list.append(merged_place)
        new_list.append(house_type)
        new_list.append(quarter)
        new_list.append(avergage_price)
        new_list.append(number_sold)


        for i in range(len(new_list)):
            new_list[i] = new_list[i].strip('"')

        def check_values():    
            if new_list[3] == "." or new_list[3] == "..":
                new_list[3] = -1
            else: 
                new_list[3] = int(new_list[3])

            if new_list[4] == "." or new_list[4] == "..":
                new_list[4] = -1
            else: 
                new_list[4] = int(new_list[4])

        for i, header in enumerate(header_list):
            data_dict[header].append(new_list[i])
    f.close()
    return data_dict

data = read_file("data.csv")
#data = read_file("data_semi.csv")

# Now over to the dataframe:
import matplotlib.pyplot as plt

# Make Nan valuea:
df = pd.DataFrame(data)
df.replace(['.', '..'], np.nan, inplace = True)


# Convert all values to numeric
df['Gjennomsnittlig kvadratmeterpris (kr)'] = pd.to_numeric(df['Gjennomsnittlig kvadratmeterpris (kr)'], errors = 'coerce')
df['Antall boligomsetninger'] = pd.to_numeric(df['Gjennomsnittlig kvadratmeterpris (kr)'], errors = 'coerce')

# remove all Nan values:
df_cleaned = df.dropna(subset=['Gjennomsnittlig kvadratmeterpris (kr)', 'Antall boligomsetninger'], how='all')

#Info about data:
#print(df_cleaned.describe())
df = df_cleaned

# find average price for each region:
average_price_by_region = df.groupby('region')['Gjennomsnittlig kvadratmeterpris (kr)'].mean().reset_index()
average_price_by_region = average_price_by_region.sort_values(by = 'Gjennomsnittlig kvadratmeterpris (kr)', ascending=False)
#print(average_price_by_region)

#Lowest and highest:
def highest_lowest():
    highest = average_price_by_region.head(1)
    lowest = average_price_by_region.tail(1)
    plt.barh(highest['region'], highest['Gjennomsnittlig kvadratmeterpris (kr)'], color='skyblue')
    plt.barh(lowest['region'], lowest['Gjennomsnittlig kvadratmeterpris (kr)'], color='red')
    plt.title('Highest and lowest sqm price')
    plt.xlabel('Average Price per Square Meter (kr)')
    plt.ylabel('Region')
    plt.tight_layout()
    plt.show()
highest_lowest()

# info about just Oslo:
def oslo_info():
    oslo_df = df[df['region'] == "0301 Oslo"]
    print(oslo_df)

    # Average per quarter:
    avg_q_oslo = oslo_df.groupby('kvartal')['Gjennomsnittlig kvadratmeterpris (kr)'].mean()

    # Change the time:

    plt.figure(figsize=(12, 6))
    avg_q_oslo.plot()
    plt.title(f'Price Change Over Time in Oslo')
    plt.xlabel('Quarter')
    plt.ylabel('Average Square Meter Price')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.show()
#oslo_info()

def country_over_time():
    avg_quarter_country = df.groupby('kvartal')['Gjennomsnittlig kvadratmeterpris (kr)'].mean()
    plt.figure(figsize=(12, 6))
    avg_quarter_country.plot()
    plt.title('Price change over time in Norway')
    plt.xlabel('Quarter')
    plt.ylabel('Average price')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
#country_over_time()

def oslo_vs_norway():
    norway = df.groupby('kvartal')['Gjennomsnittlig kvadratmeterpris (kr)'].mean()

    oslo_df = df[df['region'] == "0301 Oslo"]
    oslo = oslo_df.groupby('kvartal')['Gjennomsnittlig kvadratmeterpris (kr)'].mean()

    plt.figure(figsize=(12, 6))
    norway.plot(label = 'Norway', color = 'red')
    oslo.plot(label = "Oslo", color = "blue")
    plt.title('Price change over time in Norway vs. Oslo')
    plt.xlabel('Quarter')
    plt.ylabel('Average price per sqm')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend() # This is to show oslo and norway
    plt.show()
#oslo_vs_norway()

# til oscar:
"""
def oslo_vs_norway():

    norway = data_cleaned.groupby('kvartal')['Gjennomsnittlig kvadratmeterpris (kr)'].mean()

    oslo_df = data_cleaned[data_cleaned['region'] == "0301 Oslo"]
    oslo = oslo_df.groupby('kvartal')['Gjennomsnittlig kvadratmeterpris (kr)'].mean()

    plt.figure(figsize=(12, 6))
    norway.plot(label = 'Norway', color = 'red')
    oslo.plot(label = "Oslo", color = "blue")
    plt.title('Price change over time in Norway vs. Oslo')
    plt.xlabel('Quarter')
    plt.ylabel('Average price per sqm')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend() # This is to show oslo and norway
    plt.show()
"""