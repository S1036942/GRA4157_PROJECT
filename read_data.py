import pandas as pd
import numpy as np

def read2(file):
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
    return data_dict

data = read2("data.csv")  

# Now over to the dataframe:
df = pd.DataFrame(data)
df.replace(['.', '..'], np.nan, inplace = True)
print(df)






