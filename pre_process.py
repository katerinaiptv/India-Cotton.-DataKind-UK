import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def pre_process(data, id_column, columns_ordinal, columns_categorical, drop_variables):

    data = data.set_index(id_column)

    # drop columns
    data = data.drop(columns=drop_variables)
    
    
    ### process ordinal data
    columns_ordinal = set(data.columns).intersection(columns_ordinal)
    print(f'Ordinal columns: {columns_ordinal}')
    data_ordinal = data[columns_ordinal]
    
    # impute null values
    data_ordinal = data_ordinal.fillna(0)
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp.fit(data_ordinal)  
    # data_ordinal = pd.DataFrame(imp.transform(data_ordinal),
    #                             columns=columns_ordinal,
    #                             index=data_ordinal.index)

        
    scaler = StandardScaler()
    print(scaler.fit(data_ordinal))
    data_ordinal = pd.DataFrame(scaler.transform(data_ordinal),
                                columns=data_ordinal.columns,
                                index=data_ordinal.index)
    
    
    ### process categorical data
    columns_categorical = set(data.columns).intersection(columns_categorical)
    print(f'Categorical columns: {columns_categorical}')
    data_categorical = data[columns_categorical]

    # impute null values
    columns_categorical_str = set(data_categorical.select_dtypes(include=['object']).columns)
    columns_categorical_num = columns_categorical.difference(columns_categorical_str)

    data_categorical_str = data_categorical.loc[:, list(columns_categorical_str)].fillna('NA', inplace=False)
    data_categorical_num = data_categorical.loc[:, list(columns_categorical_num)].fillna(-99, inplace=False)
    data_categorical = pd.concat([data_categorical_str, data_categorical_num], axis=1)
   
    # encode values in columns with more than two unique values
    columns_categorical_mulitval = list(data_categorical.nunique().keys()[data_categorical.nunique()>2])
    data_categorical_multival = data_categorical[columns_categorical_mulitval]
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    data_categorical_multival = pd.DataFrame(enc.fit_transform(data_categorical_multival),
                                             columns=enc.get_feature_names(data_categorical_multival.columns),
                                             index=data_categorical_multival.index)
    # for columns with only two unique values, ensure that they are 0 and 1
    data_categorical_binary = data_categorical.drop(columns=columns_categorical_mulitval)
    scaler = MinMaxScaler()
    data_categorical_binary = pd.DataFrame(scaler.fit_transform(data_categorical_binary),
                                           columns=data_categorical_binary.columns,
                                           index=data_categorical_binary.index)

    data_categorical = pd.concat([data_categorical_multival, data_categorical_binary], axis=1)
    
    ### recombine ordinal and categorical data
    combined_data = pd.concat([data_ordinal, data_categorical], axis=1)
    
    return combined_data
