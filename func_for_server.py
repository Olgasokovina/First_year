import pickle

import numpy as np
import pandas as pd
from geopy.distance import geodesic


def round_baths(baths):

    if baths is None or baths is np.NaN:
        return None

    baths = float(baths)

    full = baths // 1
    lost = baths % 1

    if lost >= 0 and lost < 0.25:
        return full

    if lost >= 0.25 and lost < 0.75:
        lost = 0.5
        return full + lost

    if lost >= 0.75:
        return full + 1


def compare_city_major(x, size):
    if x is None:
        return False

    x = str(x)
    if str(x).strip() in size:
        return True

    return False


def length_city_capital(row):

    if (pd.isna(row['lat'])) or (pd.isna(row['long'])) or (
            pd.isna(row['lat_capital'])) or (pd.isna(row['long_capital'])):
        return None

    length = geodesic((row['lat'], row['long']),
                      (row['lat_capital'], row['long_capital'])).km

    return length


def length_city_million(row):

    if (pd.isna(row['lat'])) or (pd.isna(row['long'])):
        return None

    with open(r'mln_city.pkl', 'rb') as pkl_file:
        mln_city = pickle.load(pkl_file)

    mln_city['lat_million_city'] = mln_city['lat_million_city'].astype('float')
    mln_city['long_million_city'] = mln_city['long_million_city'].astype(
        'float')

    dist = []

    for i in mln_city.index:
        length = geodesic((row['lat'], row['long']),
                          (mln_city.loc[i, 'lat_million_city'],
                           mln_city.loc[i, 'long_million_city'])).km
        dist.append(length)

    return min(dist)


def data_preprocessing(df):
    features = pd.DataFrame(
        columns=[
            '1_level', '2_level', '3_level', '4_level', 'air_conditioner', 'auction',
            'baths', 'beds', 'Carport', 'ceiling_fan', 'central_heating', 'city_is_big',
            'city_is_capital', 'city_is_large', 'city_is_million', 'claster', 'condo',
            'cool_bool', 'Crime Rate', 'distance_capital', 'distance_million',
            'electric_heating', 'Farms/Ranches', 'firepl', 'forced_air_heating',
            'foreclosure', 'Garage', 'gas_heating', 'HDI (2021)', 'heat_bool', 'lotsize',
            'multi_family', 'multi_level', 'new', 'number_school', 'Parking_lot', 'pool',
            'PublicSchoolOverallRanking', 'pump_heating', 'Remodeled', 'school_distance_mean',
            'school_rating_mean', 'Single Family Home', 'sqft', 'Townhouse', 'Year_built'
        ])

    temp = df[['state', 'city', 'zipcode']]

    with open("crime.pkl", 'rb') as pkl_file:
        crime = pickle.load(pkl_file)
    with open("life_level.pkl", 'rb') as pkl_file:
        life_level = pickle.load(pkl_file)
    with open('public_school_rankings.pkl', 'rb') as pkl_file:
        public_school_rankings = pickle.load(pkl_file)
    with open('capital_state.pkl', 'rb') as pkl_file:
        capital_state = pickle.load(pkl_file)
    with open('lg_city.pkl', 'rb') as pkl_file:
        lg_city = pickle.load(pkl_file)
    with open('bg_city.pkl', 'rb') as pkl_file:
        bg_city = pickle.load(pkl_file)
    with open('mln_city.pkl', 'rb') as pkl_file:
        mln_city = pickle.load(pkl_file)
    with open('model_GaussianMixture.pkl', 'rb') as pkl_file:
        model_GaussianMixture = pickle.load(pkl_file)

    with open('city_center.pkl', 'rb') as pkl_file:
        city_center = pickle.load(pkl_file)

    city_center['zipcode'] = city_center['zipcode'].astype('int')

    temp = temp.merge(crime, how='left', on='state')
    temp = temp.merge(life_level, how='left', on='state')
    temp = temp.merge(public_school_rankings, how='left', on='state')
    temp = temp.merge(capital_state, how='left', on='state')
    # использовать если selenium НЕ РАБОТАЕТ НА СВОЙ СТРАХ И РИСК
    temp = temp.merge(city_center, how='left', on=['zipcode', 'state', 'city'])

    temp.loc[:, 'city_is_capital'] = temp.apply(lambda row: (
        True if row['city'] == row['capital'] else False), axis=1)
    size = lg_city['large_city'].unique()
    temp.loc[:, 'city_is_large'] = temp['city'].apply(
        lambda x: compare_city_major(x, size=size))
    size = bg_city['big_city'].unique()
    temp.loc[:, 'city_is_big'] = temp['city'].apply(
        lambda x: compare_city_major(x, size=size))
    size = mln_city['million_city'].unique()
    temp.loc[:, 'city_is_million'] = temp['city'].apply(
        lambda x: compare_city_major(x, size=size))
    temp['zipcode'] = temp['zipcode'].astype(int)

    # temp = city_state_coord(temp) # использовать только в случае работающего
    # selenium
    temp['distance_capital'] = temp.apply(
        lambda row: length_city_capital(row), axis=1)
    temp['distance_million'] = temp.apply(
        lambda row: length_city_million(row), axis=1)
    temp['claster'] = model_GaussianMixture.predict(temp[['lat', 'long']])

    df = df.merge(temp, how='left', on=['state', 'city', 'zipcode'])

    features['lotsize'] = df['lotsize'].copy()
    features['sqft'] = df['sqft'].copy()
    features['auction'] = df['auction'].copy()
    features['Remodeled'] = df['Remodeled'].copy()
    features['foreclosure'] = df['foreclosure'].copy()
    features['new'] = df['new'].copy()
    features['pool'] = df['pool'].copy()
    features['firepl'] = df['fireplace'].copy()
    features['Garage'] = df['Garage'].copy()
    features['Carport'] = df['Carport'].copy()
    features['Parking_lot'] = df['Parking_lot'].copy()
    features['cool_bool'] = df['cool_bool'].copy()
    features['air_conditioner'] = df['air_conditioner'].copy()
    features['ceiling_fan'] = df['ceiling_fan'].copy()
    features['heat_bool'] = df['heat_bool'].copy()
    features['central_heating'] = df['central_heating'].copy()
    features['pump_heating'] = df['pump_heating'].copy()
    features['gas_heating'] = df['gas_heating'].copy()
    features['forced_air_heating'] = df['forced_air_heating'].copy()
    features['electric_heating'] = df['electric_heating'].copy()
    features['school_distance_mean'] = df['school_distance_mean'].copy()
    features['school_rating_mean'] = df['school_rating_mean'].copy()
    features['number_school'] = df['number_school'].copy()
    features['city_is_big'] = df['city_is_big'].copy()
    features['city_is_capital'] = df['city_is_capital'].copy()
    features['city_is_large'] = df['city_is_large'].copy()
    features['city_is_million'] = df['city_is_million'].copy()
    features['claster'] = df['claster'].copy()
    features['Crime Rate'] = df['Crime Rate'].copy()
    features['distance_capital'] = df['distance_capital'].copy()
    features['distance_million'] = df['distance_million'].copy()
    features['HDI (2021)'] = df['HDI (2021)'].copy()
    features['PublicSchoolOverallRanking'] = df['PublicSchoolOverallRanking'].copy()

    features['Year_built'] = 2019 - df['Year_built']

    features['baths'] = df['baths'].apply(round_baths)
    features['baths'] = features['baths'].apply(
        lambda x: np.round(x) if x > 6 else x)
    features['baths'] = features['baths'].apply(lambda x: 1 if x == 0.5 else x)
    features.loc[features['baths'] > 8, 'baths'] = 8

    features['beds'] = df['beds'].copy()
    features.loc[features['beds'] > 8, 'beds'] = 8

    # может быть только:
    # 'condo',
    #  'Townhouse',
    #  'Single Family Home',
    #  'Farms/Ranches',
    #  'multi_family',
    temp = df['propertyType'].values[0]

    if temp == 'condo':
        features.loc[0, 'condo'] = 1
        features.loc[0, 'Townhouse'] = 0
        features.loc[0, 'Single Family Home'] = 0
        features.loc[0, 'Farms/Ranches'] = 0
        features.loc[0, 'multi_family'] = 0

    elif temp == 'Townhouse':
        features.loc[0, 'condo'] = 0
        features.loc[0, 'Townhouse'] = 1
        features.loc[0, 'Single Family Home'] = 0
        features.loc[0, 'Farms/Ranches'] = 0
        features.loc[0, 'multi_family'] = 0

    elif temp == 'Single Family Home':
        features.loc[0, 'condo'] = 0
        features.loc[0, 'Townhouse'] = 0
        features.loc[0, 'Single Family Home'] = 1
        features.loc[0, 'Farms/Ranches'] = 0
        features.loc[0, 'multi_family'] = 0

    elif temp == 'Farms/Ranches':
        features.loc[0, 'condo'] = 0
        features.loc[0, 'Townhouse'] = 0
        features.loc[0, 'Single Family Home'] = 0
        features.loc[0, 'Farms/Ranches'] = 1
        features.loc[0, 'multi_family'] = 0

    elif df['propertyType'].str == 'multi_family':
        features.loc[0, 'condo'] = 0
        features.loc[0, 'Townhouse'] = 0
        features.loc[0, 'Single Family Home'] = 0
        features.loc[0, 'Farms/Ranches'] = 0
        features.loc[0, 'multi_family'] = 1

    temp = df['stories'].values[0]

    if temp == 1:
        features.loc[0, '1_level'] = 1
        features.loc[0, '2_level'] = 0
        features.loc[0, '3_level'] = 0
        features.loc[0, '4_level'] = 0
        features.loc[0, 'multi_level'] = 0

    elif temp == 2:
        features.loc[0, '1_level'] = 0
        features.loc[0, '2_level'] = 1
        features.loc[0, '3_level'] = 0
        features.loc[0, '4_level'] = 0
        features.loc[0, 'multi_level'] = 0

    elif temp == 3:
        features.loc[0, '1_level'] = 0
        features.loc[0, '2_level'] = 0
        features.loc[0, '3_level'] = 1
        features.loc[0, '4_level'] = 0
        features.loc[0, 'multi_level'] = 0

    elif temp == 4:
        features.loc[0, '1_level'] = 0
        features.loc[0, '2_level'] = 0
        features.loc[0, '3_level'] = 0
        features.loc[0, '4_level'] = 4
        features.loc[0, 'multi_level'] = 0

    elif temp == 1.5:
        features.loc[0, '1_level'] = 0
        features.loc[0, '2_level'] = 1
        features.loc[0, '3_level'] = 0
        features.loc[0, '4_level'] = 0
        features.loc[0, 'multi_level'] = 0

    elif temp > 4:
        features.loc[0, '1_level'] = 0
        features.loc[0, '2_level'] = 0
        features.loc[0, '3_level'] = 0
        features.loc[0, '4_level'] = 0
        features.loc[0, 'multi_level'] = 1

    return features
