import csv
import pickle
import statistics
import time

import cython
import numpy as np
import pandas as pd
import requests
from Cython.Build import cythonize
from func_for_server import *
from geopy.distance import geodesic
from scipy.stats import mode
from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm.notebook import tqdm as tqdmn
from uszipcode import SearchEngine




@cython.cfunc
@cython.exceptval(-2, check=True)
def compare_year(row: cython.double) -> cython.double:

    if (row['Year_built'] is not None) or (row['Remodeled_year'] is not None):
        if (row['Year_built'] >= row['Remodeled_year']) and (
                row['Year_built'] >= 1884):
            return row['Remodeled_year']

    return None




@cython.cfunc
@cython.exceptval(-2, check=True)
def compare_city(row: cython.double) -> cython.double:

    if row['city'] is None or row['city_predict'] is None:
        return None

    if row['city'] in row['city_predict']:
        return True

    return False




@cython.cfunc
@cython.exceptval(-2, check=True)
def change_city_name(row: cython.double) -> cython.double:

    if row['compare']:
        return row['city']

    exceptions = ['02105', '03316', '05642', '05940', '22703', '27000', '33000',
                  '33249', '38818', '44037', '78697', '90109', '98489', '98798', '99338']
    if row['zipcode'] in exceptions:
        return None

    temp = []
    pred = find_city_usps_from_zip(row['zipcode'])
    # time.sleep(0.1)

    temp.append(pred['defaultCity'])
    if len(pred['citiesList']) > 0:
        temp.extend(pred['citiesList'])

    for i in range(1, len(temp)):
        temp[i] = temp[i]['city']

    for i in range(len(temp)):
        temp[i] = temp[i].title()

    if pd.isna(row['city']):
        row['city'] = temp[0]

    if row['city'] == '' or row['city'] == ' ':
        row['city'] = temp[0]

    if row['city'].strip() in temp:
        row['compare'] = 1
        return row['city']

    return temp




@cython.cfunc
@cython.exceptval(-2, check=True)
def acr_in_sqft(x: cython.double) -> cython.double:

    if 'None' in str(x) or x is np.NaN or '--' in str(x):
        return None

    if 'acr' in str(x) or 'Acr' in str(x):
        return float(str(x).replace(',', '').strip().split()[0]) * 43560

    if 'sqft' in str(x).lower():
        return float(str(x).replace(',', '').replace(
            'sqft', '').strip().split()[0])

    if 'sq. ft.' in str(x).lower():
        return float(str(x).replace(',', '').replace(
            'sq. ft.', '').strip().split()[0])

    return x




@cython.cfunc
@cython.exceptval(-2, check=True)
def find_x(x: cython.double, sign: cython.double) -> cython.double:

    x = str(x).lower()
    for i in sign:
        if i in str(x).lower():
            return True

    return False




@cython.cfunc
@cython.exceptval(-2, check=True)
def find_x_exc(x: cython.double, sign: cython.double,
               notsign: cython.double) -> cython.double:

    x = str(x).lower()
    for i in notsign:
        if i in str(x).lower():
            return False

    for i in sign:
        if i in str(x).lower():
            return True

    return False




@cython.cfunc
@cython.exceptval(-2, check=True)
def find_city_usps_from_zip(zipcode: cython.double) -> cython.double:

    url = 'https://tools.usps.com/tools/app/ziplookup/cityByZip'
    data = {'zip': zipcode}
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
    }
    r = requests.post(url, data=data, headers=headers)

    return r.json()




@cython.cfunc
@cython.exceptval(-2, check=True)
def find_zip_usps_from_city(city: cython.double,
                            state: cython.double) -> cython.double:

    url = 'https://tools.usps.com/tools/app/ziplookup/zipByCityState'
    data = {'city': city,
            'state': state,
            }
    headers = {
        # 'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
    }
    r = requests.post(url, data=data, headers=headers)

    return r.json()['zipList']




@cython.cfunc
@cython.exceptval(-2, check=True)
def find_zip_usps_from_adress(
        address1: cython.double, city: cython.double, state: cython.double) -> cython.double:

    url = 'https://tools.usps.com/tools/app/ziplookup/zipByAddress'
    data = {'companyName': '',
            'address1': address1,
            'address2': '',
            'city': city,
            'state': state,
            'urbanCode': '',
            'zip': ''}
    headers = {
        # 'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
    }
    r = requests.post(url, data=data, headers=headers)

    return r.json()['addressList'][0]['zip5']




@cython.cfunc
@cython.exceptval(-2, check=True)
def find_zip(city: cython.double, state: cython.double) -> cython.double:

    engine = SearchEngine()
    plases = engine.by_city_and_state(city=city, state=state)
    zip = []

    for z in plases:
        zip.append(z.zipcode)

    return zip




@cython.cfunc
@cython.exceptval(-2, check=True)
def find_city(zip_code: cython.double) -> cython.double:

    if (zip_code is None) or (zip_code is np.NaN) or (
            zip_code == '') or (zip_code == ' '):
        return None

    exceptions = ['02105', '03316', '05642', '05940', '22703', '27000', '33000',
                  '33249', '38818', '44037', '78697', '90109', '98489', '98798', '99338']

    if zip_code in exceptions:
        return None

    else:
        city_perdict = []
        search = SearchEngine()
        zipcode = search.by_zipcode(zip_code)
        city_perdict.append(zipcode.major_city)
        if zipcode.common_city_list is not None:
            city_perdict.extend(zipcode.common_city_list)

        return city_perdict




@cython.cfunc
@cython.exceptval(-2, check=True)
def find_state(zip_code: cython.double) -> cython.double:

    if (zip_code is None) or (zip_code is np.NaN) or (
            zip_code == '') or (zip_code == ' '):
        return None

    exceptions = ['02105', '03316', '05642', '05940', '22703', '27000', '33000',
                  '33249', '38818', '44037', '78697', '90109', '98489', '98798',
                  '99338', '78697', 78697, '5940', 5940]

    if zip_code in exceptions:
        return None

    search = SearchEngine()
    zipcode = search.by_zipcode(zip_code)
    state = zipcode.state_abbr

    return state




@cython.cfunc
@cython.exceptval(-2, check=True)
def school_distance_mean(x: cython.double) -> cython.double:

    x = str(x).lower().replace('[', '').replace(']', '').replace(
        "'", '').replace(",", '') .replace('mi', '')
    lst = x.split()
    lst = list(filter(None, lst))

    if len(lst) == 0:
        return None

    for i in range(len(lst)):
        lst[i] = float(lst[i])

    else:
        return round(np.mean(lst), 1)




@cython.cfunc
@cython.exceptval(-2, check=True)
def school_rating_mean(x: cython.double) -> cython.double:

    lst = list(x)

    for i in range(len(lst)):
        if '/' in lst[i]:
            lst[i] = str(lst[i]).split(sep='/')[0]
        lst[i] = lst[i].lower().replace('none', '').replace('[', '').replace(
            ']', '').replace(",", '').replace('nr', '').replace('na', '').replace("'", '')

    lst = list(filter(None, lst))

    if len(lst) == 0:
        return None

    else:
        for i in range(len(lst)):
            lst[i] = int(lst[i])

        return round(statistics.mean(lst), 1)




@cython.cfunc
@cython.exceptval(-2, check=True)
def condo_tounhome(row):

    if (row['condo'] == 1) and (row['Townhouse'] == 1):

        if row['beds'] == 1:
            row['Townhouse'] = False
            return row

        if row['sqft'] < 1500:
            row['Townhouse'] = False
            return row

        if (row['sqft'] > 3000 or row['lotsize'] > 3000):
            row['condo'] = False
            return row

        if row['beds'] >= 4:
            row['condo'] = False
            return row

        if row['Garage']:
            row['condo'] = False
            return row

    return row




@cython.cfunc
@cython.exceptval(-2, check=True)
def condo_home(row):

    if (row['condo'] == 1) and (row['Single Family Home'] == 1):

        if row['beds'] == 1:
            row['Single Family Home'] = False
            return row

        if (row['sqft'] < 1800 or row['lotsize'] < 1800):
            row['Single Family Home'] = False
            return row

        if (row['sqft'] > 3000 or row['lotsize'] > 3000):
            row['condo'] = False
            return row

        if row['Garage']:
            row['condo'] = False
            return row
        if row['beds'] >= 4:
            row['condo'] = False
            return row

    return row




@cython.cfunc
@cython.exceptval(-2, check=True)
def change_city_predict_usps(row: cython.double) -> cython.double:

    if row['compare']:
        return row['city']

    if row['compare'] is None or row['city_predict_usps'] is None:
        return row['city']

    if row['compare'] == 0:
        if pd.isna(row['city']) and len(
                str(row['city_predict_usps']).split(sep=',')) == 1:
            return row['city_predict_usps']

        temp = ['Other',
                'Other City - In The State Of Florida',
                'Other City Not In the State Of Florida',
                'Other City Value - Out Of Area',
                'Other City Value Out Of Area',
                'Outside Area (Outside Ca)',
                '--', 'Nan'
                ]

        if pd.isna(row['city']) or row['city'] == ' ' or row['city'] in temp:
            return row['city_predict_usps'][0]

    return row['city']




@cython.cfunc
@cython.exceptval(-2, check=True)
def compare_city_usps(row: cython.double) -> cython.double:

    if row['compare']:
        return row['compare']

    if row['city'] is None or row['city_predict_usps'] is None:
        return None

    if str(row['city']).strip() in row['city_predict_usps']:
        return True

    return False




def city_coord(city, state):
    # time.sleep(0.2)

    url = 'https://www.google.com/maps/search/' + city + ' ' + state + ' ' + 'USA'

    Url_With_Coordinates = []

    option = webdriver.ChromeOptions()
    prefs = {
        'profile.default_content_setting_values': {
            'images': 2,
            'javascript': 2}}
    option.add_experimental_option('prefs', prefs)

    driver = webdriver.Chrome(  # "chromedriver.exe",
        options=option,
    )
    driver.get(url)
    Url_With_Coordinates = driver.find_element(
        By.CSS_SELECTOR, 'meta[itemprop=image]').get_attribute('content')
    driver.close()

    if ('&zoom=' in Url_With_Coordinates) and (
            'center=' in Url_With_Coordinates):

        lat = Url_With_Coordinates.split('?center=')[1].split('&zoom=')[
            0].split('%2C')[0]
        long = Url_With_Coordinates.split('?center=')[1].split('&zoom=')[
            0].split('%2C')[1]
        return (lat, long)

    return (None, None)




@cython.cfunc
@cython.exceptval(-2, check=True)
def city_state_coord(df):

    df['Full_Address'] = df['city'].str.cat(
        df[['state']], sep=' ') + ' ' + 'USA'
    df['Url'] = [
        'https://www.google.com/maps/search/' +
        i for i in df['Full_Address']]
    Url_With_Coordinates = []

    option = webdriver.ChromeOptions()
    prefs = {
        'profile.default_content_setting_values': {
            'images': 2,
            'javascript': 2}}
    option.add_experimental_option('prefs', prefs)

    driver = webdriver.Chrome(options=option)

    for url in tqdmn(df.Url, leave=False):
        time.sleep(0.2)
        driver.get(url)
        Url_With_Coordinates.append(
            driver.find_element(
                By.CSS_SELECTOR,
                'meta[itemprop=image]').get_attribute('content'))

    driver.close()

    with open('Url_With_Coordinates_temp.csv', 'w') as file:
        wr = csv.writer(file)
        wr.writerow(Url_With_Coordinates)

    with open('Url_With_Coordinates_temp.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i in reader:
            Url_With_Coordinates = i
            break

    df['Url_With_Coordinates'] = Url_With_Coordinates
    df = df[df.Url_With_Coordinates.str.contains('&zoom=')]
    df = df[df.Url_With_Coordinates.str.contains('center=')]
    df['lat'] = [url.split('?center=')[1].split('&zoom=')[0].split('%2C')[
        0] for url in df['Url_With_Coordinates']]
    df['long'] = [url.split('?center=')[1].split('&zoom=')[0].split('%2C')[
        1] for url in df['Url_With_Coordinates']]

    df = df.drop(['Url', 'Full_Address', 'Url_With_Coordinates'], axis=1)

    return df




@cython.cfunc
@cython.exceptval(-2, check=True)
def length_city_city(lat_1, long_1, lat_2, long_2):

    if (lat_1 is None) or (long_1 is None) or (
            lat_2 is None) or (long_2 is None):
        return None

    length = geodesic((lat_1, long_1), (lat_2, long_2), ellipsoid='WGS-84').km

    return length




@cython.cfunc
@cython.exceptval(-2, check=True)
def filtr_less(x, quantity):

    if x is None:
        return None
    if pd.isna(x):
        return None
    if x == 'del_data':
        return 'del_data'
    if x < quantity:
        return x

    return 'del_data'




@cython.cfunc
@cython.exceptval(-2, check=True)
def filtr_more(x, quantity):

    if x is None:
        return None

    if pd.isna(x):
        return None

    if x == 'del_data':
        return 'del_data'

    if x > quantity:
        return x

    return 'del_data'




@cython.cfunc
@cython.exceptval(-2, check=True)
def my_mode(x):

    moda, _ = mode(x, axis=0, nan_policy='omit', keepdims=False)

    return np.round(moda, 2)
