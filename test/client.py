
import pandas as pd
import requests

state = 'WA'
city = 'Newcastle'
zipcode = 98059
stories = 1
Year_built = 2015
sqft = 1000
lotsize = 1000
beds = 2
baths = 1
number_school = 5
school_distance_mean = 2
school_rating_mean = 5


# Следующие переменные булевые могут быть True/False или 0/1
Garage = False
Carport = False
Parking_lot = False
pool = False
fireplace = True
new = True
Remodeled = False
cool_bool = True
ceiling_fan = True
air_conditioner = True
heat_bool = True
central_heating = True
forced_air_heating = False
pump_heating = False
gas_heating = False
electric_heating = False
auction = False
foreclosure = False


# propertyType может быть только:
# 'condo',
#  'Townhouse',
#  'Single Family Home',
#  'Farms/Ranches',
#  'multi_family',
propertyType = 'condo'

df = pd.DataFrame(
    data=[[state, city, zipcode, stories, Year_built, sqft, lotsize, beds, baths,
           pool, fireplace, Garage, Carport, Parking_lot, new, Remodeled, cool_bool,
           ceiling_fan, air_conditioner, heat_bool, central_heating, forced_air_heating,
           pump_heating, gas_heating, electric_heating, auction, foreclosure, number_school,
           school_distance_mean, school_rating_mean, propertyType]],
    columns=['state', 'city', 'zipcode', 'stories', 'Year_built', 'sqft', 'lotsize',
             'beds', 'baths', 'pool', 'fireplace', 'Garage', 'Carport', 'Parking_lot',
             'new', 'Remodeled', 'cool_bool', 'ceiling_fan', 'air_conditioner',
             'heat_bool', 'central_heating', 'forced_air_heating', 'pump_heating',
             'gas_heating', 'electric_heating', 'auction', 'foreclosure', 'number_school',
             'school_distance_mean', 'school_rating_mean', 'propertyType'])

if __name__ == '__main__':
    # выполняем POST-запрос на сервер
    r = requests.post('http://localhost:5000/predict', json=df.to_json(orient='table'),

                      )
    # выводим статус запроса
    # print(r.status_code)
    # реализуем обработку результата
    if r.status_code == 200:
        # если запрос выполнен успешно (код обработки=200),
        # выводим результат на экран
        prediction_autoML = r.json()['prediction_autoML']
        prediction = r.json()['prediction']
        print(f'Предполагаемая цена, вычисленная lightautoml {prediction_autoML}')
        print(f'Предполагаемая цена, вычисленная RandomForestRegressor {prediction}')
    else:
        # если запрос завершён с кодом, отличным от 200,
        # выводим содержимое ответа
        print(r.text)
